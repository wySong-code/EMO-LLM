"""Fresh Omni SFT: fixed unimodal mixture + language/vision/audio LoRA."""
import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import random
import time
import traceback

import torch
from transformers import get_cosine_schedule_with_warmup, set_seed

import train_omni_sft as original
import train_omni_unimodal_sft as shared
from prepare_unimodal_sft_data import GROUP_MASS, derive, read_jsonl, validate
from train_omni_modality_sft import timestamp
from train_omni_multimodal_sft import attach_lora, group_name, tracked_parameters, parameter_updates
from evaluate_omni_batched import evaluate


def train_mode(model):
    model.train()
    model.config.use_cache = False
    base = model.get_base_model()
    assert base.model.training and base.visual.training and base.audio_tower.training
    assert all(m.training for n, m in model.named_modules() if 'lora_dropout' in n)


# LoRA counts measured on the published Qwen2.5-Omni-3B run. The vision tower and
# audio encoder are the *same* modules in 3B and 7B, so those two counts must match
# exactly; only the language stack changes with the bigger text model.
REFERENCE_3B = {'language': 29933568, 'vision': 12462080, 'audio': 11796480}


def attach(base, out):
    model = attach_lora(base, out)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    counts = Counter()
    for n, p in params:
        assert 'lora_' in n and p.dtype == torch.float32
        counts[group_name(n)] += p.numel()
        if 'lora_B' in n:
            assert torch.count_nonzero(p).item() == 0, 'Must initialize fresh LoRA'
    # Encoders are architecturally identical across the 3B/7B pair.
    assert counts['vision'] == REFERENCE_3B['vision'], counts['vision']
    assert counts['audio'] == REFERENCE_3B['audio'], counts['audio']
    assert counts['language'] > 0
    original.write_json(out / 'trainable_group_counts.json',
                        {'measured': dict(counts), 'reference_3b': REFERENCE_3B,
                         'identical_encoders': True})
    print('TRAINABLE_GROUPS', json.dumps(dict(counts)), flush=True)
    train_mode(model)
    return model


def load_data(args, out):
    pool, val, processor, inputs = shared.load_data(args, out)
    manifest = read_jsonl(args.data / 'train_manifest.jsonl')
    val_manifest = read_jsonl(args.data / 'val_manifest.jsonl')
    full = [inputs.full_rows[r['source_sample_ids'][0]] for r in pool if r['task'] == 'full']
    assert [{'sample_id': r['sample_id'], 'video': r['video']} for r in full] == manifest
    assert [{'sample_id': r['sample_id'], 'video': r['video']} for r in val] == val_manifest
    derived = derive(full)
    assert pool == [r for part in derived.values() for r in part], 'Mixture differs from saved training views'
    checks = validate(derived, full, val, args.dataset)
    checks.update(saved_train_manifest_exact=True, saved_val_manifest_exact=True,
                  uploaded_pool_matches_course_views=True)
    original.write_json(out / 'data_checks.json', checks)
    return pool, val, processor, inputs


def grad_norm(model, rows):
    expected = {'language'}
    if any('video' in r['inputs'] for r in rows):
        expected.add('vision')
    if any('audio_from_video' in r['inputs'] for r in rows):
        expected.add('audio')
    groups = defaultdict(list)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, name
        elif p.grad is not None:
            groups[group_name(name)].append(p.grad.detach().float().square().sum())
    values = {}
    for group in ('language', 'vision', 'audio'):
        norm = torch.stack(groups[group]).sum().sqrt().item() if groups[group] else 0.
        assert math.isfinite(norm), (group, norm)
        assert (norm > 0) == (group in expected), (group, norm, expected)
        if group not in expected:
            assert not groups[group], f'Unexpected {group} graph for absent modality'
        values[group] = {'norm': norm, 'parameters_with_grad': len(groups[group])}
    total = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                         1., error_if_nonfinite=True).item()
    assert math.isfinite(total) and total > 0
    return total, values


def pilot(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    original.write_json(out / 'status.json', {'stage': 'loading_preflight', 'updated_utc': timestamp()})
    math_checks = shared.check_loss_math()
    pool, _, processor, inputs = load_data(args, out)
    model = attach(original.load_base(args.model, out), out)
    tracked = tracked_parameters(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    by_task = {k: [r for r in pool if r['task'] == k] for k in GROUP_MASS}
    vt = next(r for r in by_task['full'] if r['protocol'] == 'VT' and r['target']['is_conflict'] == 'yes')
    va = next(r for r in by_task['full'] if r['protocol'] == 'VA' and r['target']['is_conflict'] == 'yes')
    batches = [('VT_V_T_A', [vt, by_task['vision_only'][0], by_task['text_only'][0], by_task['audio_only'][0]]),
               ('VA_V_T_A', [va, by_task['vision_only'][1], by_task['text_only'][1], by_task['audio_only'][1]]),
               ('full_VT', [r for r in by_task['full'] if r['protocol'] == 'VT'][:4]),
               ('full_VA', [r for r in by_task['full'] if r['protocol'] == 'VA'][:4]),
               ('vision_only', by_task['vision_only'][:4]),
               ('text_only', by_task['text_only'][:4]), ('audio_only', by_task['audio_only'][:4])]
    checks = []
    for name, rows in batches:
        train_mode(model)
        optimizer.zero_grad(set_to_none=True)
        before_update = tracked_parameters(model)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        group_loss, last_shapes = 0., {}
        for first in range(0, len(rows), args.micro_batch):
            chunk = rows[first:first+args.micro_batch]
            cpu = inputs.collate(chunk)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, stats = shared.loss_for(model, cpu, chunk, len(rows))
            loss.backward()
            group_loss += stats['loss']
            last_shapes = {k: list(v.shape) for k, v in cpu.items() if torch.is_tensor(v)}
            del loss, cpu
        norm, gradients = grad_norm(model, rows)
        optimizer.step()
        torch.cuda.synchronize()
        changes = parameter_updates(before_update)
        for group in gradients:
            assert (changes[group] > 0) == (gradients[group]['norm'] > 0), (group, changes)
        check = {'name': name, 'n': len(rows), 'loss': group_loss, 'gradient_norm': norm,
                 'gradients': gradients, 'adapter_updates_l1': changes,
                 'seconds': time.monotonic()-start, 'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30,
                 'micro_batch': args.micro_batch, 'shapes': last_shapes}
        checks.append(check)
        original.write_json(out / 'batch_checks.json', checks)
        print('PREFLIGHT_BATCH', json.dumps(check), flush=True)
    updates = parameter_updates(tracked)
    assert all(v > 0 and math.isfinite(v) for v in updates.values())
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    model.gradient_checkpointing_disable()
    batch = original.on_device(inputs.collate(batches[1][1]))
    kwargs = {k: v for k, v in batch.items() if not k.startswith('_') and k != 'labels'}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        before = model(**kwargs).logits[:, -1].detach().cpu()
    model.save_pretrained(out / 'adapter')
    processor.save_pretrained(out / 'adapter')
    model.load_adapter(str(out / 'adapter'), adapter_name='reload_check', is_trainable=False)
    model.set_adapter('reload_check')
    model.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        after = model(**kwargs).logits[:, -1].detach().cpu()
    assert torch.equal(before, after)
    summary = {'passed': True, 'micro_batch': args.micro_batch, 'math': math_checks, 'batches': checks,
               'adapter_updates_l1': updates, 'save_reload_max_logit_difference': (before-after).abs().max().item(),
               'formal_initialization': 'Fresh base and fresh LoRA; discard all pilot weights', 'updated_utc': timestamp()}
    original.write_json(out / 'summary.json', summary)
    original.write_json(out / 'status.json', {'stage': 'complete', 'passed': True, 'updated_utc': timestamp()})
    print('PREFLIGHT_PASSED', json.dumps(summary), flush=True)


class _NullRun:
    """Stand-in for wandb: the shared node has no outbound access to wandb.ai."""

    def __init__(self):
        self.id, self.url, self.project, self.summary = 'disabled', '', 'MM-LLMs-Emotion', {}

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


def open_run(config, out, name):
    if os.environ.get('WANDB_MODE', 'disabled') != 'online':
        return _NullRun()
    import wandb
    return wandb.init(project='MM-LLMs-Emotion', name=name, config=config, dir=str(out),
                      mode='online', tags=['sft', 'unimodal-auxiliary', 'multimodal-lora', 'fresh-base'])


def train(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    assert json.loads((args.pilot / 'summary.json').read_text())['passed']
    set_seed(42)
    pool, val, processor, inputs = load_data(args, out)
    steps_per_epoch = math.ceil(len(pool) / 16)
    total_steps = steps_per_epoch * args.epochs
    config = {'model': args.model, 'model_id': args.model_id, 'initial_adapter': None,
        'dataset': str(args.dataset), 'data': str(args.data), 'epochs': args.epochs,
        'source_train_records': 3352, 'mixed_examples_per_epoch': len(pool), 'validation_records': len(val),
        'micro_batch': args.micro_batch, 'gradient_accumulation_steps': 16 // args.micro_batch,
        'effective_batch': 16,
        'optimizer_steps_per_epoch': steps_per_epoch, 'total_optimizer_steps': total_steps,
        'learning_rate': 1e-4, 'scheduler': 'cosine', 'warmup_ratio': .03, 'warmup_steps': math.ceil(total_steps*.03),
        'weight_decay': .01, 'max_grad_norm': 1., 'seed': 42,
        'lora_rank': 16, 'lora_alpha': 32, 'lora_dropout': .05,
        'trainable': 'language attention/MLP + vision blocks + audio encoder layers LoRA',
        'frozen': ['base weights', 'visual.merger', 'audio_tower.proj', 'convolutions', 'embeddings', 'norms', 'lm_head'],
        'dtype': 'bfloat16 base, float32 LoRA', 'attention': 'sdpa', 'gradient_checkpointing': 'non-reentrant',
        'loss': 'Fixed task weight times mean assistant-token CE; mean over actual optimizer-group examples',
        'group_mean_weights': GROUP_MASS, 'tail': f'tail {len(pool) % 16} examples; no dropping or repeating',
        'frames': 8, 'max_side': 448, 'audio_rate': 16000, 'max_length': 4096,
        'eval_batch': args.eval_batch, 'max_new_tokens': 192, 'decode': 'greedy',
        'checkpoint_selection': 'Highest full-validation local score; ties choose earliest epoch',
        'started_utc': timestamp(), 'gpu': torch.cuda.get_device_name(0),
        'versions': {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'peft')}}
    original.write_json(out / 'config.json', config)
    original.write_json(out / 'status.json', {'stage': 'loading_fresh_base', 'updated_utc': timestamp()})
    model = attach(original.load_base(args.model, out), out)
    tracked = tracked_parameters(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config['warmup_steps'], total_steps)
    run = open_run(config, out, name=f"{Path(args.model).name}-combined-sft-e{args.epochs}-mb{args.micro_batch}")
    run.define_metric('training/global_step')
    for namespace in ('train', 'label_loss', 'validation'):
        run.define_metric(namespace+'/*', step_metric='training/global_step')
    original.write_json(out / 'wandb.json', {'id': run.id, 'url': run.url, 'project': run.project})
    checkpoints, history = [], []
    step, started = 0, time.monotonic()
    try:
        for epoch in range(args.epochs):
            inputs.epoch = epoch
            indices = list(range(len(pool)))
            random.Random(42+epoch).shuffle(indices)
            original.write_json(out / f'epoch_{epoch+1}_order.json', [pool[i]['example_id'] for i in indices])
            seen = []
            for offset in range(0, len(indices), 16):
                train_mode(model)
                group = [pool[i] for i in indices[offset:offset+16]]
                group_start = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                task_losses, fields = defaultdict(list), defaultdict(list)
                weighted_loss, micro_sizes = 0., []
                for first in range(0, len(group), args.micro_batch):
                    rows = group[first:first+args.micro_batch]
                    micro_sizes.append(len(rows))
                    cpu = inputs.collate(rows)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss, stats = shared.loss_for(model, cpu, rows, len(group))
                    loss.backward()
                    weighted_loss += stats['loss']
                    for row, value in zip(rows, stats['means']):
                        task_losses[row['task']].append(value)
                    for key, values in stats['fields'].items():
                        fields[key].extend(values)
                    seen.extend(r['example_id'] for r in rows)
                    del loss, cpu
                norm, gradients = grad_norm(model, group)
                used_lr = optimizer.param_groups[0]['lr']
                optimizer.step()
                scheduler.step()
                step += 1
                torch.cuda.synchronize()
                values = {'step': step, 'total_steps': total_steps, 'epoch': epoch + len(seen)/len(pool),
                    'epoch_index': epoch+1, 'seen_in_epoch': len(seen), 'loss': weighted_loss,
                    'gradient_norm': norm, 'learning_rate': used_lr, 'group_examples': len(group),
                    'microbatch_sizes': micro_sizes, 'step_seconds': time.monotonic()-group_start,
                    'elapsed_seconds': time.monotonic()-started,
                    'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30, 'updated_utc': timestamp()}
                values.update({f'gradient_norm/{k}': v['norm'] for k, v in gradients.items()})
                values.update({f'task_loss/{k}': sum(v)/len(v) for k, v in task_losses.items()})
                values.update({f'label_loss/{k}': sum(v)/len(v) for k, v in fields.items()})
                shared.append(out / 'train_metrics.jsonl', values)
                original.write_json(out / 'status.json', {'stage': 'training', **values})
                print('TRAIN', json.dumps(values), flush=True)
                logs = {'training/global_step': step}
                logs.update({('' if k.startswith('label_loss/') else 'train/') + k: v
                             for k, v in values.items() if isinstance(v, (float, int))})
                run.log(logs)
                if step == 3:
                    updates = parameter_updates(tracked)
                    assert all(v > 0 and math.isfinite(v) for v in updates.values())
                    assert all(torch.isfinite(p).all().item() for p in model.parameters() if p.requires_grad)
                    original.write_json(out / 'startup_check.json', {'step': step, 'adapter_updates_l1': updates,
                        'gradient_norm': norm, 'gradients': gradients, 'initial_adapter': None, 'micro_batch': 4,
                        'effective_batch': 16, 'trainable_weights_finite': True,
                        'wandb_url': run.url, 'updated_utc': timestamp()})
            assert len(seen) == len(set(seen)) == len(pool) and set(seen) == set(inputs.index)
            original.write_json(out / f'epoch_{epoch+1}_coverage.json', {'examples': len(seen),
                'unique_examples': len(set(seen)), 'counts': dict(Counter(pool[i]['task'] for i in indices)),
                'all_examples_once': True, 'step': step})
            checkpoint = shared.save_checkpoint(model, processor, optimizer, scheduler, out, step, epoch+1)
            checkpoints.append(checkpoint)
            original.write_json(out / 'checkpoints.json', checkpoints)
            metrics, _ = evaluate(model, inputs.full, val, out, f'epoch_{epoch+1}', run, step, args.eval_batch)
            run.log(shared.eval_log(metrics, step, epoch+1))
            history.append({**checkpoint, 'metrics': metrics})
            original.write_json(out / 'evaluation_history.json', history)
            best = max(history, key=lambda x: x['metrics']['all']['local_score_present'])
            original.write_json(out / 'best_checkpoint.json', best)
        assert step == total_steps
        original.write_json(out / 'status.json', {'stage': 'complete', 'step': step, 'epochs': args.epochs,
            'best_checkpoint': best['path'], 'best_local_score': best['metrics']['all']['local_score_present'],
            'updated_utc': timestamp()})
        run.summary['best_checkpoint'] = best['path']
        run.summary['best_local_score'] = best['metrics']['all']['local_score_present']
        run.finish()
    except Exception:
        original.write_json(out / 'failure.json', {'step': step, 'error': traceback.format_exc(), 'updated_utc': timestamp()})
        run.finish(exit_code=1)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['pilot', 'train'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-id', default='Qwen/Qwen2.5-Omni-7B')
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--eval-batch', type=int, default=16)
    parser.add_argument('--micro-batch', type=int, default=4,
                        help='gradient accumulation keeps the effective batch at 16')
    args = parser.parse_args()
    assert args.epochs == 3
    assert 16 % args.micro_batch == 0 and 0 < args.micro_batch <= 16
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    logging.getLogger().setLevel(logging.ERROR)
    (pilot if args.mode == 'pilot' else train)(args)


if __name__ == '__main__':
    main()
