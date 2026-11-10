"""Fresh-base language-LoRA SFT on full and single-modality course views."""
import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import random
import re
import time
import traceback

import av
import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoProcessor, get_cosine_schedule_with_warmup, set_seed

from baseline import ensure_audio, prepare_media
from evaluate_omni_batched import evaluate
from prepare_unimodal_sft_data import GROUP_MASS, SINGLE_PROMPT, read_jsonl
from train_omni_modality_sft import FullInputs, timestamp
import train_omni_sft as original

LANGUAGE = r'model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'


def append(path, value):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')


class Inputs:
    def __init__(self, processor, root, cache, full_rows, pool, out):
        self.processor, self.root, self.cache = processor, root, cache
        self.full_rows = {r['sample_id']: r for r in full_rows}
        self.full = FullInputs(processor, root, cache, full_rows, audit_path=out / 'full_input_checks.jsonl')
        self.index = {r['example_id']: i for i, r in enumerate(pool)}
        self.audio_cache = cache / 'unimodal_audio'
        self.audio_cache.mkdir(exist_ok=True, parents=True)
        self.out, self.epoch = out, 0

    def build(self, row):
        source = row['inputs']
        if row['task'] == 'full':
            self.full.epoch = self.epoch
            item = self.full.build(self.full_rows[row['source_sample_ids'][0]],
                audio_first=bool((self.index[row['example_id']] + self.epoch) % 2), training=True)
        else:
            content = []
            kwargs = {'return_tensors': 'pt'}
            if 'video' in source:
                video, _, meta = prepare_media(self.root / source['video'], self.cache, 8, 448)
                with av.open(str(video)) as decoded:
                    frames = np.stack([f.to_ndarray(format='rgb24') for f in decoded.decode(video=0)])
                duration = json.loads(meta.read_text())['duration']
                content.append({'type': 'video'})
                kwargs.update(videos=[frames], videos_kwargs={'do_sample_frames': False,
                    'fps': len(frames) / duration, 'use_audio_in_video': False})
            if 'audio_from_video' in source:
                # Audio-only views never decode video frames or receive visual features.
                audio = self.audio_cache / (row['example_id'].replace(':', '-') + '.wav')
                ensure_audio(self.root / source['audio_from_video'], audio)
                wave, rate = sf.read(audio, dtype='float32')
                assert rate == 16000 and wave.ndim == 1 and len(wave)
                assert np.isfinite(wave).all() and np.any(wave != 0)
                content.append({'type': 'audio'})
                kwargs.update(audio=[wave], audio_kwargs={'sampling_rate': 16000})
            content.append({'type': 'text', 'text': '请判断所提供内容的情感。'})
            if 'text' in source:
                content.append({'type': 'text', 'text': '待分析文字：\n' + source['text']})
            messages = [{'role': 'system', 'content': [{'type': 'text', 'text': SINGLE_PROMPT}]},
                        {'role': 'user', 'content': content}]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            assert row['example_id'] not in prompt
            item = dict(self.processor(text=[prompt], **kwargs))
            prefix = item['input_ids'].shape[1]
            answer = json.dumps(row['target'], separators=(',', ':'))
            encoded = self.processor.tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
            target = torch.tensor([encoded['input_ids'] + [self.processor.tokenizer.eos_token_id]])
            item['input_ids'] = torch.cat([item['input_ids'], target], dim=1)
            item['attention_mask'] = torch.cat([item['attention_mask'], torch.ones_like(target)], dim=1)
            item['labels'] = torch.cat([torch.full((1, prefix), -100), target], dim=1)
            positions = {}
            for field in row['target']:
                start, end = re.search('"' + field + '":"([^"]+)"', answer).span(1)
                positions[field] = [prefix + i for i, (a, b) in enumerate(encoded['offset_mapping'])
                                    if a < end and b > start]
            item['_field_positions'] = positions
        assert ('pixel_values_videos' in item) == ('video' in source)
        assert ('input_features' in item) == ('audio_from_video' in source)
        assert set(item['_field_positions']) == set(row['target'])
        for positions in item['_field_positions'].values():
            assert positions and all(item['labels'][0, p] != -100 for p in positions)
        assert 0 < int((item['labels'] != -100).sum()) < item['input_ids'].shape[1] <= 4096
        # Reconstruct the supervised answer, including all JSON label values.
        tokens = item['labels'][item['labels'] != -100].tolist()
        assert tokens[-1] == self.processor.tokenizer.eos_token_id
        assert json.loads(self.processor.decode(tokens[:-1])) == row['target']
        append(self.out / 'input_checks.jsonl', {'epoch_index': self.epoch,
            'example_id': row['example_id'], 'task': row['task'], 'tokens': item['input_ids'].shape[1],
            'supervised_tokens': len(tokens), 'video': 'pixel_values_videos' in item,
            'audio': 'input_features' in item, 'loss_weight': row['loss_weight'],
            'fields': {k: len(v) for k, v in item['_field_positions'].items()}})
        return item

    def collate(self, rows):
        items = [self.build(r) for r in rows]
        width = max(x['input_ids'].shape[1] for x in items)
        batch = {'_field_positions': [x['_field_positions'] for x in items]}
        for key in sorted(set().union(*(set(x) for x in items)) - {'_field_positions'}):
            values = [x[key] for x in items if key in x]
            if key in ('input_ids', 'attention_mask', 'labels'):
                assert len(values) == len(items)
                fill = {'input_ids': self.processor.tokenizer.pad_token_id, 'attention_mask': 0, 'labels': -100}[key]
                batch[key] = torch.cat([F.pad(v, (0, width-v.shape[1]), value=fill) for v in values])
            elif all(torch.is_tensor(v) for v in values):
                if key in ('input_features', 'feature_attention_mask'):
                    maximum = max(v.shape[-1] for v in values)
                    values = [F.pad(v, (0, maximum-v.shape[-1])) for v in values]
                assert all(v.shape[1:] == values[0].shape[1:] for v in values), key
                batch[key] = torch.cat(values)
            elif all(isinstance(v, list) for v in values):
                batch[key] = [part for value in values for part in value]
            else:
                assert all(v == values[0] for v in values), key
                batch[key] = values[0]
        for key, input_key in [('video_grid_thw', 'video'), ('input_features', 'audio_from_video')]:
            count = sum(input_key in r['inputs'] for r in rows)
            assert (key in batch) == bool(count)
            if count:
                assert batch[key].shape[0] == count
        for i, item in enumerate(items):
            n = item['input_ids'].shape[1]
            assert torch.equal(batch['input_ids'][i, :n], item['input_ids'][0])
            assert torch.equal(batch['labels'][i, :n], item['labels'][0])
            assert (batch['labels'][i, n:] == -100).all()
            assert (batch['labels'][i][~batch['attention_mask'][i].bool()] == -100).all()
        return batch


def attach(base, out):
    targets = [n for n, m in base.named_modules() if isinstance(m, torch.nn.Linear) and re.fullmatch(LANGUAGE, n)]
    assert targets
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
        target_modules=targets, bias='none', task_type='CAUSAL_LM'))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n and '.model.layers.' in n for n, _ in trainable)
    assert all(p.dtype == torch.float32 for _, p in trainable)
    assert all(torch.count_nonzero(p).item() == 0 for n, p in trainable if 'lora_B' in n)
    original.write_json(out / 'trainable_parameters.json', {'count': sum(p.numel() for _, p in trainable),
        'targets': targets, 'names': [n for n, _ in trainable], 'initial_adapter': None,
        'frozen': ['base language weights', 'visual encoder', 'audio encoder', 'lm_head']})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    train_mode(model)
    return model


def train_mode(model):
    model.train()
    model.config.use_cache = False
    model.get_base_model().visual.eval()
    model.get_base_model().audio_tower.eval()


def answer_ce(logits, labels):
    """Select supervised positions before FP32 CE; equivalent to masked token CE."""
    gold = labels[:, 1:]
    valid = gold != -100
    count = valid.sum(1)
    assert torch.all(count > 0)
    losses = F.cross_entropy(logits[:, :-1][valid].float(), gold[valid], reduction='none')
    by_row = torch.zeros(labels.shape[0], device=logits.device, dtype=losses.dtype)
    by_row = by_row.index_add(0, valid.nonzero()[:, 0], losses) / count
    matrix = torch.zeros_like(gold, dtype=losses.dtype).masked_scatter(valid, losses)
    return by_row, matrix


def loss_for(model, cpu, rows, group_size):
    batch = original.on_device(cpu)
    args = {k: v for k, v in batch.items() if not k.startswith('_') and k != 'labels'}
    outputs = model(**args)
    means, tokens = answer_ce(outputs.logits, batch['labels'])
    weights = torch.tensor([r['loss_weight'] for r in rows], device=means.device, dtype=torch.float32)
    loss = (means * weights).sum() / group_size
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite weighted SFT loss')
    fields = defaultdict(list)
    with torch.no_grad():
        for i, positions in enumerate(batch['_field_positions']):
            for field, indices in positions.items():
                fields[f"{rows[i]['task']}/{field}"].append(tokens[i, torch.tensor(indices, device=means.device)-1].mean().item())
    stats = {'loss': loss.item(), 'means': means.detach().tolist(), 'fields': dict(fields)}
    return loss, stats


def grad_norm(model):
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, name
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    assert math.isfinite(norm.item()) and norm.item() > 0
    return norm.item()


def check_loss_math():
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn((10, 8, 13), generator=generator, requires_grad=True)
    labels = torch.randint(13, (10, 8), generator=generator)
    labels[:, :3] = -100
    labels[::2, -2:] = -100
    means, _ = answer_ce(logits, labels)
    direct = torch.stack([F.cross_entropy(logits[i, :-1], labels[i, 1:], ignore_index=-100) for i in range(10)])
    torch.testing.assert_close(means, direct)
    weights = torch.tensor([1.7, .46, .57, .62, 1.7, .46, .57, .62, 1.7, .46])
    loss = (means * weights).mean()
    exact_grad = torch.autograd.grad(loss, logits)[0]
    # The final optimizer group has 4 + 4 + 2 rows, normalized by 10.
    split_logits = logits.detach().clone().requires_grad_(True)
    for first in range(0, 10, 4):
        part, _ = answer_ce(split_logits[first:first+4], labels[first:first+4])
        ((part * weights[first:first+4]).sum() / 10).backward()
    torch.testing.assert_close(exact_grad, split_logits.grad)
    return {'selected_ce_matches_full_masked_ce': True, 'tail_4_4_2_gradient_matches_batch10': True}


def load_data(args, out):
    pool = read_jsonl(args.data / 'train_pool.jsonl')
    raw = read_jsonl(args.dataset / 'train.jsonl')
    val = read_jsonl(args.dataset / 'val.jsonl')
    raw_by_id = {r['sample_id']: r for r in raw}
    full_rows = [raw_by_id[r['source_sample_ids'][0]] for r in pool if r['task'] == 'full']
    assert len(full_rows) == 3352 and len(pool) == 8730 and len(val) == 666
    counts = Counter(r['task'] for r in pool)
    assert counts == {'full': 3352, 'vision_only': 2103, 'text_only': 1702, 'audio_only': 1573}
    for task, mass in GROUP_MASS.items():
        assert abs(sum(r['loss_weight'] for r in pool if r['task'] == task) / len(pool) - mass) < 1e-10
    processor = AutoProcessor.from_pretrained(args.model)
    inputs = Inputs(processor, args.dataset, args.cache, full_rows, pool, out)
    return pool, val, processor, inputs


def pilot(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    original.write_json(out / 'status.json', {'stage': 'loading_preflight', 'updated_utc': timestamp()})
    math_checks = check_loss_math()
    pool, _, processor, inputs = load_data(args, out)
    model = attach(original.load_base(args.model, out), out)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    by_task = {k: [r for r in pool if r['task'] == k] for k in GROUP_MASS}
    vt = next(r for r in by_task['full'] if r['protocol'] == 'VT')
    va = next(r for r in by_task['full'] if r['protocol'] == 'VA')
    batches = [('VT_V_T_A', [vt, by_task['vision_only'][0], by_task['text_only'][0], by_task['audio_only'][0]]),
               ('VA_V_T_A', [va, by_task['vision_only'][1], by_task['text_only'][1], by_task['audio_only'][1]]),
               ('full_VA', [r for r in by_task['full'] if r['protocol'] == 'VA'][:4]),
               ('text_only', by_task['text_only'][:4]), ('audio_only', by_task['audio_only'][:4])]
    tracked = next(p for n, p in model.named_parameters() if p.requires_grad and 'lora_B' in n)
    initial = tracked.detach().cpu().clone()
    checks = []
    for name, rows in batches:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        cpu = inputs.collate(rows)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, stats = loss_for(model, cpu, rows, len(rows))
        loss.backward()
        norm = grad_norm(model)
        optimizer.step()
        torch.cuda.synchronize()
        check = {'name': name, 'n': len(rows), 'loss': stats['loss'], 'gradient_norm': norm,
                 'seconds': time.monotonic()-start, 'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30,
                 'shapes': {k: list(v.shape) for k, v in cpu.items() if torch.is_tensor(v)}}
        checks.append(check)
        original.write_json(out / 'batch_checks.json', checks)
        print('PREFLIGHT_BATCH', json.dumps(check), flush=True)
        del loss, cpu
    delta = (tracked.detach().cpu()-initial).abs().sum().item()
    assert delta > 0
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    model.gradient_checkpointing_disable()
    cpu = inputs.collate(batches[1][1])
    batch = original.on_device(cpu)
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
    summary = {'passed': True, 'micro_batch': 4, 'math': math_checks, 'batches': checks,
               'adapter_update_l1': delta, 'save_reload_max_logit_difference': (before-after).abs().max().item(),
               'formal_initialization': 'Fresh base and fresh LoRA; discard all pilot weights', 'updated_utc': timestamp()}
    original.write_json(out / 'summary.json', summary)
    original.write_json(out / 'status.json', {'stage': 'complete', 'passed': True, 'updated_utc': timestamp()})
    print('PREFLIGHT_PASSED', json.dumps(summary), flush=True)


def save_checkpoint(model, processor, optimizer, scheduler, out, step, epoch):
    checkpoint = out / 'checkpoints' / f'checkpoint-{step}'
    checkpoint.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint)
    processor.save_pretrained(checkpoint)
    saved = load_file(checkpoint / 'adapter_model.safetensors')
    current = get_peft_model_state_dict(model)
    assert set(saved) == set(current)
    assert all(torch.equal(saved[k], current[k].detach().cpu()) for k in saved)
    original.write_json(checkpoint / 'save_verification.json', {'all_adapter_tensors_equal': True,
        'tensor_count': len(saved), 'updated_utc': timestamp()})
    torch.save(optimizer.state_dict(), checkpoint / 'optimizer.pt')
    torch.save(scheduler.state_dict(), checkpoint / 'scheduler.pt')
    torch.save({'python': random.getstate(), 'numpy': np.random.get_state(), 'cpu': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all()}, checkpoint / 'rng_state.pth')
    state = {'global_step': step, 'epoch': epoch, 'path': str(checkpoint), 'updated_utc': timestamp()}
    original.write_json(checkpoint / 'trainer_state.json', state)
    original.write_json(checkpoint / 'experiment_config.json', json.loads((out / 'config.json').read_text()))
    return state


def eval_log(metrics, step, epoch):
    logs = {'training/global_step': step, 'validation/epoch': epoch}
    for group in ('all', 'VT', 'VA', 'original', 'generated'):
        m = metrics[group]
        for key in ('local_score_present', 'emotion_accuracy', 'conflict_recall', 'aligned_specificity', 'all_fields_correct_n'):
            if m[key] is not None:
                logs[f'validation/{group}/{key}'] = m[key]
        for field, v in m['fields'].items():
            logs[f'validation/{group}/{field}_accuracy'] = v['accuracy']
        for key, v in m['conflict_confusion'].items():
            logs[f'validation/{group}/conflict_{key}'] = v
    return logs


def train(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    assert json.loads((args.pilot / 'summary.json').read_text())['passed']
    set_seed(42)
    pool, val, processor, inputs = load_data(args, out)
    steps_per_epoch = math.ceil(len(pool) / 16)
    total_steps = steps_per_epoch * args.epochs
    config = {'model': args.model, 'model_id': 'Qwen/Qwen2.5-Omni-3B', 'initial_adapter': None,
        'dataset': str(args.dataset), 'data': str(args.data), 'epochs': args.epochs,
        'source_train_records': 3352, 'mixed_examples_per_epoch': len(pool), 'validation_records': len(val),
        'micro_batch': 4, 'gradient_accumulation_steps': 4, 'effective_batch': 16,
        'optimizer_steps_per_epoch': steps_per_epoch, 'total_optimizer_steps': total_steps,
        'learning_rate': 1e-4, 'scheduler': 'cosine', 'warmup_ratio': .03, 'warmup_steps': math.ceil(total_steps*.03),
        'weight_decay': .01, 'max_grad_norm': 1., 'seed': 42,
        'lora_rank': 16, 'lora_alpha': 32, 'lora_dropout': .05, 'trainable': 'language LoRA only',
        'dtype': 'bfloat16 base, float32 LoRA', 'attention': 'sdpa', 'gradient_checkpointing': 'non-reentrant',
        'loss': 'Fixed task weight times mean assistant-token CE; mean over actual optimizer-group examples',
        'group_mean_weights': GROUP_MASS, 'tail': '4+4+2 examples / 10; no dropping or repeating',
        'frames': 8, 'max_side': 448, 'audio_rate': 16000, 'max_length': 4096,
        'eval_batch': args.eval_batch, 'max_new_tokens': 192, 'decode': 'greedy',
        'checkpoint_selection': 'Highest full-validation local score; ties choose earliest epoch',
        'started_utc': timestamp(), 'gpu': torch.cuda.get_device_name(0),
        'versions': {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'peft', 'wandb')}}
    original.write_json(out / 'config.json', config)
    original.write_json(out / 'status.json', {'stage': 'loading_fresh_base', 'updated_utc': timestamp()})
    model = attach(original.load_base(args.model, out), out)
    tracked = next(p for n, p in model.named_parameters() if p.requires_grad and 'lora_B' in n)
    initial = tracked.detach().cpu().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config['warmup_steps'], total_steps)
    import wandb
    run = wandb.init(project='MM-LLMs-Emotion', entity='youran09111001-beihang-university',
                     name='omni3b-unimodal-sft-fresh-e3-mb4-20261026',
                     config=config, dir=str(out), mode='online', tags=['sft', 'unimodal-auxiliary', 'fresh-base'])
    run.define_metric('training/global_step')
    run.define_metric('train/*', step_metric='training/global_step')
    run.define_metric('label_loss/*', step_metric='training/global_step')
    run.define_metric('validation/*', step_metric='training/global_step')
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
                weighted_loss = 0.
                micro_sizes = []
                for first in range(0, len(group), 4):
                    rows = group[first:first+4]
                    micro_sizes.append(len(rows))
                    cpu = inputs.collate(rows)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss, stats = loss_for(model, cpu, rows, len(group))
                    loss.backward()
                    weighted_loss += stats['loss']
                    for row, value in zip(rows, stats['means']):
                        task_losses[row['task']].append(value)
                    for key, values in stats['fields'].items():
                        fields[key].extend(values)
                    seen.extend(r['example_id'] for r in rows)
                    del loss, cpu
                norm = grad_norm(model)
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
                values.update({f'task_loss/{k}': sum(v)/len(v) for k, v in task_losses.items()})
                values.update({f'label_loss/{k}': sum(v)/len(v) for k, v in fields.items()})
                append(out / 'train_metrics.jsonl', values)
                original.write_json(out / 'status.json', {'stage': 'training', **values})
                print('TRAIN', json.dumps(values), flush=True)
                logs = {'training/global_step': step}
                logs.update({('' if k.startswith('label_loss/') else 'train/') + k: v
                             for k, v in values.items() if isinstance(v, (float, int))})
                run.log(logs)
                if step == 3:
                    delta = (tracked.detach().cpu()-initial).abs().sum().item()
                    assert delta > 0
                    original.write_json(out / 'startup_check.json', {'step': step, 'adapter_update_l1': delta,
                        'gradient_norm': norm, 'initial_adapter': None, 'micro_batch': 4,
                        'effective_batch': 16, 'wandb_url': run.url, 'updated_utc': timestamp()})
            assert len(seen) == len(set(seen)) == len(pool) and set(seen) == set(inputs.index)
            original.write_json(out / f'epoch_{epoch+1}_coverage.json', {'examples': len(seen),
                'unique_examples': len(set(seen)), 'counts': dict(Counter(pool[i]['task'] for i in indices)),
                'all_examples_once': True, 'step': step})
            checkpoint = save_checkpoint(model, processor, optimizer, scheduler, out, step, epoch+1)
            checkpoints.append(checkpoint)
            original.write_json(out / 'checkpoints.json', checkpoints)
            metrics, _ = evaluate(model, inputs.full, val, out, f'epoch_{epoch+1}', run, step, args.eval_batch)
            run.log(eval_log(metrics, step, epoch+1))
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
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--eval-batch', type=int, default=16)
    args = parser.parse_args()
    assert args.epochs == 3 and args.eval_batch == 16
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    logging.getLogger().setLevel(logging.ERROR)
    (pilot if args.mode == 'pilot' else train)(args)


if __name__ == '__main__':
    main()
