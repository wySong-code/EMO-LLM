"""Four-label SFT with PEFT LoRA in the language, vision and audio blocks."""
import argparse
from collections import defaultdict
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import time
import traceback

import torch
from torch.nn import functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Trainer, TrainerCallback, TrainingArguments, set_seed

import train_omni_sft as original
import omni_modality_task as task
from train_omni_modality_sft import FullInputs, timestamp
from evaluate_omni_batched import evaluate


LANGUAGE = r'model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'


def group_name(name):
    if '.visual.blocks.' in '.' + name:
        return 'vision'
    if '.audio_tower.layers.' in '.' + name:
        return 'audio'
    if '.model.layers.' in '.' + name:
        return 'language'
    raise ValueError(f'Unexpected trainable parameter: {name}')


def attach_lora(base, out):
    targets = [name for name, module in base.named_modules()
               if isinstance(module, torch.nn.Linear) and
               (re.fullmatch(LANGUAGE, name) or name.startswith('visual.blocks.')
                or name.startswith('audio_tower.layers.'))]
    assert {group_name(n) for n in targets} == {'language', 'vision', 'audio'}
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
        bias='none', task_type='CAUSAL_LM', target_modules=targets))
    counts = defaultdict(int)
    names = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert 'lora_' in name
            counts[group_name(name)] += parameter.numel()
            names.append(name)
    original.write_json(out / 'trainable_parameters.json', {
        'count': sum(counts.values()), 'by_group': dict(counts),
        'target_modules': targets, 'parameters': names,
        'frozen': ['base weights', 'visual.merger', 'audio_tower.proj',
                   'convolutions', 'embeddings', 'norms', 'lm_head']})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    base = model.get_base_model()
    checkpointing = {group: all(getattr(layer, 'gradient_checkpointing', False) for layer in layers)
                    for group, layers in [('language', base.model.layers),
                                          ('vision', base.visual.blocks), ('audio', base.audio_tower.layers)]}
    original.write_json(out / 'gradient_checkpointing.json', checkpointing)
    assert all(checkpointing.values()), checkpointing
    return model


class BatchInputs(FullInputs):
    def collate(self, rows):
        items = [self.build(row, audio_first=bool((self.index[row['sample_id']] + self.epoch) % 2),
                            training=True) for row in rows]
        width = max(x['input_ids'].shape[1] for x in items)
        result = {'_field_positions': [item['_field_positions'] for item in items]}
        for key in set().union(*(set(x) for x in items)) - {'_field_positions'}:
            values = [x[key] for x in items if key in x]
            if key in ('input_ids', 'attention_mask', 'labels'):
                assert len(values) == len(items)
                fill = {'input_ids': self.processor.tokenizer.pad_token_id,
                        'attention_mask': 0, 'labels': -100}[key]
                result[key] = torch.cat([F.pad(v, (0, width-v.shape[1]), value=fill) for v in values])
            elif all(torch.is_tensor(v) for v in values):
                if key in ('input_features', 'feature_attention_mask'):
                    maximum = max(v.shape[-1] for v in values)
                    values = [F.pad(v, (0, maximum-v.shape[-1])) for v in values]
                assert all(v.shape[1:] == values[0].shape[1:] for v in values), key
                result[key] = torch.cat(values)
            elif all(isinstance(v, list) for v in values):
                result[key] = [part for value in values for part in value]
            else:
                assert all(v == values[0] for v in values), key
                result[key] = values[0]
        assert result['video_grid_thw'].shape[0] == len(rows)
        if any(row['protocol'] == 'VA' for row in rows):
            assert result['input_features'].shape[0] == sum(row['protocol'] == 'VA' for row in rows)
        for i, item in enumerate(items):
            length = item['input_ids'].shape[1]
            assert torch.equal(result['input_ids'][i, :length], item['input_ids'][0])
            assert torch.equal(result['labels'][i, :length], item['labels'][0])
            assert (result['labels'][i, length:] == -100).all()
            assert torch.all(result['labels'][i][~result['attention_mask'][i].bool()] == -100)
        return result


def forward_loss(model, inputs):
    batch = {k: v for k, v in inputs.items() if not k.startswith('_') and k != 'labels'}
    labels = inputs['labels']
    outputs = model(**batch)
    # Match the old batch-1 objective: average each answer, then average examples.
    shifted = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(outputs.logits[:, :-1].float().reshape(-1, outputs.logits.shape[-1]),
                                  shifted.reshape(-1), ignore_index=-100, reduction='none')
    token_losses = token_losses.view_as(shifted)
    loss = (token_losses.sum(-1) / (shifted != -100).sum(-1)).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite SFT loss')
    fields = defaultdict(list)
    with torch.no_grad():
        for row, positions in enumerate(inputs['_field_positions']):
            for field, indices in positions.items():
                fields[field].append(token_losses[row, torch.tensor(indices, device=labels.device)-1].mean().item())
    return loss, outputs, fields


class MultimodalTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.field_losses = defaultdict(list)
        self.audio_in_step = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Encoder LoRA dropout must follow model.train()/eval(), unlike frozen-encoder SFT.
        if model.training:
            base = model.get_base_model()
            assert base.visual.training and base.audio_tower.training
        self.audio_in_step |= 'input_features' in inputs
        loss, outputs, fields = forward_loss(model, inputs)
        for field, values in fields.items():
            self.field_losses[field].extend(values)
        return (loss, outputs) if return_outputs else loss


def gradient_stats(model, expect_audio=True):
    groups = defaultdict(list)
    frozen_gradients = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None:
                frozen_gradients.append(name)
        elif p.grad is not None:
            g = p.grad.detach().float()
            groups[group_name(name)].append(g.square().sum())
    assert not frozen_gradients, frozen_gradients[:3]
    result = {}
    for group in ('language', 'vision', 'audio'):
        norm = torch.stack(groups[group]).sum().sqrt().item() if groups[group] else 0.
        assert math.isfinite(norm), (group, norm)
        if group != 'audio' or expect_audio:
            assert norm > 0, f'Missing {group} gradients'
        result[group] = {'norm': norm, 'parameters_with_grad': len(groups[group])}
    return result


def tracked_parameters(model):
    tracked = {}
    for name, p in model.named_parameters():
        if p.requires_grad and 'lora_B' in name:
            tracked.setdefault(group_name(name), (name, p, p.detach().cpu().clone()))
    assert len(tracked) == 3
    return tracked


def parameter_updates(tracked):
    return {group: (p.detach().float().cpu() - before.float()).abs().sum().item()
            for group, (_, p, before) in tracked.items()}


def pilot(model_path, root, out):
    out.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    torch.set_num_threads(4)
    train, val = original.clean_split(root, out)
    assert (len(train), len(val)) == (3352, 666)
    buckets = {(protocol, kind): next(r for r in train if r['protocol'] == protocol and r['sample_type'] == kind)
               for protocol in ('VT', 'VA') for kind in ('Aligned', 'Conflict')}
    rows = list(buckets.values())
    original.write_json(out / 'samples.json', [{'sample_id': r['sample_id'], 'protocol': r['protocol'],
                                             'sample_type': r['sample_type']} for r in rows])
    processor = AutoProcessor.from_pretrained(model_path)
    inputs = BatchInputs(processor, root, root.parent / 'media_cache', train, audit_path=out / 'input_checks.jsonl')
    model = attach_lora(original.load_base(model_path, out), out)
    model.train()
    tracked = tracked_parameters(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    batches = [('mixed', rows), ('VT', [r for r in train if r['protocol'] == 'VT'][:4]),
               ('VA', [r for r in train if r['protocol'] == 'VA'][:4])]
    checks = []
    for name, sample in batches:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        batch = original.on_device(inputs.collate(sample))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, result, fields = forward_loss(model, batch)
        loss.backward()
        gradients = gradient_stats(model, expect_audio=name != 'VT')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        torch.cuda.synchronize()
        check = {'batch': name, 'n': len(sample), 'loss': loss.item(), 'gradients': gradients,
                 'seconds': time.monotonic()-start, 'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30,
                 'shapes': {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)}}
        checks.append(check)
        original.write_json(out / 'batch_checks.json', checks)
        print('PILOT_BATCH', json.dumps(check), flush=True)
        del loss, result, batch
    updates = parameter_updates(tracked)
    assert all(x > 0 for x in updates.values())
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    model.gradient_checkpointing_disable()
    batch = original.on_device(inputs.collate(rows))
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
    max_diff = (before.float()-after.float()).abs().max().item()
    assert torch.equal(before, after), max_diff
    summary = {'passed': True, 'batch_size': 4, 'batches': checks, 'adapter_updates_l1': updates,
               'save_reload_max_logit_difference': max_diff, 'updated_utc': timestamp(),
               'scope': 'Implementation check on training rows; pilot weights are discarded before formal SFT.'}
    original.write_json(out / 'summary.json', summary)
    original.write_json(out / 'status.json', {'stage': 'complete', 'passed': True, 'updated_utc': timestamp()})
    print('PILOT_PASSED', json.dumps(summary), flush=True)


class Progress(TrainerCallback):
    def __init__(self, out, inputs, val, run, eval_batch):
        self.out, self.inputs, self.val, self.run, self.eval_batch = out, inputs, val, run, eval_batch
        self.trainer = None
        self.checkpoints = []
        self.gradients = {}
        self.step_started = None
        self.tracked = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.inputs.epoch = int(round(state.epoch or 0))

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_started = time.monotonic()

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        self.gradients = gradient_stats(model, self.trainer.audio_in_step)
        self.trainer.audio_in_step = False

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 3:
            updates = parameter_updates(self.tracked)
            assert all(x > 0 for x in updates.values()), updates
            original.write_json(self.out / 'startup_check.json', {
                'global_step': state.global_step, 'adapter_updates_l1': updates,
                'gradients': self.gradients, 'updated_utc': timestamp()})

    def on_log(self, args, state, control, logs=None, **kwargs):
        values = {'step': state.global_step, 'epoch': state.epoch, **(logs or {}),
                  'step_seconds': time.monotonic()-self.step_started if self.step_started else None,
                  'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30,
                  'updated_utc': timestamp()}
        for group, stat in self.gradients.items():
            values[f'gradient_norm/{group}'] = stat['norm']
        for field, losses in self.trainer.field_losses.items():
            if losses:
                values[f'label_loss/{field}'] = sum(losses)/len(losses)
        self.trainer.field_losses.clear()
        with (self.out / 'train_metrics.jsonl').open('a') as f:
            f.write(json.dumps(values)+'\n')
        original.write_json(self.out / 'status.json', {'stage': 'training', **values})
        print('TRAIN', json.dumps(values), flush=True)
        if self.run:
            self.run.log({'training/global_step': state.global_step, **{k:v for k,v in values.items() if isinstance(v,(int,float))}})

    def on_save(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
        for name in ('adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json'):
            assert (checkpoint / name).is_file(), checkpoint / name
        self.inputs.processor.save_pretrained(checkpoint)
        entry = {'epoch': epoch, 'global_step': state.global_step, 'path': str(checkpoint)}
        self.checkpoints.append(entry)
        original.write_json(self.out / 'checkpoints.json', self.checkpoints)
        metrics, _ = evaluate(model, self.inputs, self.val, self.out, f'epoch_{epoch:02d}_val',
                              self.run, state.global_step, batch_size=self.eval_batch)
        entry['validation_metrics'] = metrics
        original.write_json(self.out / 'checkpoints.json', self.checkpoints)
        if self.run:
            values = {'evaluation/global_step': state.global_step}
            for group, stats in metrics.items():
                if isinstance(stats, dict) and stats.get('n'):
                    for key in ('local_score_present', 'emotion_macro_f1_present', 'conflict_recall', 'aligned_specificity'):
                        if stats.get(key) is not None:
                            values[f'validation/{group}/{key}'] = stats[key]
            self.run.log(values)


def train(args):
    out = Path(args.output)
    summary = json.loads((Path(args.pilot) / 'summary.json').read_text())
    assert summary['passed']
    out.mkdir(parents=True, exist_ok=False)
    original.write_json(out / 'status.json', {'stage': 'preparing', 'updated_utc': timestamp()})
    set_seed(42)
    torch.set_num_threads(4)
    root = Path(args.dataset)
    rows, val = original.clean_split(root, out)
    assert (len(rows), len(val)) == (3352, 666)
    for row in rows + val:
        task.target(row)
    config = {**vars(args), 'model_id': 'Qwen/Qwen2.5-Omni-3B', 'initial_adapter': None,
              'framework': 'Transformers Trainer + PEFT', 'train_records': len(rows), 'val_records': len(val),
              'micro_batch': 4, 'gradient_accumulation_steps': 4, 'effective_batch': 16,
              'learning_rate': 1e-4, 'scheduler': 'cosine', 'warmup_ratio': .03,
              'weight_decay': .01, 'max_grad_norm': 1., 'lora_rank': 16, 'lora_alpha': 32, 'lora_dropout': .05,
              'seed': 42, 'frames': 8, 'max_side': 448, 'audio_rate': 16000, 'max_length': 4096,
              'max_new_tokens': 192, 'dtype': 'bfloat16', 'attention': 'sdpa', 'epochs': 3,
              'supervision': ['views.M1.label', 'views.M2.label', 'sample_type', 'views.M12.label'],
              'loss': 'Full answer next-token CE, equal per-example mean; prompt and padding masked.',
              'prompt_orders': 'Alternated by epoch and original row index, independent of protocol.',
              'prompts': {'VT_first': task.system_prompt(False), 'VA_first': task.system_prompt(True)},
              'gpu': torch.cuda.get_device_name(0), 'started_utc': timestamp(),
              'versions': {k: importlib.metadata.version(k) for k in ('torch','transformers','peft','accelerate','wandb')}}
    original.write_json(out / 'config.json', config)
    shutil.copyfile(Path(args.pilot) / 'summary.json', out / 'pilot_summary.json')
    source_dir = out / 'scripts'
    source_dir.mkdir()
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(p, source_dir / p.name)
    processor = AutoProcessor.from_pretrained(args.model)
    inputs = BatchInputs(processor, root, root.parent / 'media_cache', rows, audit_path=out / 'input_checks.jsonl')
    base = original.load_base(args.model, out)
    if args.reference_adapter:
        reference = PeftModel.from_pretrained(base, args.reference_adapter, is_trainable=False)
        evaluate(reference, inputs, val, out, 'sft_matched_val', None, 0, batch_size=args.eval_batch)
        base = reference.unload()
        del reference
        set_seed(42)
    model = attach_lora(base, out)
    tracked = tracked_parameters(model)
    run = None
    if args.wandb_mode != 'disabled':
        import wandb
        run = wandb.init(project='MM-LLMs-Emotion', entity='youran09111001-beihang-university',
                         name=out.name, config=config, dir=str(out), mode=args.wandb_mode,
                         settings=wandb.Settings(save_code=False, disable_git=True, console='off'))
        run.define_metric('training/global_step')
        run.define_metric('evaluation/global_step')
        for key in ('loss','learning_rate','grad_norm','label_loss/*','gradient_norm/*','peak_gpu_gib','step_seconds','epoch'):
            run.define_metric(key, step_metric='training/global_step')
        run.define_metric('validation/*', step_metric='evaluation/global_step')
        original.write_json(out / 'wandb_run.json', {'id': run.id, 'url': run.url, 'entity': run.entity})
        print('WANDB_RUN_URL=' + str(run.url), flush=True)
    callback = Progress(out, inputs, val, run, args.eval_batch)
    callback.tracked = tracked
    training_args = TrainingArguments(
        output_dir=str(out / 'checkpoints'), per_device_train_batch_size=4, gradient_accumulation_steps=4,
        num_train_epochs=3, learning_rate=1e-4, lr_scheduler_type='cosine', warmup_ratio=.03,
        weight_decay=.01, bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, optim='adamw_torch', max_grad_norm=1.,
        logging_steps=1, save_strategy='epoch', save_total_limit=None, eval_strategy='no',
        dataloader_num_workers=0, remove_unused_columns=False, label_names=['labels'], report_to=[],
        seed=42, data_seed=42, disable_tqdm=True)
    trainer = MultimodalTrainer(model=model, args=training_args, train_dataset=original.Rows(rows),
                               data_collator=inputs.collate, callbacks=[callback])
    callback.trainer = trainer
    trainer.model_accepts_loss_kwargs = False
    trained = trainer.train()
    original.write_json(out / 'training_summary.json', {**trained.metrics, 'global_step': trainer.state.global_step,
                         'adapter_updates_l1': parameter_updates(tracked),
                         'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30})
    trainer.save_model(str(out / 'final_adapter'))
    processor.save_pretrained(out / 'final_adapter')
    assert len(callback.checkpoints) == 3
    best = max(callback.checkpoints, key=lambda c:(c['validation_metrics']['all']['local_score_present'], -c['epoch']))
    original.write_json(out / 'checkpoint_selection.json', best)
    shutil.copytree(best['path'], out / 'best_adapter',
                    ignore=shutil.ignore_patterns('optimizer.pt','scheduler.pt','rng_state.pth','training_args.bin'))
    original.write_json(out / 'status.json', {'stage': 'complete', 'global_step': trainer.state.global_step,
                        'best_epoch': best['epoch'], 'updated_utc': timestamp()})
    if run:
        run.summary['best_epoch'] = best['epoch']
        run.summary['best_validation_local_score'] = best['validation_metrics']['all']['local_score_present']
        run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('pilot','train'), required=True)
    parser.add_argument('--pilot')
    parser.add_argument('--reference-adapter')
    parser.add_argument('--eval-batch', type=int, default=16)
    parser.add_argument('--wandb-mode', choices=('online','offline','disabled'), default='online')
    args = parser.parse_args()
    try:
        if args.mode == 'pilot':
            pilot(args.model, Path(args.dataset), Path(args.output))
        else:
            train(args)
    except Exception:
        if Path(args.output).is_dir():
            original.write_json(Path(args.output) / 'failure.json', {'error': traceback.format_exc(), 'updated_utc': timestamp()})
            original.write_json(Path(args.output) / 'status.json', {'stage': 'failed', 'updated_utc': timestamp()})
        raise


if __name__ == '__main__':
    main()
