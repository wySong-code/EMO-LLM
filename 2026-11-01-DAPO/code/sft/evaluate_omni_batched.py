"""Greedy four-label evaluation with the already tested modality-aware batching."""
import json
import random
import time

import numpy as np
import torch

from sample_omni_dpo_data import build, generate, groups
import score_omni_modality as scoring
from train_omni_modality_sft import timestamp
from train_omni_sft import write_json


def evaluate(model, inputs, rows, out, name, run, step, batch_size=16):
    assert batch_size > 0 and rows
    previous_cache, previous_training = getattr(model.config, 'use_cache', True), model.training
    python_state, numpy_state = random.getstate(), np.random.get_state()
    records = []
    started = time.monotonic()
    model.eval()
    model.config.use_cache = True
    try:
        # The shared generation helper seeds RNGs. Evaluation must not change
        # the following epoch's sampling or training randomness.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            with (out / f'{name}_predictions.jsonl').open('x', encoding='utf-8') as handle:
                for batch in groups(rows, batch_size):
                    batch_rows = [row for _, row in batch]
                    items = [build(inputs, row) for row in batch_rows]
                    inputs.audit.clear()
                    answers = generate(model, inputs.processor, items, batch_rows, False, 20261026)
                    for answer in answers:
                        answer['prompt_order'] = 'VT_first'
                        records.append(answer)
                        handle.write(json.dumps(answer, ensure_ascii=False) + '\n')
                    handle.flush()
                    del items, answers
                    status = {'stage': 'evaluating', 'evaluation': name, 'completed': len(records),
                              'n': len(rows), 'global_step': step, 'batch_size': batch_size,
                              'updated_utc': timestamp()}
                    write_json(out / 'status.json', status)
                    print(json.dumps(status), flush=True)
        assert [r['sample_id'] for r in records] == [r['sample_id'] for r in rows]
        metrics = scoring.score(records)
        metrics.update(seconds=time.monotonic() - started, batch_size=batch_size)
        write_json(out / f'{name}_metrics.json', metrics)
        print('EVALUATION', name, json.dumps(metrics['all']), flush=True)
        return metrics, records
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        model.config.use_cache = previous_cache
        model.train(previous_training)
