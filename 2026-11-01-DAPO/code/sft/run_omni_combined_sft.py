"""Isolated combined-SFT supervisor; one preflight, one fresh training run."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--pilot-only', action='store_true')
    args = parser.parse_args()
    root = args.root
    root.mkdir(exist_ok=True, parents=True)
    lock = (root / 'job.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((root / 'launch_config.json').read_text())
    env = {**os.environ, 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4',
           'WANDB_BASE_URL': 'https://api.wandb.ai', 'WANDB_DIR': str(root),
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    common = [sys.executable, '-u', str(root / 'scripts/train_omni_combined_sft.py'),
              '--model', config['model'], '--dataset', config['dataset'],
              '--data', str(root/'data'), '--cache', config['media_cache']]
    state = {'supervisor_pid': os.getpid(), 'started_utc': stamp(), 'initial_adapter': None}

    def save(stage, **kw):
        state.update(stage=stage, updated_utc=stamp(), **kw)
        write(root/'job_status.json', state)
        print(json.dumps(state), flush=True)

    def run(stage, command):
        with (root/f'{stage}.log').open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            save(stage, child_pid=child.pid)
            while child.poll() is None:
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True)
                status = root/stage/'status.json'
                write(root/'health.json', {'updated_utc': stamp(), 'child_pid': child.pid, 'stage': stage,
                    'gpu': gpu.stdout.strip(), 'status': json.loads(status.read_text()) if status.is_file() else None})
                time.sleep(15)
            if child.returncode:
                raise RuntimeError(f'{stage} exited with code {child.returncode}; see {stage}.log')

    try:
        assert config['epochs'] == 3 and config['initial_adapter'] is None
        previous = json.loads(Path(config['previous_job_status']).read_text())
        assert previous['stage'] == 'complete', 'Previous encoder-LoRA run must finish first'
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                              '--format=csv,noheader,nounits'], text=True).strip()
        assert not processes, f'GPU already in use: {processes}'
        if not (root/'preflight/summary.json').exists():
            run('preflight', common + ['--mode', 'pilot', '--output', str(root/'preflight')])
        assert json.loads((root/'preflight/summary.json').read_text())['passed']
        if args.pilot_only:
            save('preflight_passed')
            return
        run('train', common + ['--mode', 'train', '--output', str(root/'train'),
                              '--pilot', str(root/'preflight'), '--epochs', '3', '--eval-batch', '16'])
        assert json.loads((root/'train/status.json').read_text())['stage'] == 'complete'
        save('complete')
    except Exception:
        save('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
