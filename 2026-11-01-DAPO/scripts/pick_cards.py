"""Pick the least-loaded GPUs on the shared 8xGPU node.

Other users' jobs arrive and leave at random, so every launch re-reads nvidia-smi
instead of trusting a fixed card id. Exits non-zero when no card is quiet enough,
which lets the caller wait and retry rather than start into a doomed run.
"""
import subprocess
import sys

count = int(sys.argv[1]) if len(sys.argv) > 1 else 1
foreign_max = int(sys.argv[2]) if len(sys.argv) > 2 else 1200   # MiB another user may hold
rows = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    capture_output=True, text=True, check=True).stdout.strip().splitlines()
cards = []
for row in rows:
    index, used = (part.strip() for part in row.split(","))
    cards.append((int(used), int(index)))
admissible = sorted(c for c in cards if c[0] <= foreign_max)
if len(admissible) < count:
    print("no admissible cards: %s (limit %d MiB)" % (cards, foreign_max), file=sys.stderr)
    raise SystemExit(1)
print(",".join(str(index) for _, index in admissible[:count]))
