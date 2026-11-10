# Run audit — ~/grpo_mm/MM-Emotion-Combined-SFT-20261026/2026-10-29_23-22-GRPO/runs/pilot_01

## 1. Run identity

- out dir: `~/grpo_mm/MM-Emotion-Combined-SFT-20261026/2026-10-29_23-22-GRPO/runs/pilot_01`
- base: `~/grpo_mm/MM-Emotion-Combined-SFT-20261026/base_model`
- adapter (SFT start point): `~/grpo_mm/MM-Emotion-Combined-SFT-20261026/model_repo`
- policy device: cuda:0; reference: cuda:1
- attempts: 600, initial candidates 4, rescue 4
- temperature 1.25 / rescue 1.45, top_p 0.95
- lr 2e-06, kl 0.02, clip 0.2, ppo_epochs 2
- advantage scale decision 1.5 / other 0.5
- eval every 100, eval batch 4, patience 3
- guards: f1 drop 0.01, specificity drop 0.01
- environment: torch 2.8.0, transformers 4.57.6, peft 0.17.1
- elapsed: 2.26 h

## 2. Attempt accounting

| item | value |
|---|---|
| attempts run | 600 |
| candidates generated | 2496 |
| active groups updated | 193 |
| all_correct groups | 399 |
| all_same_not_perfect groups | 8 |
| groups whose decision field varied | 26 |
| invalid JSON candidates | 0 |
| optimizer steps | 386 |
| original conflicts covered | 143/143 |

## 3. Candidate reward distribution

- candidates: 2496; reward == 1.0: 2028/2496 = 0.8125
- mean reward: 0.9277
- distinct reward values: 5

| reward | count |
|---|---|
| 1.00 | 2028 |
| 0.90 | 5 |
| 0.65 | 396 |
| 0.45 | 54 |
| 0.10 | 13 |

## 4. Group outcomes

| initial status | count |
|---|---|
| all_correct | 399 |
| active | 177 |
| all_same_not_perfect | 24 |

| final status | count | share |
|---|---|---|
| all_correct | 399 | 0.6650 |
| active | 193 | 0.3217 |
| all_same_not_perfect | 8 | 0.0133 |

## 5. Sampling mix actually realised

| bucket | prompts | active groups | active rate |
|---|---|---|---|
| original_conflict | 450 | 184 | 0.4089 |
| generated_conflict | 30 | 6 | 0.2000 |
| aligned | 120 | 3 | 0.0250 |

## 6. Active rate by 100-attempt block

| block | prompts | active | all_correct | all_same_not_perfect | decision_varies |
|---|---|---|---|---|---|
| 1-100 | 100 | 47 | 53 | 0 | 9 |
| 101-200 | 100 | 37 | 61 | 2 | 5 |
| 201-300 | 100 | 36 | 64 | 0 | 3 |
| 301-400 | 100 | 26 | 70 | 4 | 2 |
| 401-500 | 100 | 21 | 78 | 1 | 3 |
| 501-600 | 100 | 26 | 73 | 1 | 4 |

## 7. Validation checkpoints (same 666 rows each time)

| attempt | local score | emotion macro-F1 | conflict recall | aligned specificity | eligible | improved |
|---|---|---|---|---|---|---|
| 0 (SFT baseline) | 0.8835 | 0.9419 | 0.7703 | 0.9071 | - | - |
| 100 | 0.8852 | 0.9387 | 0.7838 | 0.9037 | True | True |
| 200 | 0.8947 | 0.9415 | 0.8108 | 0.9037 | True | True |
| 300 | 0.8959 | 0.9445 | 0.8108 | 0.9020 | True | True |
| 400 | 0.8937 | 0.9475 | 0.7973 | 0.9037 | True | False |
| 500 | 0.8882 | 0.9461 | 0.7838 | 0.9003 | True | False |
| 600 | 0.8840 | 0.9463 | 0.7703 | 0.8986 | True | False |

## 8. Conflict recall split by source

| attempt | original recall | generated recall | format valid rate |
|---|---|---|---|
| 0100 | 0.4400 (n=25) | 0.9592 (n=49) | 1.0000 |
| 0200 | 0.5200 (n=25) | 0.9592 (n=49) | 1.0000 |
| 0300 | 0.5200 (n=25) | 0.9592 (n=49) | 1.0000 |
| 0400 | 0.4800 (n=25) | 0.9592 (n=49) | 1.0000 |
| 0500 | 0.4400 (n=25) | 0.9592 (n=49) | 1.0000 |
| 0600 | 0.4400 (n=25) | 0.9388 (n=49) | 1.0000 |

## 9. Selected checkpoint

- source: `runs/pilot_01/checkpoints/attempt_0300`
- attempt: 300
- score 0.8959, F1 0.9445, recall 0.8108108108108109, specificity 0.902027027027027

