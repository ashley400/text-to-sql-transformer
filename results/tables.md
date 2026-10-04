# Results Tables

## Table 1 – Data

| | Train | Dev | Test |
|---|---|---|---|
| Pairs | 56,355 | 8,421 | 15,878 |
| Mean / max source length (tokens) | 42.5 / 222 | 42.5 / 167 | 42.7 / 260 |
| Mean / max target length (tokens) | 14.8 / 65 | 14.8 / 44 | 14.9 / 46 |
| Pairs dropped as too long | 19 | - | - |

## Table 2 – Model and training

| Item | Value |
|---|---|
| Trainable parameters | 7,577,600 |
| Epochs trained / best epoch | 20 / 18 |
| Best dev loss | 1.5194 (label smoothing 0.1) |
| Training time and GPU | ~21.0 min, Tesla T4 (Colab) |

## Table 3 – Official metrics

| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |
|---|---|---|---|---|
| Dev | greedy | 51.78 | 59.47 | 0.56 |
| Dev | beam (4) | 51.67 | 59.49 | 0.52 |
| Test | beam (4) | 52.16 | 59.26 | 0.49 |

Gold round-trip (dev): execution 99.49%, logical form 99.49%, 0 parse failures.

## Table 4 – Component accuracy (dev, greedy decoding)

| Component | Accuracy (%) |
|---|---|
| sel column correct | 81.01 |
| agg correct | 88.77 |
| WHERE clause correct | 64.22 |