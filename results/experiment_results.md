# Experiment Records

All records below are successful end-to-end runs (train + fold validation), collected from cloud logs.

| Run ID | Date | Config | Fold | Precision | Recall | F1 | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `quick_smoke_fold0` | 2026-03-18 | `configs/qwen35_9b_lora_quick.yaml` | 0 | 0.5350 | 0.3661 | 0.4347 | Smoke test on subset |
| `base_full_fold0` | 2026-03-19 | `configs/qwen35_9b_lora.yaml` | 0 | 0.7719 | 0.7778 | 0.7748 | Full fold0, 2 epochs |
| `aggressive_full_fold0` | 2026-03-20 | `configs/qwen35_9b_lora_aggressive.yaml` | 0 | 0.7854 | 0.7968 | 0.7911 | Full fold0, 3 epochs, LoRA r=64 |

Current best verified fold score in this project: **0.7911** (`aggressive_full_fold0`).

Target `F1 >= 0.85` has **not** been reached yet on verified offline fold runs.

Available submission files (validated format):

- `results/submissions/submit_qwen35_lora_fold0.csv`
- `results/submissions/submit_qwen35_lora_aggr_fold0.csv`
