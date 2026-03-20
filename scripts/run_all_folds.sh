#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_all_folds.sh [config]
# Example:
#   bash scripts/run_all_folds.sh configs/qwen35_9b_lora.yaml

CONFIG_PATH="${1:-configs/qwen35_9b_lora.yaml}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-3600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("hf_transfer") else 1)
PY
then
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
else
  export HF_HUB_ENABLE_HF_TRANSFER=0
fi

for fold in 0 1 2 3 4; do
  echo "===== train fold ${fold} ====="
  python scripts/lora_qwen35_pipeline.py train-fold \
    --config "${CONFIG_PATH}" \
    --fold "${fold}"
done

python scripts/summarize_cv.py --reports-dir reports --prefix qwen35_lora_fold
