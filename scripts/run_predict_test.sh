#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_predict_test.sh [config] [adapter_dir] [output_csv]
# Example:
#   bash scripts/run_predict_test.sh configs/qwen35_9b_lora.yaml outputs/qwen35_lora_fold0/adapter outputs/submit_qwen35_fold0.csv

CONFIG_PATH="${1:-configs/qwen35_9b_lora.yaml}"
ADAPTER_DIR="${2:-}"
OUTPUT_CSV="${3:-outputs/submit_qwen35_lora.csv}"

if [[ -z "${ADAPTER_DIR}" ]]; then
  echo "[ERROR] missing adapter_dir argument" >&2
  exit 1
fi

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

python scripts/lora_qwen35_pipeline.py predict-test \
  --config "${CONFIG_PATH}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --output "${OUTPUT_CSV}"

python scripts/check_submission.py --submission "${OUTPUT_CSV}"
