#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_resume_fold.sh [config] [fold] [checkpoint_dir]
# Example:
#   bash scripts/run_resume_fold.sh configs/qwen35_9b_lora.yaml 0 outputs/qwen35_lora_fold0/checkpoint-120

CONFIG_PATH="${1:-configs/qwen35_9b_lora.yaml}"
FOLD="${2:-0}"
RESUME_DIR="${3:-}"

if [[ -z "${RESUME_DIR}" ]]; then
  echo "[ERROR] missing checkpoint_dir argument" >&2
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

python scripts/lora_qwen35_pipeline.py train-fold \
  --config "${CONFIG_PATH}" \
  --fold "${FOLD}" \
  --resume-from "${RESUME_DIR}"
