#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup_env.sh
# Optional:
#   AUTO_UPGRADE_TRANSFORMERS=1 bash scripts/setup_env.sh

AUTO_UPGRADE_TRANSFORMERS="${AUTO_UPGRADE_TRANSFORMERS:-1}"

python -m pip install -U pip
python -m pip install -r requirements.txt

if python - <<'PY'
import importlib
import sys

mods = {
    "pandas": "pandas",
    "numpy": "numpy",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "peft": "peft",
    "bitsandbytes": "bitsandbytes",
}

missing: list[tuple[str, str]] = []
for mod, pkg in mods.items():
    try:
        importlib.import_module(mod)
    except Exception as exc:  # pragma: no cover
        missing.append((pkg, str(exc)))

if missing:
    print("[ERROR] Missing required packages:")
    for pkg, err in missing:
        print(f"  - {pkg}: {err}")
    sys.exit(2)

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
ok = "qwen3_5" in CONFIG_MAPPING
print(f"[INFO] transformers supports qwen3_5: {ok}")
if not ok:
    sys.exit(3)

print("[INFO] dependency check passed.")
PY
then
  :
else
  status=$?
  if [[ "$status" -eq 3 && "${AUTO_UPGRADE_TRANSFORMERS}" == "1" ]]; then
    echo "[INFO] Installing latest transformers from source for Qwen3.5 support..."
    python -m pip install -U "git+https://github.com/huggingface/transformers.git"
    python - <<'PY'
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
if "qwen3_5" not in CONFIG_MAPPING:
    raise SystemExit("[ERROR] transformers upgrade completed but qwen3_5 is still unsupported.")
print("[INFO] qwen3_5 support check passed after upgrade.")
PY
  else
    echo "[ERROR] setup failed. Set AUTO_UPGRADE_TRANSFORMERS=1 to auto-upgrade transformers."
    exit "$status"
  fi
fi
