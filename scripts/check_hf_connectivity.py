from __future__ import annotations

import argparse
import os
import sys

import requests
from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description="Check connectivity to official Hugging Face endpoints.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--token", type=str, default=None, help="HF token (optional).")
    args = parser.parse_args()

    print("== Env check ==")
    print(f"HF_ENDPOINT={os.getenv('HF_ENDPOINT')}")
    print(f"HF_HOME={os.getenv('HF_HOME')}")
    if os.getenv("HF_ENDPOINT"):
        print("[WARN] HF_ENDPOINT is set. For official hub, unset it: unset HF_ENDPOINT")

    print("\n== HTTP check ==")
    for url in ["https://huggingface.co", "https://huggingface.co/api/models"]:
        try:
            r = requests.get(url, timeout=20)
            print(f"{url} -> {r.status_code}")
        except Exception as exc:
            print(f"{url} -> ERROR: {exc}")

    print("\n== Hub API check ==")
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)
    try:
        info = api.model_info(args.model_id)
        print(f"model_info ok: {info.id}")
        print("Connectivity looks good.")
    except Exception as exc:
        print(f"model_info failed: {exc}")
        print("Suggestion:")
        print("1) unset HF_ENDPOINT")
        print("2) export HF_TOKEN=*** and retry")
        print("3) if blocked by network policy, use local predownload + offline mode")
        sys.exit(1)


if __name__ == "__main__":
    main()

