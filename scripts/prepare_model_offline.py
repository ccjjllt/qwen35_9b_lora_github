from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download model files for offline training.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--local-dir", type=Path, default=Path("models") / "Qwen3.5-9B")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--endpoint", type=str, default=None, help="Optional custom endpoint, e.g. https://hf-mirror.com")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--only-required",
        action="store_true",
        help="Download only files needed for training/inference (recommended).",
    )
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "3600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

    allow_patterns = None
    if args.only_required:
        allow_patterns = [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "merges.txt",
            "vocab.json",
            "special_tokens_map.json",
            "*.safetensors",
            "*.model",
            "*.txt",
            "*.json",
        ]

    args.local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.model_id,
        repo_type="model",
        local_dir=str(args.local_dir),
        revision=args.revision,
        token=token,
        allow_patterns=allow_patterns,
        max_workers=max(1, int(args.max_workers)),
    )
    total_bytes = sum(p.stat().st_size for p in iter_files(args.local_dir))
    total_gb = total_bytes / (1024**3)
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "revision": args.revision,
                "local_dir": str(args.local_dir),
                "snapshot_path": path,
                "total_bytes": total_bytes,
                "total_gb": round(total_gb, 3),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
