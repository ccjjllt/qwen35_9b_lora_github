from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fold metrics.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--prefix", type=str, default="qwen35_lora_fold")
    args = parser.parse_args()

    files = sorted(args.reports_dir.glob(f"{args.prefix}*_metrics.json"))
    if not files:
        raise FileNotFoundError(f"No metrics found in {args.reports_dir} with prefix {args.prefix}")

    rows = []
    for p in files:
        obj = json.loads(p.read_text(encoding="utf-8"))
        rows.append((int(obj["fold"]), float(obj["f1"])))
    rows.sort(key=lambda x: x[0])
    f1s = [x[1] for x in rows]

    print("fold_f1:")
    for f, v in rows:
        print(f"  fold{f}: {v:.6f}")
    print(f"mean_f1: {statistics.mean(f1s):.6f}")
    print(f"std_f1:  {statistics.pstdev(f1s):.6f}")


if __name__ == "__main__":
    main()

