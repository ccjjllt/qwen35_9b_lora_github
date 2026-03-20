from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import SUBMIT_COLS, discover_data_paths, normalize_label_df, read_csv_utf8


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission format.")
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args()

    if not args.submission.exists():
        raise FileNotFoundError(args.submission)

    pred = pd.read_csv(args.submission, header=None, encoding="utf-8")
    if pred.shape[1] != 5:
        raise ValueError(f"submission must have 5 columns, got {pred.shape[1]}")
    pred.columns = SUBMIT_COLS
    pred = normalize_label_df(pred)

    paths = discover_data_paths(ROOT)
    test_reviews = read_csv_utf8(paths.test_reviews)
    expected_ids = test_reviews["id"].astype(int).tolist()

    if not pred["id"].is_monotonic_increasing:
        raise ValueError("id column must be non-decreasing")
    got_ids = pred["id"].drop_duplicates().tolist()
    if got_ids != expected_ids:
        raise ValueError("id coverage/order mismatch with Test_reviews.csv")

    train_labels = normalize_label_df(read_csv_utf8(paths.train_labels))
    category_set = set(train_labels["Categories"].astype(str).unique().tolist()) | {"_"}
    polarity_set = set(train_labels["Polarities"].astype(str).unique().tolist()) | {"_"}
    if not pred["Categories"].isin(category_set).all():
        raise ValueError("Categories column contains illegal labels")
    if not pred["Polarities"].isin(polarity_set).all():
        raise ValueError("Polarities column contains illegal labels")

    print("Submission check passed.")


if __name__ == "__main__":
    main()

