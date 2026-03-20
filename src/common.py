from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


QUAD_COLS = ["AspectTerms", "OpinionTerms", "Categories", "Polarities"]
SUBMIT_COLS = ["id", *QUAD_COLS]


@dataclass(frozen=True)
class DataPaths:
    train_reviews: Path
    train_labels: Path
    test_reviews: Path
    result_example: Path


def discover_data_paths(root: Path) -> DataPaths:
    data_dir = root / "data"
    return DataPaths(
        train_reviews=data_dir / "Train_reviews.csv",
        train_labels=data_dir / "Train_labels.csv",
        test_reviews=data_dir / "Test_reviews.csv",
        result_example=data_dir / "Result(example).csv",
    )


def read_csv_utf8(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


def normalize_label_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "id" in out.columns:
        out["id"] = pd.to_numeric(out["id"], errors="raise").astype(int)
    for c in QUAD_COLS:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    for c in ["A_start", "A_end", "O_start", "O_end"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def as_tuple_set(df: pd.DataFrame, cols: Iterable[str] = ("id", *QUAD_COLS)) -> set[tuple]:
    use_cols = list(cols)
    return set(map(tuple, df[use_cols].values.tolist()))


def save_csv_no_header(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, header=False, encoding="utf-8")

