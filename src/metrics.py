from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .common import QUAD_COLS, as_tuple_set


@dataclass
class ScoreResult:
    precision: float
    recall: float
    f1: float
    pred_count: int
    gold_count: int
    hit_count: int


def score_quadruples(pred_df: pd.DataFrame, gold_df: pd.DataFrame) -> ScoreResult:
    pred_set = as_tuple_set(pred_df, cols=("id", *QUAD_COLS))
    gold_set = as_tuple_set(gold_df, cols=("id", *QUAD_COLS))
    hit = len(pred_set & gold_set)
    p = len(pred_set)
    g = len(gold_set)
    precision = hit / p if p else 0.0
    recall = hit / g if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ScoreResult(
        precision=precision,
        recall=recall,
        f1=f1,
        pred_count=p,
        gold_count=g,
        hit_count=hit,
    )

