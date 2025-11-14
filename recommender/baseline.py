import numpy as np
import pandas as pd
from typing import Iterable, Dict

class PopularityModel:
    """
    Scores items by global popularity (count or sum of ratings).
    Deterministic, fast, robust when data is small or sparse.
    """
    def __init__(self):
        self.item_scores_: Dict[int, float] = {}
        self.default_score_: float = 0.0

    def fit(self, df: pd.DataFrame, user_col: str, item_col: str, rating_col: str | None = None):
        if rating_col and rating_col in df.columns:
            grp = df.groupby(item_col)[rating_col].sum()
        else:
            grp = df.groupby(item_col)[user_col].count()
        grp = grp.sort_values(ascending=False)
        self.item_scores_ = grp.to_dict()
        self.default_score_ = float(grp.mean()) if len(grp) else 0.0
        return self

    def score_items(self, user_id: int, item_ids: Iterable[int]) -> np.ndarray:
        scores = []
        for it in item_ids:
            scores.append(self.item_scores_.get(int(it), self.default_score_))
        return np.asarray(scores, dtype=float)