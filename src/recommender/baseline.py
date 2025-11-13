import numpy as np
import pandas as pd
from typing import Optional

class PopularityRecommender:
    """
    Popularity baseline:
    - fit(): ranks items by interaction count (most popular first)
    - score_items(): score per item (higher = more popular)
    - recommend(): top-K popular items (excluding seen)
    """
    def __init__(
        self,
        user_col: str = "userId",
        item_col: str = "movieId",
        rating_col: Optional[str] = None,      
        timestamp_col: Optional[str] = None,   
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self._order = None

    def fit(self, interactions: pd.DataFrame):
        if self.user_col not in interactions.columns or self.item_col not in interactions.columns:
            raise ValueError(
                f"Expected columns '{self.user_col}' and '{self.item_col}' in interactions; "
                f"got {list(interactions.columns)}"
            )
        counts = (
            interactions
            .groupby(self.item_col)[self.user_col]
            .count()
            .sort_values(ascending=False)
        )
        self._order = counts.index.to_numpy()
        return self

    def score_items(self, user_id, item_ids):
        if self._order is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        ranks = {int(item): -i for i, item in enumerate(self._order)}
        return np.array([ranks.get(int(x), -1e9) for x in item_ids], dtype=float)

    def recommend(self, user_id, k: int = 10, exclude=None):
        if self._order is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        exclude = set(exclude or [])
        recs = [int(i) for i in self._order if int(i) not in exclude]
        return recs[:k]
