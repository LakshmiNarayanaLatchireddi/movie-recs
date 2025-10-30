from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd


@dataclass
class EvalResult:
    users: int
    k: int
    hr: float
    ndcg: float


def _sample_negatives_for_user(
    user_id: int,
    pos_items_user: Set[int],
    candidate_items: Sequence[int],
    n: int,
) -> List[int]:
    # Exclude user's positives from sampling space
    pool = [iid for iid in candidate_items if iid not in pos_items_user]
    if len(pool) < n:
        # fall back (rare in very small toy sets)
        return random.sample(pool, len(pool))
    return random.sample(pool, n)


def evaluate_topk(
    model,
    test_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    k: int = 10,
    train_df: Optional[pd.DataFrame] = None,
    negatives_per_user: int = 99,
    items_df: Optional[pd.DataFrame] = None,
    rng_seed: int = 42,
) -> EvalResult:
    """
    Leave-one-out style evaluation:
      For each user with exactly one positive in test_df, sample N negatives,
      score [1 positive + N negatives], compute HR@K and NDCG@K, then average.

    The model must implement:
      score_items(user_id: int, item_ids: Sequence[int]) -> np.ndarray[float]
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    # Build candidate item universe
    if items_df is not None and item_col in items_df.columns:
        all_items = items_df[item_col].astype(int).unique().tolist()
    else:
        # fallback to all items seen anywhere
        series_list = [test_df[item_col]]
        if train_df is not None and item_col in train_df.columns:
            series_list.append(train_df[item_col])
        all_items = pd.concat(series_list).astype(int).unique().tolist()

    # Map user -> set of positives (train+test) to avoid leakage in negatives
    user_pos = {}
    if train_df is not None:
        for u, g in train_df.groupby(user_col):
            user_pos[int(u)] = set(map(int, g[item_col].tolist()))
    for u, g in test_df.groupby(user_col):
        user_pos.setdefault(int(u), set()).update(map(int, g[item_col].tolist()))

    # Keep only users with exactly ONE positive in test set (leave-one-out)
    test_counts = test_df.groupby(user_col)[item_col].count()
    loo_users = set(map(int, test_counts[test_counts == 1].index.tolist()))
    if not loo_users:
        return EvalResult(users=0, k=k, hr=0.0, ndcg=0.0)

    # Fast lookup: user -> the single positive item in test
    test_pos_map = (
        test_df.groupby(user_col)[item_col].first().astype(int).to_dict()
    )

    hits = 0.0
    ndcgs = 0.0
    user_count = 0

    for u in loo_users:
        pos_item = int(test_pos_map[u])
        negs = _sample_negatives_for_user(
            user_id=u,
            pos_items_user=user_pos.get(u, set()),
            candidate_items=all_items,
            n=negatives_per_user,
        )
        candidates = [pos_item] + negs

        # Score
        scores = model.score_items(u, candidates)
        # Higher score = better rank
        order = np.argsort(-scores)
        ranked_items = [candidates[i] for i in order]

        # Find rank (1-based)
        rank = ranked_items.index(pos_item) + 1

        # HR@K
        if rank <= k:
            hits += 1.0
            # NDCG@K
            ndcgs += 1.0 / np.log2(rank + 1)
        # else add 0 to both

        user_count += 1

    hr = hits / max(1, user_count)
    ndcg = ndcgs / max(1, user_count)
    return EvalResult(users=user_count, k=k, hr=hr, ndcg=ndcg)
