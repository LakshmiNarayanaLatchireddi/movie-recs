import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Sequence, Any


@dataclass
class TopKEvalResult:
    """Container for top-K evaluation metrics."""
    users: int
    hr: float
    ndcg: float


def _sample_negatives_for_user(
    all_items: np.ndarray,
    pos_items: Sequence[Any],
    train_items: Optional[Sequence[Any]],
    negatives_per_user: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Sample negative items for a single user, excluding positives (test + train).
    """
    exclude = set(pos_items)
    if train_items is not None:
        exclude.update(train_items)

    # candidate negatives = all_items \ exclude
    mask = np.isin(all_items, list(exclude), invert=True)
    candidates = all_items[mask]

    if len(candidates) == 0:
        return np.array([], dtype=all_items.dtype)

    if len(candidates) <= negatives_per_user:
        return candidates

    return rng.choice(candidates, size=negatives_per_user, replace=False)


def evaluate_topk(
    model,
    test_df: pd.DataFrame,
    user_col: str,
    item_col: str,
    k: int = 10,
    train_df: Optional[pd.DataFrame] = None,
    negatives_per_user: int = 99,
    all_items: Optional[Sequence[Any]] = None,
    random_state: int = 42,
) -> TopKEvalResult:
    """
    Compute HR@K and NDCG@K by sampling negatives per user and ranking with the model.

    The model must implement:
        score_items(user_id, item_ids) -> 1D np.ndarray of scores
    """

    rng = np.random.RandomState(random_state)

    if all_items is None:
        if train_df is not None:
            all_items = pd.concat(
                [train_df[item_col], test_df[item_col]], axis=0
            ).drop_duplicates().values
        else:
            all_items = test_df[item_col].drop_duplicates().values

    all_items = np.asarray(all_items)

    user_groups = test_df.groupby(user_col)
    hrs = []
    ndcgs = []
    user_count = 0

    for user_id, group in user_groups:
        pos_items = group[item_col].values
        pos_item = pos_items[0]

        if train_df is not None:
            user_train = train_df.loc[train_df[user_col] == user_id, item_col].values
        else:
            user_train = None

        neg_items = _sample_negatives_for_user(
            all_items=all_items,
            pos_items=[pos_item],
            train_items=user_train,
            negatives_per_user=negatives_per_user,
            rng=rng,
        )

        if neg_items.size == 0:
            continue

        candidates = np.concatenate([[pos_item], neg_items])

        scores = model.score_items(user_id, candidates)
        scores = np.asarray(scores, dtype=float)

        order = np.argsort(-scores)
        topk_idx = order[:k]
        topk_items = candidates[topk_idx]

        hit = 1.0 if pos_item in topk_items else 0.0

        if hit:
            rank = int(np.where(topk_items == pos_item)[0][0]) + 1
            ndcg = 1.0 / np.log2(rank + 1)
        else:
            ndcg = 0.0

        hrs.append(hit)
        ndcgs.append(ndcg)
        user_count += 1

    if user_count == 0:
        return TopKEvalResult(users=0, hr=0.0, ndcg=0.0)

    return TopKEvalResult(
        users=user_count,
        hr=float(np.mean(hrs)),
        ndcg=float(np.mean(ndcgs)),
    )