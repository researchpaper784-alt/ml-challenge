"""Phase 4 — entity-level folds and labelled blocked pairs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def entity_folds(truth: dict[str, set[str]], country: dict[str, str], k: int, seed: int) -> dict[str, int]:
    """Split by S1 entity, stratified by (match-count bucket, country). Never by pair."""
    rng = np.random.default_rng(seed)
    strata = {}
    for e, t in truth.items():
        strata.setdefault((min(len(t), 3), country.get(e, "")), []).append(e)
    fold = {}
    offset = 0
    for key in sorted(strata, key=str):
        ents = sorted(strata[key])
        rng.shuffle(ents)
        for n, e in enumerate(ents):
            fold[e] = (n + offset) % k
        offset += len(ents)          # rotate so small strata don't all land in fold 0
    return fold


def label_pairs(cands: pd.DataFrame, truth: dict[str, set[str]]) -> np.ndarray:
    return np.fromiter((c in truth.get(s, ()) for s, c in zip(cands["s1_id"], cands["cand_id"])),
                       dtype=np.int8, count=len(cands))


def training_mask(F: pd.DataFrame, labels: np.ndarray, s1_ids: np.ndarray, max_neg: int | None) -> np.ndarray:
    """Keep all positives + the hardest `max_neg` blocked negatives per entity (by composite)."""
    if not max_neg:
        return np.ones(len(F), dtype=bool)
    df = pd.DataFrame({"s1": s1_ids, "y": labels, "c": F["composite"].to_numpy()})
    neg_rank = df[df.y == 0].groupby("s1")["c"].rank(ascending=False, method="first")
    keep = np.ones(len(F), dtype=bool)
    keep[neg_rank.index[neg_rank.to_numpy() > max_neg]] = False
    return keep
