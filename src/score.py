"""Phase 8 — exact metric (per S1 entity, macro-averaged, singletons included) + loss decomposition."""
from __future__ import annotations

import pandas as pd


def f_beta_half(pred: set[str], true: set[str]) -> float:
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1.25 * p * r) / (0.25 * p + r)


def macro_score(preds: dict[str, set[str]], truth: dict[str, set[str]]) -> float:
    if not truth:
        return 0.0
    return sum(f_beta_half(set(preds.get(e, ())), t) for e, t in truth.items()) / len(truth)


def breakdown(preds, truth, candidates=None, country: dict[str, str] | None = None) -> dict:
    """Where is the score lost? blocking loss vs model loss vs false merges vs singleton errors."""
    rows = []
    for e, t in truth.items():
        p = set(preds.get(e, ()))
        cset = set(candidates.get(e, ())) if candidates is not None else None
        rows.append({
            "entity": e, "f": f_beta_half(p, t), "n_true": len(t),
            "tp": len(p & t), "fp": len(p - t), "fn": len(t - p),
            "fn_blocking": len(t - cset) if cset is not None else 0,
            "country": (country or {}).get(e, ""),
        })
    df = pd.DataFrame(rows)
    df["bucket"] = df["n_true"].clip(upper=3).map({0: "0 (singleton)", 1: "1", 2: "2", 3: "3+"})
    out = {
        "macro_f05": df["f"].mean(),
        "n_entities": len(df),
        "singleton_rate": (df["n_true"] == 0).mean(),
        "singleton_score": df.loc[df.n_true == 0, "f"].mean() if (df.n_true == 0).any() else None,
        "false_merges_on_singletons": int(((df.n_true == 0) & (df.fp > 0)).sum()),
        "tp": int(df.tp.sum()), "fp": int(df.fp.sum()), "fn": int(df.fn.sum()),
        "fn_from_blocking": int(df.fn_blocking.sum()),
        "fn_from_model": int(df.fn.sum() - df.fn_blocking.sum()),
        "pair_precision": df.tp.sum() / max(df.tp.sum() + df.fp.sum(), 1),
        "pair_recall": df.tp.sum() / max(df.tp.sum() + df.fn.sum(), 1),
        "by_match_count": df.groupby("bucket")["f"].agg(["mean", "size"]).round(4).to_dict("index"),
    }
    if country:
        out["by_country"] = df.groupby("country")["f"].agg(["mean", "size"]).round(4).to_dict("index")
    return out


def _selftest():
    assert abs(f_beta_half({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}) - 0.714) < 1e-3
    assert f_beta_half(set(), set()) == 1.0 and f_beta_half({"a"}, set()) == 0.0
    assert f_beta_half(set(), {"a"}) == 0.0


_selftest()
