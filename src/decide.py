"""Phase 7 — decision layer: turn pair probabilities into per-entity ID lists, tuned on macro F0.5."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def _prep(s1_ids, cand_ids, p):
    df = pd.DataFrame({"s1": s1_ids, "cand": cand_ids, "p": p})
    df["src"] = df["cand"].str[:2]
    g = df.groupby("s1")["p"]
    df["best"] = g.transform("max")
    df["rank_src"] = df.groupby(["s1", "src"])["p"].rank(ascending=False, method="first")
    return df


def _mask(df, tau, margin, cap, s3_offset, emit_bar):
    thr = np.where(df["src"].to_numpy() == "S3", tau + s3_offset, tau)
    m = df["p"].to_numpy() >= thr
    if margin is not None:
        m &= df["p"].to_numpy() >= df["best"].to_numpy() - margin
    if cap is not None:
        m &= df["rank_src"].to_numpy() <= cap
    if emit_bar > 0:                       # stricter bar before an entity may emit anything
        m &= df["best"].to_numpy() >= tau + emit_bar
    return m


def decide(s1_ids, cand_ids, p, params: dict) -> dict[str, list[str]]:
    df = _prep(s1_ids, cand_ids, p)
    m = _mask(df, params["tau"], params.get("margin"), params.get("per_source_cap"),
              params.get("s3_offset", 0.0), params.get("emit_bar", 0.0))
    out: dict[str, list[str]] = {}
    for s, c in zip(df["s1"].to_numpy()[m], df["cand"].to_numpy()[m]):
        out.setdefault(s, []).append(c)
    return out


def tune(s1_ids, cand_ids, p, labels, truth: dict[str, set[str]], cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Vectorised grid search of macro F0.5. Entities with zero candidates still count (they get the
    empty prediction). Picks the best *plateau* (smoothed over neighbouring tau), not a spike."""
    df = _prep(s1_ids, cand_ids, p)
    df["y"] = labels
    ents = sorted(truth)
    code = {e: n for n, e in enumerate(ents)}
    ec = df["s1"].map(code).to_numpy()
    keep = ~np.isnan(ec.astype(float))
    df, ec = df[keep], ec[keep].astype(int)
    n_true = np.array([len(truth[e]) for e in ents], dtype=float)
    y = df["y"].to_numpy()
    E = len(ents)

    def macro(m):
        npred = np.bincount(ec[m], minlength=E).astype(float)
        tp = np.bincount(ec[m & (y == 1)], minlength=E).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            prec, rec = tp / npred, tp / n_true
            f = np.where(tp > 0, 1.25 * prec * rec / (0.25 * prec + rec), 0.0)
        f = np.where((n_true == 0) & (npred == 0), 1.0, f)
        return f.mean()

    lo, hi, st = cfg["tau_grid"]
    taus = np.round(np.arange(lo, hi + 1e-9, st), 4)
    rows = []
    for margin, cap, off, bar in itertools.product(cfg["margin_grid"], cfg["per_source_cap_grid"],
                                                   cfg["source_offset_grid"], cfg.get("emit_bar_grid", [0.0])):
        for tau in taus:
            rows.append({"tau": tau, "margin": margin, "per_source_cap": cap, "s3_offset": off,
                         "emit_bar": bar, "f05": macro(_mask(df, tau, margin, cap, off, bar))})
    grid = pd.DataFrame(rows)
    hw = cfg.get("plateau_halfwidth", 2)
    keys = ["margin", "per_source_cap", "s3_offset", "emit_bar"]
    grid["f05_smooth"] = (grid.groupby(keys, dropna=False)["f05"]
                          .transform(lambda s: s.rolling(2 * hw + 1, center=True, min_periods=1).mean()))
    best = grid.loc[grid["f05_smooth"].idxmax()]
    params = {"tau": float(best.tau),
              "margin": None if pd.isna(best.margin) else float(best.margin),
              "per_source_cap": None if pd.isna(best.per_source_cap) else int(best.per_source_cap),
              "s3_offset": float(best.s3_offset), "emit_bar": float(best.emit_bar),
              "val_f05": float(best.f05), "val_f05_smooth": float(best.f05_smooth),
              "val_f05_grid_max": float(grid.f05.max())}
    return params, grid
