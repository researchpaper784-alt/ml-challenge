"""Phase 5 — pair features. All symmetric and country-agnostic (country used only as same/different)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import strsim as ss
from .blocking import BlockingIndex


def _rowcos(A, B, ia, ib) -> np.ndarray:
    return np.asarray(A[ia].multiply(B[ib]).sum(axis=1)).ravel()


def _idf_overlap(a: set, b: set, idf: dict, default: float) -> tuple[float, float]:
    if not a or not b:
        return 0.0, 0.0
    inter, union = a & b, a | b
    w = lambda s: sum(idf.get(t, default) for t in s)
    return w(inter) / w(union), max((idf.get(t, default) for t in inter), default=0.0)


def _prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def pair_features(ix: BlockingIndex, cands: pd.DataFrame) -> pd.DataFrame:
    s1, pool = ix.s1, ix.pool
    i, j = cands["s1_idx"].to_numpy(), cands["pool_idx"].to_numpy()
    idf = ix.token_idf
    default_idf = float(np.log(1 + idf["__N__"]) + 1)

    F = pd.DataFrame(index=cands.index)
    F["name_char_cos"] = _rowcos(ix.s1_char, ix.pool_char, i, j)
    F["word_cos"] = _rowcos(ix.s1_word, ix.pool_word, i, j)

    cols = ["business_name", "name_full", "name_core", "name_sorted", "name_initials", "name_phon",
            "name_nospace", "addr_core", "addr_key", "addr_numbers", "addr_postcode",
            "addr_landmarks", "business_address", "country_norm"]
    A = s1[cols].to_numpy()[i]
    B = pool[cols].to_numpy()[j]
    c = {k: n for n, k in enumerate(cols)}

    rows = []
    for a, b in zip(A, B):
        an, bn = a[c["name_core"]], b[c["name_core"]]
        at, bt = set(an.split()), set(bn.split())
        aa, ba = a[c["addr_key"]], b[c["addr_key"]]
        aat, bat = set(aa.split()), set(ba.split())
        anum, bnum = set(a[c["addr_numbers"]].split()), set(b[c["addr_numbers"]].split())
        apc, bpc = a[c["addr_postcode"]], b[c["addr_postcode"]]
        araw, braw = a[c["business_address"]].strip(), b[c["business_address"]].strip()
        n_w, n_max = _idf_overlap(at, bt, idf, default_idf)
        a_w, _ = _idf_overlap(aat, bat, idf, default_idf)
        lm_a, lm_b = set(a[c["addr_landmarks"]].split()), set(b[c["addr_landmarks"]].split())
        rows.append((
            # ---- name
            ss.jaro_winkler(an, bn), ss.lev_sim(an, bn), ss.token_sort(an, bn),
            ss.token_set(a[c["name_full"]], b[c["name_full"]]), ss.partial_ratio(an, bn),
            ss.jaccard(at, bt),
            ss.jaccard(ss.ngrams(a[c["name_nospace"]]), ss.ngrams(b[c["name_nospace"]])),
            ss.lcs_ratio(a[c["name_nospace"]], b[c["name_nospace"]]),
            float(an == bn), float(a[c["name_full"]] == b[c["name_full"]]),
            float(a[c["business_name"]].casefold().strip() == b[c["business_name"]].casefold().strip()),
            float(a[c["name_sorted"]] == b[c["name_sorted"]]),
            float(a[c["name_initials"]] == b[c["name_initials"]]),
            float(a[c["name_phon"]] == b[c["name_phon"]]),
            ss.jaro_winkler(a[c["name_phon"]], b[c["name_phon"]]),
            n_w, n_max,
            abs(len(an) - len(bn)) / max(len(an), len(bn), 1),
            float(an.replace(" ", "") in b[c["name_nospace"]] or bn.replace(" ", "") in a[c["name_nospace"]]),
            # ---- address
            ss.jaro_winkler(a[c["addr_core"]], b[c["addr_core"]]),
            ss.token_set(a[c["addr_core"]], b[c["addr_core"]]),
            ss.jaccard(aat, bat), a_w,
            ss.jaccard(ss.ngrams(aa), ss.ngrams(ba)) if aa and ba else 0.0,
            float(bool(apc) and apc == bpc), _prefix_len(apc, bpc) if apc and bpc else -1,
            float(bool(apc) != bool(bpc)), float(not apc and not bpc),
            ss.jaccard(anum, bnum), float(bool(anum & bnum)), float(bool(anum) != bool(bnum)),
            ss.jaccard(lm_a, lm_b) if lm_a and lm_b else -1.0,
            float(not araw and not braw), float(bool(araw) != bool(braw)),
            # name tokens appearing in the other's address (DBA / name-in-address noise)
            ss.jaccard(at, bat | set(b[c["addr_core"]].split())) if at else 0.0,
            # ---- context
            float(a[c["country_norm"]] == b[c["country_norm"]]),
            float(not an), float(not bn),
        ))
    names = [
        "name_jw", "name_lev", "name_tsort", "name_tset", "name_partial", "name_tok_jacc",
        "name_c3_jacc", "name_lcs", "name_core_eq", "name_full_eq", "name_raw_eq", "name_sorted_eq",
        "name_initials_eq", "name_phon_eq", "name_phon_jw", "name_idf_jacc", "name_max_shared_idf",
        "name_len_diff", "name_contains",
        "addr_jw", "addr_tset", "addr_tok_jacc", "addr_idf_jacc", "addr_c3_jacc",
        "pc_eq", "pc_prefix", "pc_missing_one", "pc_missing_both",
        "num_jacc", "num_overlap", "num_missing_one", "landmark_jacc",
        "addr_empty_both", "addr_empty_one", "name_in_other_addr",
        "same_country", "s1_name_empty", "cand_name_empty",
    ]
    F = pd.concat([F, pd.DataFrame(rows, columns=names, index=cands.index)], axis=1)

    # ---- blocking provenance
    for col in cands.columns:
        if col.startswith("blk_"):
            F[col] = cands[col].to_numpy()
    F["is_s3"] = cands["cand_id"].str.startswith("S3-").astype(float).to_numpy()

    # ---- rarity of the S1 name (common names need stronger evidence)
    s1_core = s1["name_core"].to_numpy()[i]
    F["s1_name_mean_idf"] = [np.mean([idf.get(t, default_idf) for t in s.split()]) if s else 0.0 for s in s1_core]
    F["s1_name_ntok"] = [len(s.split()) for s in s1_core]

    return add_relational(F, cands)


def add_relational(F: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
    """Per-entity context: rank/margin within the S1 entity, popularity and mutual-best of the candidate."""
    F = F.copy()
    F["composite"] = (F["name_char_cos"] + F["name_tset"] + F["name_jw"] + F["word_cos"]
                      + F["addr_tset"] + 0.5 * F["pc_eq"] + 0.5 * F["num_overlap"]) / 5.0
    key = cands["s1_id"].to_numpy()
    cand = cands["cand_id"].to_numpy()
    src = np.where(F["is_s3"].to_numpy() > 0, "S3", "S2")
    tmp = pd.DataFrame({"s1": key, "cand": cand, "src": src, "comp": F["composite"].to_numpy(),
                        "ccos": F["name_char_cos"].to_numpy(), "wcos": F["word_cos"].to_numpy()})
    g = tmp.groupby("s1")
    for col in ("comp", "ccos", "wcos"):
        F[f"{col}_rank"] = g[col].rank(ascending=False, method="min").to_numpy()
        best = g[col].transform("max").to_numpy()
        # margin to best *other* candidate: second-best when this one is the best
        srt = tmp.sort_values(["s1", col], ascending=[True, False])
        sec = srt[srt.groupby("s1").cumcount() == 1].set_index("s1")[col]
        second = tmp["s1"].map(sec).fillna(0.0).to_numpy()
        v = tmp[col].to_numpy()
        F[f"{col}_margin"] = np.where(v >= best, v - second, v - best)
    gs = tmp.groupby(["s1", "src"])["comp"]
    F["comp_rank_in_src"] = gs.rank(ascending=False, method="min").to_numpy()
    best_src = gs.transform("max")
    F["comp_margin_in_src"] = (tmp["comp"] - best_src).to_numpy()
    other_best = tmp.assign(o=np.where(tmp["src"] == "S2", "S3", "S2")).merge(
        tmp.groupby(["s1", "src"])["comp"].max().rename("ob").reset_index().rename(columns={"src": "o"}),
        on=["s1", "o"], how="left")["ob"].fillna(-1).to_numpy()
    F["comp_minus_other_src_best"] = np.where(other_best < 0, 0.0, tmp["comp"].to_numpy() - other_best)
    F["has_other_src_cand"] = (other_best >= 0).astype(float)
    F["n_cands"] = g["comp"].transform("size").to_numpy()
    # candidate side: popularity and mutual-best
    gc = tmp.groupby("cand")["comp"]
    F["cand_popularity"] = gc.transform("size").to_numpy()
    F["cand_rank_among_s1"] = gc.rank(ascending=False, method="min").to_numpy()
    F["cand_margin_among_s1"] = (tmp["comp"] - gc.transform("max")).to_numpy()
    return F


FEATURE_EXCLUDE = {"s1_idx", "pool_idx", "s1_id", "cand_id", "label"}


def feature_columns(F: pd.DataFrame) -> list[str]:
    return [c for c in F.columns if c not in FEATURE_EXCLUDE]
