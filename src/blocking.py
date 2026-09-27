"""Phase 3 — recall-oriented candidate generation (union of complementary indexes).

The output of `generate_candidates` IS candidate_pairs.tsv: it is exactly the set the model scores.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



@dataclass
class BlockingIndex:
    s1: pd.DataFrame
    pool: pd.DataFrame
    char_vec: TfidfVectorizer
    word_vec: TfidfVectorizer
    s1_char: sparse.csr_matrix
    pool_char: sparse.csr_matrix
    s1_word: sparse.csr_matrix
    pool_word: sparse.csr_matrix
    token_idf: dict


def _word_doc(df: pd.DataFrame) -> pd.Series:
    return (df["name_full"] + " " + df["addr_key"] + " " + df["addr_postcode"]).str.strip()


def build_index(s1: pd.DataFrame, pool: pd.DataFrame, cfg: dict) -> BlockingIndex:
    """Vectorizers are fit on this split's own corpus (S1 + pool); identical procedure for
    train/val and test, so feature scales are comparable."""
    lo, hi = cfg["char_ngram_range"]
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(lo, hi), min_df=1,
                               sublinear_tf=True, dtype=np.float32)
    char_vec.fit(pd.concat([s1["name_core"], pool["name_core"]]))
    word_vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", min_df=1,
                               sublinear_tf=True, dtype=np.float32)
    word_vec.fit(pd.concat([_word_doc(s1), _word_doc(pool)]))

    # token IDF over name_core + addr_key of all records (used by rare-token index and features)
    docs = pd.concat([s1["name_core"] + " " + s1["addr_key"], pool["name_core"] + " " + pool["addr_key"]])
    n = len(docs)
    dfc = defaultdict(int)
    for d in docs:
        for t in set(d.split()):
            dfc[t] += 1
    token_idf = {t: float(np.log((1 + n) / (1 + c)) + 1) for t, c in dfc.items()}
    token_idf["__N__"] = n

    return BlockingIndex(
        s1=s1, pool=pool, char_vec=char_vec, word_vec=word_vec,
        s1_char=char_vec.transform(s1["name_core"]).tocsr(),
        pool_char=char_vec.transform(pool["name_core"]).tocsr(),
        s1_word=word_vec.transform(_word_doc(s1)).tocsr(),
        pool_word=word_vec.transform(_word_doc(pool)).tocsr(),
        token_idf=token_idf,
    )


def _topk_sparse(A: sparse.csr_matrix, B: sparse.csr_matrix, rows: np.ndarray, cols: np.ndarray,
                 k: int, chunk_cap: int, min_score: float = 1e-6):
    """Yield (s1_row, pool_col, score, rank) for top-k cosine of A[rows] vs B[cols]."""
    if len(rows) == 0 or len(cols) == 0 or k <= 0:
        return
    Bt = B[cols].T.tocsc()
    chunk = int(max(1, min(chunk_cap, 2e7 // max(len(cols), 1))))
    kk = min(k, len(cols))
    for st in range(0, len(rows), chunk):
        r = rows[st:st + chunk]
        S = (A[r] @ Bt).toarray()
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        sc = np.take_along_axis(S, idx, axis=1)
        order = np.argsort(-sc, axis=1)
        idx, sc = np.take_along_axis(idx, order, 1), np.take_along_axis(sc, order, 1)
        for i in range(len(r)):
            for rank in range(kk):
                if sc[i, rank] > min_score:
                    yield r[i], cols[idx[i, rank]], float(sc[i, rank]), rank + 1


def _country_groups(ix: BlockingIndex, mode: str):
    """Yields (s1_rows, pool_cols). same_first: same-country pool, fall back to all when empty.
    Country is compared as an opaque string — an unseen label (e.g. France) behaves identically."""
    all_cols = np.arange(len(ix.pool))
    if mode == "all":
        yield np.arange(len(ix.s1)), all_cols
        return
    pc = ix.pool["country_norm"].to_numpy()
    for c, rows in ix.s1.groupby("country_norm").indices.items():
        cols = np.flatnonzero(pc == c)
        yield np.asarray(rows), (cols if len(cols) else all_cols)


def _rare_token_index(ix: BlockingIndex, cfg: dict, rows, cols):
    n = ix.token_idf["__N__"]
    max_df = cfg["rare_token_max_df"]
    idf_thresh = np.log((1 + n) / (1 + max(2, max_df * n))) + 1
    rare = sorted(t for t, v in ix.token_idf.items() if t != "__N__" and v >= idf_thresh)
    if not rare:
        return
    cv = CountVectorizer(vocabulary=rare, binary=True, token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
    w = sparse.diags(np.array([ix.token_idf[t] for t in rare], dtype=np.float32))
    s1n, pn = cv.transform(ix.s1["name_core"]), cv.transform(ix.pool["name_core"])
    s1a, pa = cv.transform(ix.s1["addr_key"]), cv.transform(ix.pool["addr_key"])
    k = cfg["rare_token_k"]
    PnT, PaT = (pn[cols] @ w).T.tocsc(), pa[cols].T.tocsc()
    PaTw = (pa[cols] @ w).T.tocsc()
    for st in range(0, len(rows), 1000):
        r = rows[st:st + 1000]
        name_s = (s1n[r] @ PnT).toarray()
        addr_cnt = (s1a[r] @ PaT).toarray()
        score = name_s + 0.5 * (s1a[r] @ PaTw).toarray()
        score[(name_s <= 0) & (addr_cnt < 2)] = 0
        kk = min(k, score.shape[1])
        idx = np.argpartition(-score, kk - 1, axis=1)[:, :kk]
        for i in range(len(r)):
            sc = score[i, idx[i]]
            order = np.argsort(-sc)
            for rank, j in enumerate(order, 1):
                if sc[j] > 0:
                    yield r[i], cols[idx[i, j]], float(sc[j]), rank


def _key_blocks(ix: BlockingIndex, cfg: dict):
    """Exact-key blocks that rescue typo-heavy pairs the vector indexes rank low."""
    def keys(df):
        out = []
        for r in df.itertuples(index=False):
            ks = []
            if r.addr_postcode and r.name_core:
                ks.append(("pc3", r.addr_postcode, r.name_core[:3]))
            if r.name_phon:
                ks.append(("phon", r.name_phon))
            if r.addr_numbers and r.name_initials:
                ks.append(("numini", r.addr_numbers, r.name_initials))
            if r.name_nospace:
                ks.append(("nospace", r.name_nospace))
            out.append(ks)
        return out
    pool_map = defaultdict(list)
    for j, ks in enumerate(keys(ix.pool)):
        for k in ks:
            pool_map[k].append(j)
    cap = cfg["key_block_max_size"]
    for i, ks in enumerate(keys(ix.s1)):
        hits = defaultdict(int)
        for k in ks:
            js = pool_map.get(k, ())
            if 0 < len(js) <= cap:
                for j in js:
                    hits[j] += 1
        for j, c in hits.items():
            yield i, j, float(c), 1


def generate_candidates(ix: BlockingIndex, cfg: dict) -> pd.DataFrame:
    """Returns one row per (s1_id, cand_id) with per-index score/rank columns (NaN = not produced)."""
    recs = defaultdict(dict)

    def add(name, it):
        for i, j, s, rk in it:
            d = recs[(i, j)]
            if s > d.get(f"blk_{name}_score", -1):
                d[f"blk_{name}_score"], d[f"blk_{name}_rank"] = s, rk

    cap = cfg["chunk_size"]
    for rows, cols in _country_groups(ix, cfg["country_mode"]):
        add("char", _topk_sparse(ix.s1_char, ix.pool_char, rows, cols, cfg["char_tfidf_k"], cap))
        add("word", _topk_sparse(ix.s1_word, ix.pool_word, rows, cols, cfg["word_tfidf_k"], cap))
        add("rare", _rare_token_index(ix, cfg, rows, cols))
    if cfg["country_mode"] != "all" and cfg.get("cross_country_k", 0) > 0:
        add("xc", _topk_sparse(ix.s1_char, ix.pool_char, np.arange(len(ix.s1)),
                               np.arange(len(ix.pool)), cfg["cross_country_k"], cap))
    add("key", _key_blocks(ix, cfg))

    if not recs:
        return pd.DataFrame(columns=["s1_idx", "pool_idx", "s1_id", "cand_id"])
    keys = list(recs)
    out = pd.DataFrame([recs[k] for k in keys])
    out.insert(0, "pool_idx", [k[1] for k in keys])
    out.insert(0, "s1_idx", [k[0] for k in keys])
    out.insert(2, "s1_id", ix.s1["entity_id"].to_numpy()[out["s1_idx"]])
    out.insert(3, "cand_id", ix.pool["entity_id"].to_numpy()[out["pool_idx"]])
    for name in ("char", "word", "rare", "xc", "key"):      # fixed schema across splits
        for suf in ("score", "rank"):
            if f"blk_{name}_{suf}" not in out:
                out[f"blk_{name}_{suf}"] = np.nan
    blk_cols = [c for c in out.columns if c.startswith("blk_") and c.endswith("_score")]
    out["blk_n_indexes"] = out[blk_cols].notna().sum(axis=1)
    return out.sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)


def blocking_report(cands: pd.DataFrame, truth: dict[str, set[str]], n_s1: int, n_pool: int) -> dict:
    """Recall ceiling + reduction ratio (both go in the methodology doc)."""
    cset = set(zip(cands["s1_id"], cands["cand_id"]))
    true_pairs = [(s, c) for s, cs in truth.items() for c in cs]
    hit = sum(p in cset for p in true_pairs)
    per = cands.groupby("s1_id").size().reindex(list(truth), fill_value=0) if truth else cands.groupby("s1_id").size()
    return {
        "recall_ceiling": hit / max(len(true_pairs), 1),
        "true_pairs": len(true_pairs),
        "missed_pairs": len(true_pairs) - hit,
        "n_candidates": len(cands),
        "reduction_ratio": 1 - len(cands) / max(n_s1 * n_pool, 1),
        "cands_per_entity_mean": float(per.mean()) if len(per) else 0.0,
        "cands_per_entity_median": float(per.median()) if len(per) else 0.0,
        "cands_per_entity_p99": float(per.quantile(0.99)) if len(per) else 0.0,
    }
