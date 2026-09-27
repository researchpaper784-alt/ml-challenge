"""Phase 3 — recall-oriented candidate generation (union of complementary indexes).

The output of `generate_candidates` IS candidate_pairs.tsv: it is exactly the set the model scores.
"""
from __future__ import annotations

import resource
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def _mem(label: str) -> None:
    """Peak RSS so far, for pinpointing which step of blocking dominates memory at scale."""
    print(f"[mem] {label}: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}MB", flush=True)



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
    rare_idf_diag: sparse.dia_matrix | None
    s1_rare_name: sparse.csr_matrix | None
    pool_rare_name: sparse.csr_matrix | None
    s1_rare_addr: sparse.csr_matrix | None
    pool_rare_addr: sparse.csr_matrix | None


def _word_doc(df: pd.DataFrame) -> pd.Series:
    return (df["name_full"] + " " + df["addr_key"] + " " + df["addr_postcode"]).str.strip()


def build_index(s1: pd.DataFrame, pool: pd.DataFrame, cfg: dict) -> BlockingIndex:
    """Vectorizers are fit on this split's own corpus (S1 + pool); identical procedure for
    train/val and test, so feature scales are comparable.

    `max_features` caps each vectorizer's vocabulary to its top-N terms by corpus
    frequency (sklearn's own ranking) — at real-dataset scale (millions of documents,
    multiple scripts), an uncapped min_df=1 vocabulary balloons to the point of
    exhausting memory before a single candidate is generated, dominated by n-grams/
    tokens that appear once or twice and carry no real blocking signal anyway.
    """
    _mem("build_index start")
    lo, hi = cfg["char_ngram_range"]
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(lo, hi), min_df=1,
                               max_features=cfg.get("char_max_features"),
                               sublinear_tf=True, dtype=np.float32)
    char_vec.fit(pd.concat([s1["name_core"], pool["name_core"]]))
    _mem(f"char_vec fit (vocab={len(char_vec.vocabulary_):,})")
    word_vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", min_df=1,
                               max_features=cfg.get("word_max_features"),
                               sublinear_tf=True, dtype=np.float32)
    word_vec.fit(pd.concat([_word_doc(s1), _word_doc(pool)]))
    _mem(f"word_vec fit (vocab={len(word_vec.vocabulary_):,})")

    # token IDF over name_core + addr_key of all records (used by rare-token index and features).
    # Tokens appearing exactly once can never produce a *shared* rare-token match between two
    # records, so they're pure vocabulary bloat here; dropping them is a no-op for recall.
    docs = pd.concat([s1["name_core"] + " " + s1["addr_key"], pool["name_core"] + " " + pool["addr_key"]])
    n = len(docs)
    dfc = defaultdict(int)
    for d in docs:
        for t in set(d.split()):
            dfc[t] += 1
    token_idf = {t: float(np.log((1 + n) / (1 + c)) + 1) for t, c in dfc.items() if c > 1}
    token_idf["__N__"] = n
    _mem(f"token_idf built ({len(token_idf):,} tokens)")

    # Rare-token vectorization, done once here rather than inside _rare_token_index: that
    # function used to be called once per country group and rebuilt + re-transformed this
    # over the FULL s1/pool corpus on every single call (the country subsetting only
    # happened afterwards, on the already-transformed matrices) — for N country groups that's
    # N redundant full-corpus CountVectorizer transforms, needlessly multiplying peak memory.
    max_df = cfg["rare_token_max_df"]
    idf_thresh = np.log((1 + n) / (1 + max(2, max_df * n))) + 1
    rare = sorted(t for t, v in token_idf.items() if t != "__N__" and v >= idf_thresh)
    _mem(f"rare token list built ({len(rare):,} tokens)")
    rare_idf_diag = s1_rare_name = pool_rare_name = s1_rare_addr = pool_rare_addr = None
    if rare:
        rare_cv = CountVectorizer(vocabulary=rare, binary=True, token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
        rare_idf_diag = sparse.diags(np.array([token_idf[t] for t in rare], dtype=np.float32))
        s1_rare_name = rare_cv.transform(s1["name_core"]).tocsr()
        pool_rare_name = rare_cv.transform(pool["name_core"]).tocsr()
        s1_rare_addr = rare_cv.transform(s1["addr_key"]).tocsr()
        pool_rare_addr = rare_cv.transform(pool["addr_key"]).tocsr()
    _mem("rare-token matrices transformed")

    s1_char = char_vec.transform(s1["name_core"]).tocsr()
    pool_char = char_vec.transform(pool["name_core"]).tocsr()
    _mem(f"char matrices transformed (pool nnz={pool_char.nnz:,})")
    s1_word = word_vec.transform(_word_doc(s1)).tocsr()
    pool_word = word_vec.transform(_word_doc(pool)).tocsr()
    _mem(f"word matrices transformed (pool nnz={pool_word.nnz:,})")

    return BlockingIndex(
        s1=s1, pool=pool, char_vec=char_vec, word_vec=word_vec,
        s1_char=s1_char, pool_char=pool_char, s1_word=s1_word, pool_word=pool_word,
        token_idf=token_idf,
        rare_idf_diag=rare_idf_diag, s1_rare_name=s1_rare_name, pool_rare_name=pool_rare_name,
        s1_rare_addr=s1_rare_addr, pool_rare_addr=pool_rare_addr,
    )


def _row_topk(indices: np.ndarray, data: np.ndarray, kk: int, min_score: float):
    """Top-k (index, score) pairs from one sparse row's (indices, data), sorted desc."""
    if data.size == 0:
        return indices[:0], data[:0]
    if data.size > kk:
        part = np.argpartition(-data, kk - 1)[:kk]
    else:
        part = np.arange(data.size)
    order = part[np.argsort(-data[part])]
    idx, sc = indices[order], data[order]
    keep = sc > min_score
    return idx[keep], sc[keep]


def _topk_sparse(A: sparse.csr_matrix, B: sparse.csr_matrix, rows: np.ndarray, cols: np.ndarray,
                 k: int, chunk_cap: int, min_score: float = 1e-6):
    """Yield one (s1_rows, pool_cols, scores, ranks) NumPy-array tuple per row-chunk of
    top-k cosine hits for A[rows] vs B[cols].

    Stays sparse end to end: A[r] @ Bt is a sparse product (cost tracks actual
    shared-token overlap, not rows*cols), and top-k is taken from each row's nonzero
    (indices, data) slice directly — never a dense (chunk_rows, len(cols)) array, which
    at millions of pool columns would blow up memory/time regardless of chunk size.

    Yields per-chunk arrays rather than one Python scalar tuple per hit: at tens of
    millions of candidate pairs, boxing each (row, col, score, rank) as individual
    Python objects costs far more memory than the actual data (measured, live: this was
    the dominant cost in a run that otherwise had every other stage under control).
    """
    if len(rows) == 0 or len(cols) == 0 or k <= 0:
        return
    Bt = B[cols].T.tocsr()
    kk = min(k, len(cols))
    chunk = max(1, min(chunk_cap, 5000))
    for st in range(0, len(rows), chunk):
        r = rows[st:st + chunk]
        S = (A[r] @ Bt).tocsr()
        indptr, indices, data = S.indptr, S.indices, S.data
        out_i, out_j, out_s, out_rk = [], [], [], []
        for i in range(len(r)):
            lo, hi = indptr[i], indptr[i + 1]
            idx, sc = _row_topk(indices[lo:hi], data[lo:hi], kk, min_score)
            if idx.size:
                out_i.append(np.full(idx.size, r[i], dtype=np.int64))
                out_j.append(cols[idx])
                out_s.append(sc.astype(np.float32))
                out_rk.append(np.arange(1, idx.size + 1, dtype=np.int32))
        if out_i:
            yield (np.concatenate(out_i), np.concatenate(out_j),
                  np.concatenate(out_s), np.concatenate(out_rk))


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


def _sparse_row_dict(S: sparse.csr_matrix, i: int) -> dict:
    lo, hi = S.indptr[i], S.indptr[i + 1]
    return dict(zip(S.indices[lo:hi].tolist(), S.data[lo:hi].tolist()))


def _rare_token_index(ix: BlockingIndex, cfg: dict, rows, cols):
    """score = name_overlap + 0.5*addr_overlap (idf-weighted), zeroed unless the name
    matches or the address shares >=2 rare tokens. Kept sparse end to end (see
    `_topk_sparse`): combining three separate sparse products per row means working
    from each row's nonzero (index -> value) map instead of a dense (chunk, len(cols))
    array, which at millions of pool columns is not just slow but not allocatable.
    Yields one array-tuple per row-chunk (see `_topk_sparse` for why).

    The rare-token CountVectorizer transforms are precomputed once in build_index, not
    here: this function runs once per country group, and redoing a full-corpus transform
    on every call multiplied peak memory by the number of country groups for no reason.
    """
    if ix.s1_rare_name is None:
        return
    k = cfg["rare_token_k"]
    w = ix.rare_idf_diag
    s1n, pn, s1a, pa = ix.s1_rare_name, ix.pool_rare_name, ix.s1_rare_addr, ix.pool_rare_addr
    PnT = (pn[cols] @ w).T.tocsr()
    PaT = pa[cols].T.tocsr()
    PaTw = (pa[cols] @ w).T.tocsr()
    chunk = max(1, min(cfg.get("chunk_size", 2000), 5000))
    for st in range(0, len(rows), chunk):
        r = rows[st:st + chunk]
        Sname = (s1n[r] @ PnT).tocsr()
        Sacnt = (s1a[r] @ PaT).tocsr()
        Saw = (s1a[r] @ PaTw).tocsr()
        out_i, out_j, out_s, out_rk = [], [], [], []
        for i in range(len(r)):
            name_row = _sparse_row_dict(Sname, i)
            aw_row = _sparse_row_dict(Saw, i)
            if not name_row and not aw_row:
                continue
            acnt_row = _sparse_row_dict(Sacnt, i)
            scored = {}
            for j in set(name_row) | set(aw_row):
                ns = name_row.get(j, 0.0)
                sc = ns + 0.5 * aw_row.get(j, 0.0)
                if ns <= 0 and acnt_row.get(j, 0.0) < 2:
                    continue
                if sc > 0:
                    scored[j] = sc
            top = sorted(scored.items(), key=lambda kv: -kv[1])[:k]
            if top:
                out_i.append(np.full(len(top), r[i], dtype=np.int64))
                out_j.append(cols[np.array([j for j, _ in top], dtype=np.int64)])
                out_s.append(np.array([sc for _, sc in top], dtype=np.float32))
                out_rk.append(np.arange(1, len(top) + 1, dtype=np.int32))
        if out_i:
            yield (np.concatenate(out_i), np.concatenate(out_j),
                  np.concatenate(out_s), np.concatenate(out_rk))


def _key_blocks(ix: BlockingIndex, cfg: dict):
    """Exact-key blocks that rescue typo-heavy pairs the vector indexes rank low.
    Yields one array-tuple per row-chunk (see `_topk_sparse` for why)."""
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
    s1_keys = keys(ix.s1)
    chunk = max(1, min(cfg.get("chunk_size", 2000), 5000))
    for st in range(0, len(s1_keys), chunk):
        out_i, out_j, out_s, out_rk = [], [], [], []
        for i in range(st, min(st + chunk, len(s1_keys))):
            hits = defaultdict(int)
            for k in s1_keys[i]:
                js = pool_map.get(k, ())
                if 0 < len(js) <= cap:
                    for j in js:
                        hits[j] += 1
            if hits:
                out_i.append(np.full(len(hits), i, dtype=np.int64))
                out_j.append(np.fromiter(hits.keys(), dtype=np.int64, count=len(hits)))
                out_s.append(np.fromiter(hits.values(), dtype=np.float32, count=len(hits)))
                out_rk.append(np.ones(len(hits), dtype=np.int32))
        if out_i:
            yield (np.concatenate(out_i), np.concatenate(out_j),
                  np.concatenate(out_s), np.concatenate(out_rk))


def _index_frame(name: str, chunked_iters: list) -> pd.DataFrame | None:
    """Concatenate one index's per-chunk (i, j, score, rank) arrays into a DataFrame.

    Columnar (NumPy-backed) accumulation, not a Python dict keyed by every candidate
    pair: at tens of millions of pairs, a dict entry's own object overhead (~200+ bytes,
    independent of the four numbers it holds) was the dominant memory cost in this
    pipeline — enough on its own to exhaust the box's RAM before blocking finished.
    """
    i_parts, j_parts, s_parts, r_parts = [], [], [], []
    for it in chunked_iters:
        for i_arr, j_arr, s_arr, r_arr in it:
            i_parts.append(i_arr); j_parts.append(j_arr); s_parts.append(s_arr); r_parts.append(r_arr)
    if not i_parts:
        return None
    return pd.DataFrame({
        "s1_idx": np.concatenate(i_parts),
        "pool_idx": np.concatenate(j_parts),
        f"blk_{name}_score": np.concatenate(s_parts),
        f"blk_{name}_rank": np.concatenate(r_parts),
    })


def generate_candidates(ix: BlockingIndex, cfg: dict) -> pd.DataFrame:
    """Returns one row per (s1_id, cand_id) with per-index score/rank columns (NaN = not produced)."""
    _mem("generate_candidates start")
    cap = cfg["chunk_size"]
    char_iters, word_iters, rare_iters = [], [], []
    for rows, cols in _country_groups(ix, cfg["country_mode"]):
        char_iters.append(_topk_sparse(ix.s1_char, ix.pool_char, rows, cols, cfg["char_tfidf_k"], cap))
        word_iters.append(_topk_sparse(ix.s1_word, ix.pool_word, rows, cols, cfg["word_tfidf_k"], cap))
        rare_iters.append(_rare_token_index(ix, cfg, rows, cols))

    frames = {}
    frames["char"] = _index_frame("char", char_iters)
    _mem(f"char index frame built ({0 if frames['char'] is None else len(frames['char']):,} rows)")
    frames["word"] = _index_frame("word", word_iters)
    _mem(f"word index frame built ({0 if frames['word'] is None else len(frames['word']):,} rows)")
    frames["rare"] = _index_frame("rare", rare_iters)
    _mem(f"rare index frame built ({0 if frames['rare'] is None else len(frames['rare']):,} rows)")
    if cfg["country_mode"] != "all" and cfg.get("cross_country_k", 0) > 0:
        frames["xc"] = _index_frame("xc", [_topk_sparse(
            ix.s1_char, ix.pool_char, np.arange(len(ix.s1)), np.arange(len(ix.pool)),
            cfg["cross_country_k"], cap)])
        _mem(f"xc index frame built ({0 if frames['xc'] is None else len(frames['xc']):,} rows)")
    frames["key"] = _index_frame("key", [_key_blocks(ix, cfg)])
    _mem(f"key index frame built ({0 if frames['key'] is None else len(frames['key']):,} rows)")
    frames = {k: v for k, v in frames.items() if v is not None}

    if not frames:
        return pd.DataFrame(columns=["s1_idx", "pool_idx", "s1_id", "cand_id"])
    out = None
    for name, df in frames.items():
        out = df if out is None else out.merge(df, on=["s1_idx", "pool_idx"], how="outer")
        _mem(f"merged {name} -> {len(out):,} rows")

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
