"""Strict TSV I/O. Every read/write in the pipeline goes through here."""
from __future__ import annotations

import csv
import glob
import os

import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str) -> pd.DataFrame:
    # dtype=str + keep_default_na=False: empty address stays "" (never NaN), IDs never become floats.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, encoding="utf-8")


def _find(split_dir: str, suffix: str) -> str:
    hits = sorted(glob.glob(os.path.join(split_dir, f"*{suffix}")))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one *{suffix} in {split_dir}, found {hits}")
    return hits[0]


def load_sources(split_dir: str) -> dict[str, pd.DataFrame]:
    out = {}
    for s in ("source1", "source2", "source3"):
        df = read_tsv(_find(split_dir, f"_{s}.tsv"))
        missing = set(SOURCE_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"{s}: missing columns {missing}")
        df = df[SOURCE_COLS].copy()
        for c in SOURCE_COLS:
            df[c] = df[c].str.strip()
        if df["entity_id"].duplicated().any():
            raise ValueError(f"{s}: duplicate entity_id")
        out[s] = df
    return out


def parse_id_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def load_ground_truth(split_dir: str) -> dict[str, set[str]]:
    gt = read_tsv(_find(split_dir, "_ground_truth.tsv"))
    return {r.source1_entity_id: set(parse_id_list(r.matched_entity_ids))
            for r in gt.itertuples(index=False)}


def write_id_lists(path: str, s1_ids: list[str], lists: dict[str, list[str]],
                   header: tuple[str, str]) -> None:
    """One row per S1 id (taken from the S1 file, not from `lists`), sorted, no quoting."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{header[0]}\t{header[1]}\n")
        for sid in sorted(s1_ids):
            ids = sorted(dict.fromkeys(lists.get(sid, [])))
            f.write(f"{sid}\t{','.join(ids)}\n")


def assert_submission(s1_ids, matches, candidates, valid_other_ids) -> None:
    """Hard gate mirroring the official validator's rules."""
    s1_set = set(s1_ids)
    assert len(s1_set) == len(s1_ids), "duplicate S1 ids"
    for name, d in (("matches", matches), ("candidates", candidates)):
        assert set(d) <= s1_set, f"{name}: unknown S1 ids"
        for sid, ids in d.items():
            assert len(ids) == len(set(ids)), f"{name}: duplicate ids for {sid}"
            for i in ids:
                assert not i.startswith("S1-"), f"{name}: S1 id {i} in list for {sid}"
                assert i in valid_other_ids, f"{name}: {i} not in test S2/S3"
    for sid, ids in matches.items():
        extra = set(ids) - set(candidates.get(sid, []))
        assert not extra, f"matches not subset of candidates for {sid}: {extra}"
