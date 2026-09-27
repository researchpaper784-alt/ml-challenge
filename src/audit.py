"""Phase 1 — data audit. Prints every number the methodology doc should cite."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import pandas as pd

from .normalize import base_clean
from .score import macro_score


def _scripts(texts) -> Counter:
    c = Counter()
    for t in texts:
        for ch in t:
            if ch.isalpha():
                c[unicodedata.name(ch, "?").split()[0]] += 1
    return c


def run_audit(src: dict[str, pd.DataFrame], truth: dict[str, set[str]] | None) -> dict:
    rep = {}
    for s, df in src.items():
        rep[f"{s}_rows"] = len(df)
        rep[f"{s}_country_counts"] = df["country"].value_counts().to_dict()
        rep[f"{s}_empty_name"] = float((df["business_name"] == "").mean())
        rep[f"{s}_empty_addr"] = float((df["business_address"] == "").mean())
    allrec = pd.concat(src.values())
    rep["country_values_exact"] = sorted(allrec["country"].unique().tolist())
    rep["scripts"] = dict(_scripts(allrec["business_name"] + " " + allrec["business_address"]).most_common(8))
    rep["postcode_patterns"] = Counter(
        len(m) for a in allrec["business_address"] for m in re.findall(r"(?<!\d)\d{5,6}(?!\d)", a)).most_common()
    for c, g in allrec.groupby("country"):
        toks = Counter(t for x in g["business_name"] for t in re.findall(r"\w+", base_clean(x)))
        rep[f"top_name_tokens_{c}"] = toks.most_common(25)
        toks = Counter(t for x in g["business_address"] for t in re.findall(r"[^\W\d]+", base_clean(x)))
        rep[f"top_addr_tokens_{c}"] = toks.most_common(25)

    if truth is not None:
        s1_ids = set(src["source1"]["entity_id"])
        other = set(src["source2"]["entity_id"]) | set(src["source3"]["entity_id"])
        n = pd.Series({e: len(t) for e, t in truth.items()})
        rep["gt_entities"] = len(truth)
        rep["gt_entities_missing_from_s1"] = len(set(truth) - s1_ids)
        rep["s1_missing_from_gt"] = len(s1_ids - set(truth))
        rep["gt_ids_missing_from_s2s3"] = sum(c not in other for t in truth.values() for c in t)
        rep["match_count_dist"] = n.clip(upper=3).value_counts(normalize=True).sort_index().round(4).to_dict()
        rep["s2_count_dist"] = pd.Series({e: sum(c.startswith("S2-") for c in t) for e, t in truth.items()}
                                         ).clip(upper=3).value_counts(normalize=True).sort_index().round(4).to_dict()
        rep["s3_count_dist"] = pd.Series({e: sum(c.startswith("S3-") for c in t) for e, t in truth.items()}
                                         ).clip(upper=3).value_counts(normalize=True).sort_index().round(4).to_dict()
        rep["entities_with_2plus_from_same_source"] = float(pd.Series(
            {e: max(sum(c.startswith("S2-") for c in t), sum(c.startswith("S3-") for c in t))
             for e, t in truth.items()}).ge(2).mean())
        # does a pool record belong to more than one S1 entity?
        owner = Counter(c for t in truth.values() for c in t)
        rep["pool_ids_matched_to_2plus_s1"] = sum(v > 1 for v in owner.values())
        cty = dict(zip(allrec["entity_id"], allrec["country"]))
        pairs = [(e, c) for e, t in truth.items() for c in t]
        rep["cross_country_true_pairs"] = sum(cty.get(e) != cty.get(c) for e, c in pairs)
        rep["true_pairs"] = len(pairs)
        rep["all_empty_baseline_f05"] = macro_score({}, truth)
    return rep
