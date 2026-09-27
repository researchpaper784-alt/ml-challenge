"""CLI for the whole pipeline.

  python -m src.run_pipeline audit     --split train
  python -m src.run_pipeline baseline  --split test          # all-empty, format-valid submission
  python -m src.run_pipeline train                           # block + features + CV + tune + final model
  python -m src.run_pipeline predict   --split test          # writes output/*.tsv
  python -m src.run_pipeline holdout-country                 # France proxy: train on one country, test on other
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time

import joblib
import numpy as np
import pandas as pd
import yaml

from . import io_utils as io
from .audit import run_audit
from .blocking import blocking_report, build_index, generate_candidates
from .decide import decide, tune
from .features import feature_columns, pair_features
from .model import PairModel, oof_predict
from .normalize import normalize_frame
from .pairs import entity_folds, label_pairs, training_mask
from .score import breakdown, macro_score


def load_cfg(path):
    with open(path) as f:
        raw = f.read()
    cfg = yaml.safe_load(raw)
    cfg["_hash"] = hashlib.sha1(raw.encode()).hexdigest()[:10]
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=2, ensure_ascii=False, default=str)


def prepare(split_dir, cfg):
    """normalize -> block -> features. Returns everything downstream steps need."""
    t0 = time.time()
    src = io.load_sources(split_dir)
    s1 = normalize_frame(src["source1"]).reset_index(drop=True)
    pool = normalize_frame(pd.concat([src["source2"], src["source3"]], ignore_index=True)).reset_index(drop=True)
    print(f"[prep] normalized {len(s1):,} S1 / {len(pool):,} S2+S3 in {time.time()-t0:.1f}s")
    ix = build_index(s1, pool, cfg["blocking"])
    cands = generate_candidates(ix, cfg["blocking"])
    print(f"[prep] blocking -> {len(cands):,} candidate pairs in {time.time()-t0:.1f}s")
    F = pair_features(ix, cands)
    print(f"[prep] features {F.shape} in {time.time()-t0:.1f}s")
    return src, s1, pool, ix, cands, F


def cand_lists(cands):
    return cands.groupby("s1_id")["cand_id"].apply(list).to_dict()


# ------------------------------------------------------------------ commands
def cmd_audit(args, cfg):
    d = os.path.join(cfg["paths"]["dataset_dir"], args.split)
    src = io.load_sources(d)
    truth = io.load_ground_truth(d) if args.split == "train" else None
    rep = run_audit(src, truth)
    dump(rep, os.path.join(cfg["paths"]["artifacts_dir"], f"audit_{args.split}.json"))
    for k, v in rep.items():
        print(f"{k}: {v}")


def write_submission(split_dir, cfg, s1_ids, matches, cands_map, pool_ids):
    io.assert_submission(s1_ids, matches, cands_map, pool_ids)
    out = cfg["paths"]["output_dir"]
    io.write_id_lists(os.path.join(out, "matching_results.tsv"), s1_ids, matches,
                      ("source1_entity_id", "matched_entity_ids"))
    io.write_id_lists(os.path.join(out, "candidate_pairs.tsv"), s1_ids, cands_map,
                      ("source1_entity_id", "candidate_entity_ids"))
    print(f"[out] wrote {out}/matching_results.tsv and candidate_pairs.tsv ({len(s1_ids):,} rows each)")


def cmd_baseline(args, cfg):
    d = os.path.join(cfg["paths"]["dataset_dir"], args.split)
    src = io.load_sources(d)
    s1_ids = src["source1"]["entity_id"].tolist()
    pool_ids = set(src["source2"]["entity_id"]) | set(src["source3"]["entity_id"])
    write_submission(d, cfg, s1_ids, {}, {}, pool_ids)


def cmd_train(args, cfg):
    seed = cfg["seed"]
    art = cfg["paths"]["artifacts_dir"]
    d = os.path.join(cfg["paths"]["dataset_dir"], "train")
    truth_all = io.load_ground_truth(d)
    src, s1, pool, ix, cands, F = prepare(d, cfg)
    s1_ids = s1["entity_id"].tolist()
    truth = {e: truth_all.get(e, set()) for e in s1_ids}       # every S1 entity, singletons included
    country = dict(zip(s1["entity_id"], s1["country"]))

    blk = blocking_report(cands, truth, len(s1), len(pool))
    print("[blocking]", json.dumps(_jsonable(blk)))
    # per-index contribution: recall if that index were the only one
    per_index = {}
    for col in [c for c in cands.columns if c.endswith("_score") and c.startswith("blk_")]:
        sub = cands[cands[col].notna()]
        if sub.empty:
            continue
        per_index[col] = blocking_report(sub, truth, len(s1), len(pool))["recall_ceiling"]
    blk["recall_by_index_alone"] = per_index
    print("[blocking] recall by index alone:", per_index)

    y = label_pairs(cands, truth)
    folds = entity_folds(truth, country, cfg["pairs"]["cv_folds"], seed)
    fold_row = cands["s1_id"].map(folds).to_numpy()
    tmask = training_mask(F, y, cands["s1_id"].to_numpy(), cfg["pairs"]["max_neg_per_entity"])
    X = F[feature_columns(F)]
    print(f"[pairs] {len(y):,} pairs, {y.sum():,} positive, training rows {tmask.sum():,} "
          f"(neg:pos = {(tmask & (y == 0)).sum() / max(y.sum(), 1):.1f})")

    oof = oof_predict(X, y, fold_row, tmask, cfg["model"], seed)
    from sklearn.metrics import average_precision_score, roc_auc_score
    pair_metrics = {"oof_auc": roc_auc_score(y, oof), "oof_pr_auc": average_precision_score(y, oof),
                    "pos_score_quantiles": np.quantile(oof[y == 1], [.05, .25, .5, .75, .95]).round(4).tolist(),
                    "neg_score_quantiles": np.quantile(oof[y == 0], [.5, .9, .99, .999]).round(4).tolist()}
    print("[model] OOF", pair_metrics)

    params, grid = tune(cands["s1_id"].to_numpy(), cands["cand_id"].to_numpy(), oof, y, truth, cfg["decision"])
    print("[decide] chosen", params)
    preds = decide(cands["s1_id"].to_numpy(), cands["cand_id"].to_numpy(), oof, params)
    bd = breakdown({k: set(v) for k, v in preds.items()}, truth,
                   {k: set(v) for k, v in cand_lists(cands).items()}, country)
    print("[eval] OOF macro F0.5 =", round(bd["macro_f05"], 4))
    print("[eval] breakdown", json.dumps(_jsonable({k: v for k, v in bd.items()}), indent=1))

    final = PairModel(cfg["model"], seed).fit(X[tmask], y[tmask])
    os.makedirs(art, exist_ok=True)
    joblib.dump(final, os.path.join(art, "model.joblib"))
    dump(params, os.path.join(art, "decision.json"))
    grid.to_csv(os.path.join(art, "decision_grid.tsv"), sep="\t", index=False)
    imp = final.importances()
    if len(imp):
        imp.to_csv(os.path.join(art, "feature_importance.tsv"), sep="\t", header=["gain"])
    dump({"config_hash": cfg["_hash"], "blocking": blk, "pair_metrics": pair_metrics,
          "decision": params, "oof_breakdown": bd,
          "all_empty_baseline": macro_score({}, truth)}, os.path.join(art, "train_report.json"))
    print(f"[train] artifacts saved to {art}/")


def cmd_predict(args, cfg):
    art = cfg["paths"]["artifacts_dir"]
    d = os.path.join(cfg["paths"]["dataset_dir"], args.split)
    model: PairModel = joblib.load(os.path.join(art, "model.joblib"))
    with open(os.path.join(art, "decision.json")) as f:
        params = json.load(f)
    src, s1, pool, ix, cands, F = prepare(d, cfg)
    p = model.predict_proba(F[feature_columns(F)]) if len(cands) else np.array([])
    matches = decide(cands["s1_id"].to_numpy(), cands["cand_id"].to_numpy(), p, params)
    s1_ids = s1["entity_id"].tolist()
    pool_ids = set(pool["entity_id"])
    write_submission(d, cfg, s1_ids, matches, cand_lists(cands), pool_ids)

    n_match = pd.Series({e: len(matches.get(e, [])) for e in s1_ids})
    stats = {"split": args.split, "config_hash": cfg["_hash"],
             "pred_empty_rate": float((n_match == 0).mean()),
             "pred_count_dist": n_match.clip(upper=3).value_counts(normalize=True).sort_index().round(4).to_dict(),
             "by_country_empty_rate": n_match.groupby(s1.set_index("entity_id")["country"]).apply(
                 lambda s: round(float((s == 0).mean()), 4)).to_dict(),
             "n_candidates": len(cands)}
    print("[predict]", stats)
    if args.split == "train":        # sanity: in-sample score (optimistic, not a validation number)
        truth = io.load_ground_truth(d)
        print("[predict] in-sample F0.5 (optimistic):",
              round(macro_score({k: set(v) for k, v in matches.items()},
                                {e: truth.get(e, set()) for e in s1_ids}), 4))
    log = os.path.join(art, "leaderboard_log.tsv")
    new = not os.path.exists(log)
    with open(log, "a") as f:
        if new:
            f.write("timestamp\tconfig_hash\tsplit\toof_f05\tpublic_score\n")
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{cfg['_hash']}\t{args.split}\t"
                f"{params.get('val_f05', ''):.4f}\t\n")


def cmd_holdout_country(args, cfg):
    """France proxy: fit + tune on one country, score on another never seen in training."""
    seed = cfg["seed"]
    d = os.path.join(cfg["paths"]["dataset_dir"], "train")
    truth_all = io.load_ground_truth(d)
    src, s1, pool, ix, cands, F = prepare(d, cfg)
    truth = {e: truth_all.get(e, set()) for e in s1["entity_id"]}
    country = dict(zip(s1["entity_id"], s1["country"]))
    y = label_pairs(cands, truth)
    X = F[feature_columns(F)]
    cc = cands["s1_id"].map(country).to_numpy()
    results = {}
    for held in sorted(set(country.values())):
        tr_ents = {e: t for e, t in truth.items() if country[e] != held}
        te_ents = {e: t for e, t in truth.items() if country[e] == held}
        if not tr_ents or not te_ents:
            continue
        tr_rows, te_rows = cc != held, cc == held
        folds = entity_folds(tr_ents, country, 3, seed)
        fold_row = cands["s1_id"].map(folds).fillna(-1).to_numpy().astype(int)
        tmask = training_mask(F, y, cands["s1_id"].to_numpy(), cfg["pairs"]["max_neg_per_entity"])
        oof = oof_predict(X[tr_rows], y[tr_rows], fold_row[tr_rows], tmask[tr_rows], cfg["model"], seed)
        params, _ = tune(cands["s1_id"].to_numpy()[tr_rows], cands["cand_id"].to_numpy()[tr_rows],
                         oof, y[tr_rows], tr_ents, cfg["decision"])
        m = PairModel(cfg["model"], seed).fit(X[tr_rows & tmask], y[tr_rows & tmask])
        p = m.predict_proba(X[te_rows])
        preds = decide(cands["s1_id"].to_numpy()[te_rows], cands["cand_id"].to_numpy()[te_rows], p, params)
        results[held] = {"in_country_oof_f05": params["val_f05"],
                         "unseen_country_f05": macro_score({k: set(v) for k, v in preds.items()}, te_ents),
                         "unseen_country_all_empty": macro_score({}, te_ents)}
        print(f"[holdout] held out {held}: {results[held]}")
    dump(results, os.path.join(cfg["paths"]["artifacts_dir"], "holdout_country.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["audit", "baseline", "train", "predict", "holdout-country"])
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    seed_all(cfg["seed"])
    {"audit": cmd_audit, "baseline": cmd_baseline, "train": cmd_train, "predict": cmd_predict,
     "holdout-country": cmd_holdout_country}[args.command](args, cfg)


if __name__ == "__main__":
    main()
