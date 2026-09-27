# Business Entity Resolution — Amazon ML Challenge

Pipeline: **normalize → block (recall) → pair features → LightGBM → decision layer (tuned on macro F0.5)**.
`candidate_pairs.tsv` is exactly the set the model scores; `matching_results.tsv` is always a subset of it.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Place the official data so that `dataset/train/*_source{1,2,3}.tsv`, `dataset/train/*_ground_truth.tsv`
and `dataset/test/*_source{1,2,3}.tsv` exist (or edit `paths.dataset_dir` in `config.yaml`).
No network access is used at any point; no external data, geocoding or registries.

## Run (from this directory)
```bash
python -m src.run_pipeline audit --split train        # Phase 1: numbers for the write-up -> artifacts/audit_train.json
python -m src.run_pipeline baseline --split test      # all-empty, format-valid submission (singleton credit)
python -m src.run_pipeline train                      # blocking report, 5-fold grouped OOF, decision tuning, final model
python -m src.run_pipeline predict --split test       # -> output/matching_results.tsv, output/candidate_pairs.tsv
python -m src.run_pipeline holdout-country            # unseen-country (France) proxy
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test   # official validator (from student_resource)
```
`artifacts/train_report.json` holds recall ceiling, reduction ratio, OOF AUC/PR-AUC, chosen thresholds and the
loss decomposition (blocking loss / model loss / false merges / singleton errors, per country and per match count).
`artifacts/leaderboard_log.tsv` gets one row per `predict`; paste the public score in by hand.

## Layout
| File | Phase | Role |
|---|---|---|
| `src/io_utils.py` | 0 | strict TSV I/O (`sep="\t"`, `dtype=str`, no NA coercion, no quoting) + submission asserts |
| `src/audit.py` | 1 | data audit |
| `src/normalize.py` | 2 | name/address normalization, pattern-based postcodes, landmark bag, phonetic key |
| `src/blocking.py` | 3 | union of char TF-IDF, word TF-IDF, rare-token index, exact-key blocks; same-country-first with fallback |
| `src/pairs.py` | 4 | entity-level stratified folds, labels, hard-negative capping |
| `src/features.py` | 5 | ~65 name/address/context features incl. within-entity rank/margin, popularity, mutual-best |
| `src/model.py` | 6 | LightGBM classifier + grouped OOF |
| `src/decide.py` | 7 | threshold / margin / per-source cap / emit bar, grid-tuned on macro F0.5 with plateau smoothing |
| `src/score.py` | 8 | exact metric + breakdown |
| `tests/` | – | unit tests and a synthetic-data generator for smoke tests only |

## Models and licences
| Component | Licence | Params |
|---|---|---|
| LightGBM | MIT | GBDT, ≪ 8B |
| scikit-learn (TF-IDF only; not the final model) | BSD-3 | – |
| rapidfuzz (string similarity) | MIT | – |

`model.backend: sklearn_hgb` exists only so the code runs where LightGBM is not installable; the submission uses LightGBM.
Train and predict must use the same environment (rapidfuzz and LightGBM installed): the pure-Python
fallbacks give slightly different feature values. Seeds are fixed in `config.yaml`; the config hash is logged with every run.
