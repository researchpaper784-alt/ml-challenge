"""Phase 6 — GBDT pair classifier (LightGBM, MIT). sklearn fallback only for dependency-less smoke tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False


class PairModel:
    def __init__(self, cfg: dict, seed: int):
        self.cfg, self.seed = cfg, seed
        self.backend = cfg["backend"]
        if self.backend == "lightgbm" and not HAVE_LGB:
            print("[model] lightgbm not installed -> falling back to sklearn_hgb (install lightgbm for the real run)")
            self.backend = "sklearn_hgb"
        self.features: list[str] = []
        self.clf = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.features = list(X.columns)
        p = self.cfg["params"]
        if self.backend == "lightgbm":
            self.clf = lgb.LGBMClassifier(objective="binary", random_state=self.seed, n_jobs=-1,
                                          verbose=-1, deterministic=True, force_row_wise=True,
                                          scale_pos_weight=1.0, **p)
            self.clf.fit(X, y)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.clf = HistGradientBoostingClassifier(
                max_iter=p.get("n_estimators", 400), learning_rate=p.get("learning_rate", 0.05),
                max_leaf_nodes=p.get("num_leaves", 31), min_samples_leaf=p.get("min_child_samples", 20),
                l2_regularization=p.get("reg_lambda", 1.0), random_state=self.seed)
            self.clf.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.features)          # missing blocking columns -> NaN
        return self.clf.predict_proba(X)[:, 1]

    def importances(self) -> pd.Series:
        if self.backend == "lightgbm":
            imp = self.clf.booster_.feature_importance(importance_type="gain")
            return pd.Series(imp, index=self.features).sort_values(ascending=False)
        return pd.Series(dtype=float)


def oof_predict(X: pd.DataFrame, y: np.ndarray, fold_of_row: np.ndarray, train_mask: np.ndarray,
                cfg: dict, seed: int) -> np.ndarray:
    """Grouped K-fold (group = S1 entity) out-of-fold probabilities for every candidate row."""
    oof = np.full(len(X), np.nan)
    for f in np.unique(fold_of_row):
        tr = (fold_of_row != f) & train_mask
        te = fold_of_row == f
        m = PairModel(cfg, seed).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])
        print(f"[cv] fold {f}: train rows {tr.sum():,} (pos {y[tr].sum():,}) -> predicted {te.sum():,}")
    return oof
