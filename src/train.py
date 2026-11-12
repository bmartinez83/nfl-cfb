"""Walk-forward model training.

Models (scikit-learn):
  * Win probability: LogisticRegression on MARGIN_FEATURES.
  * Margin: Ridge regression on MARGIN_FEATURES -> fair spread (home margin).
  * Total: Ridge regression on TOTAL_FEATURES -> fair total.

WALK-FORWARD ONLY: a model used for a given week is fit on games from
strictly earlier (season, week) — fit_bundle() takes an explicit training
frame and callers (backtest.py) slice it; no random splits anywhere.

Residual SDs (needed to turn point predictions into cover probabilities)
are estimated honestly inside the training window: fit on all-but-the-last
season of the window, measure residuals on that held-out last season, then
refit on the full window for the shipped coefficients.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import MARGIN_FEATURES, TOTAL_FEATURES, build_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def _pipe(model) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def fit_bundle(train: pd.DataFrame) -> dict:
    """Fit all three models on `train` (played games only) + residual SDs."""
    train = train[train["home_score"].notna()].copy()
    if len(train) < 100:
        raise ValueError(f"training frame too small: {len(train)} games")

    seasons = sorted(train["season"].unique())
    holdout_season = seasons[-1]
    fit_part = train[train["season"] < holdout_season]
    hold_part = train[train["season"] == holdout_season]
    if len(fit_part) < 100 or len(hold_part) < 30:
        fit_part, hold_part = train.iloc[:-272], train.iloc[-272:]

    # Residual SDs from the within-window holdout.
    m_hold = _pipe(Ridge(alpha=10.0)).fit(fit_part[MARGIN_FEATURES], fit_part["margin"])
    margin_sd = float(np.std(hold_part["margin"] - m_hold.predict(hold_part[MARGIN_FEATURES])))
    t_hold = _pipe(Ridge(alpha=10.0)).fit(fit_part[TOTAL_FEATURES], fit_part["total_pts"])
    total_sd = float(np.std(hold_part["total_pts"] - t_hold.predict(hold_part[TOTAL_FEATURES])))

    bundle = {
        "win_model": _pipe(LogisticRegression(max_iter=1000, C=1.0)).fit(
            train[MARGIN_FEATURES], train["home_win"].fillna(0.0)),
        "margin_model": _pipe(Ridge(alpha=10.0)).fit(train[MARGIN_FEATURES], train["margin"]),
        "total_model": _pipe(Ridge(alpha=10.0)).fit(train[TOTAL_FEATURES], train["total_pts"]),
        "margin_sd": margin_sd,
        "total_sd": total_sd,
        "n_train": int(len(train)),
        "trained_through": f"{int(train['season'].max())} wk {int(train[train['season'] == train['season'].max()]['week'].max())}",
    }
    return bundle


def predict_frame(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model_prob_home"] = bundle["win_model"].predict_proba(df[MARGIN_FEATURES])[:, 1]
    out["model_margin"] = bundle["margin_model"].predict(df[MARGIN_FEATURES])
    out["model_total"] = bundle["total_model"].predict(df[TOTAL_FEATURES])
    return out


def main() -> int:
    df = build_features()
    played = df[df["home_score"].notna()]
    bundle = fit_bundle(played)
    MODELS.mkdir(parents=True, exist_ok=True)
    with open(MODELS / "model_bundle.pkl", "wb") as fh:
        pickle.dump(bundle, fh)
    meta = {k: bundle[k] for k in ("margin_sd", "total_sd", "n_train", "trained_through")}
    (MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"trained on {bundle['n_train']} games through {bundle['trained_through']}")
    print(f"residual SDs (holdout season): margin {bundle['margin_sd']:.2f} pts, "
          f"total {bundle['total_sd']:.2f} pts")
    print(f"saved: {MODELS / 'model_bundle.pkl'}, {MODELS / 'model_meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
