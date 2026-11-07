"""Drift and performance monitoring from the prediction log.

* Feature drift: Population Stability Index of each feature in recent traffic vs the
  training reference that was logged with the model. Rule of thumb: < 0.1 stable,
  0.1-0.25 watch, > 0.25 act.
* Prediction drift: PSI of the predicted probability.
* Performance: once delivery outcomes arrive, AUC and calibration on recent shipments
  compared with what the model scored at training time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CATEGORICAL, NUMERIC

WATCH, ACT = 0.10, 0.25


def reference(df: pd.DataFrame, bins: int = 10) -> dict:
    ref: dict = {"numeric": {}, "categorical": {}}
    for c in NUMERIC:
        edges = np.unique(np.quantile(df[c], np.linspace(0, 1, bins + 1)))
        hist = np.histogram(df[c], bins=np.r_[-np.inf, edges[1:-1], np.inf])[0]
        ref["numeric"][c] = {"edges": edges[1:-1].tolist(), "freq": (hist / hist.sum()).tolist()}
    for c in CATEGORICAL:
        vc = df[c].astype(str).value_counts(normalize=True)
        ref["categorical"][c] = vc.to_dict()
    return ref


def _psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    e = np.clip(expected, eps, None)
    a = np.clip(actual, eps, None)
    return float(((a - e) * np.log(a / e)).sum())


def feature_psi(ref: dict, df: pd.DataFrame) -> dict[str, float]:
    out = {}
    for c, r in ref["numeric"].items():
        hist = np.histogram(df[c], bins=np.r_[-np.inf, r["edges"], np.inf])[0]
        out[c] = _psi(np.array(r["freq"]), hist / max(hist.sum(), 1))
    for c, r in ref["categorical"].items():
        vc = df[c].astype(str).value_counts(normalize=True)
        cats = sorted(set(r) | set(vc.index))  # unseen categories count as drift
        out[c] = _psi(np.array([r.get(k, 0.0) for k in cats]), np.array([vc.get(k, 0.0) for k in cats]))
    return out


def status(psi: float) -> str:
    return "act" if psi > ACT else ("watch" if psi > WATCH else "ok")


def read_log(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text("utf-8").splitlines() if line.strip()]
    return pd.DataFrame(
        [
            {
                **r["features"],
                "p_late": r["p_late"],
                "shipment_id": r["shipment_id"],
                "model_version": r["model_version"],
            }
            for r in rows
        ]
    )


def report(
    ref: dict,
    recent: pd.DataFrame,
    outcomes: pd.DataFrame | None = None,
    train_auc: float | None = None,
    max_auc_drop: float = 0.03,
) -> dict:
    from sklearn.metrics import roc_auc_score

    from .train import ece

    psi = feature_psi(ref, recent)
    worst = max(psi, key=psi.get)
    out: dict = {
        "rows": int(len(recent)),
        "feature_psi": {k: round(v, 4) for k, v in sorted(psi.items(), key=lambda kv: -kv[1])},
        "worst_feature": worst,
        "worst_psi": round(psi[worst], 4),
        "mean_p_late": float(recent["p_late"].mean()) if "p_late" in recent else None,
    }
    alerts = [f"{k} PSI {v:.2f}" for k, v in psi.items() if v > ACT]
    reasons: list[str] = []
    if outcomes is not None and len(outcomes):
        j = recent.drop(columns=["late"], errors="ignore").merge(outcomes[["shipment_id", "late"]], on="shipment_id")
        if len(j) >= 100 and j["late"].nunique() == 2:
            auc = float(roc_auc_score(j["late"], j["p_late"]))
            out["performance"] = {
                "labelled": int(len(j)),
                "auc": auc,
                "train_auc": train_auc,
                "ece": ece(j["late"].to_numpy(), j["p_late"].to_numpy()),
                "observed_late_rate": float(j["late"].mean()),
                "predicted_late_rate": float(j["p_late"].mean()),
            }
            if train_auc is not None and auc < train_auc - max_auc_drop:
                reasons.append(f"AUC {auc:.3f} vs {train_auc:.3f} at training")
            gap = out["performance"]["observed_late_rate"] - out["performance"]["predicted_late_rate"]
            if abs(gap) > 0.05:
                reasons.append(f"late rate {gap:+.0%} vs predicted (calibration)")
    # Drift alone is a reason to look, not to retrain: a model that has the drifting
    # feature as input may handle it. With outcomes we decide on performance; without
    # them (labels arrive days later) strong drift is all we have.
    if "performance" not in out:
        reasons = alerts
    out["drift_alerts"] = alerts
    out["retrain"] = bool(reasons)
    out["reasons"] = reasons
    return out
