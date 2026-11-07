"""Training, evaluation and MLflow logging.

Every run logs parameters, metrics on a time-ordered holdout, per-slice AUCs, the
reference feature distributions for drift monitoring, and the model itself with its
signature. The new version is registered under the alias ``challenger``; only the gate
(``fdm.gate``) moves ``champion``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import monitor
from .data import CATEGORICAL, FEATURES, NUMERIC, TARGET

MODEL_NAME = "freight-delay"
# MLflow stores sklearn models with skops, which refuses types it cannot verify. These are
# the ones our own pipelines need; the models are produced by this code, never uploaded.
SKOPS_TRUSTED = [
    "functools.partial",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.utils.validation.check_array",
]


def time_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier pickups, test on the latest ones (no peeking into the future)."""
    df = df.sort_values("pickup_date", kind="stable")
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]


def build(kind: str = "gbm", **params) -> Pipeline:
    if kind == "logreg":
        pre = ColumnTransformer(
            [
                ("num", StandardScaler(), NUMERIC),
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
            ]
        )
        return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=params.get("C", 1.0)))])
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )
    clf = HistGradientBoostingClassifier(
        learning_rate=params.get("learning_rate", 0.05),
        max_iter=params.get("max_iter", 400),
        max_leaf_nodes=params.get("max_leaf_nodes", 31),
        l2_regularization=params.get("l2_regularization", 1.0),
        categorical_features=list(range(len(NUMERIC), len(NUMERIC) + len(CATEGORICAL))),
        early_stopping=True,
        validation_fraction=0.1,
        random_state=params.get("seed", 0),
    )
    return Pipeline([("pre", pre), ("clf", clf)])


class CarrierRate:
    """Baseline a forwarder already has: the carrier's historical late rate."""

    def fit(self, df: pd.DataFrame) -> CarrierRate:
        self.global_ = float(df[TARGET].mean())
        g = df.groupby("carrier_id")[TARGET].agg(["sum", "count"])
        self.rate_ = ((g["sum"] + 10 * self.global_) / (g["count"] + 10)).to_dict()  # smoothed
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = df["carrier_id"].map(self.rate_).fillna(self.global_).to_numpy()
        return np.column_stack([1 - p, p])


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: does a predicted 30% mean 30% late?"""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(
        sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    )


SLICES = {
    "domestic": lambda d: d["origin"] == d["destination"],
    "international": lambda d: d["origin"] != d["destination"],
    "reefer": lambda d: d["load_type"] == "reefer",
    "ferry": lambda d: d["ferry"] == "yes",
    "bad_weather": lambda d: d["weather_severity"] >= 2,
}


def evaluate(model, df: pd.DataFrame) -> dict:
    y = df[TARGET].to_numpy()
    p = model.predict_proba(df[FEATURES] if hasattr(model, "named_steps") else df)[:, 1]
    out = {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "late_rate": float(y.mean()),
        "n": int(len(y)),
    }
    for name, f in SLICES.items():
        m = f(df).to_numpy()
        if m.sum() >= 50 and len(np.unique(y[m])) == 2:
            out[f"auc_{name}"] = float(roc_auc_score(y[m], p[m]))
    return out


def choose(train: pd.DataFrame, params: dict) -> tuple[str, dict]:
    """Pick the model family on a validation slice cut from the end of the training data.

    The test set is not touched, so its numbers stay an honest estimate.
    """
    fit, val = time_split(train, test_frac=0.15)
    scores = {}
    for k in ("logreg", "gbm"):
        m = build(k, **params).fit(fit[FEATURES], fit[TARGET])
        scores[k] = evaluate(m, val)["auc"]
    best = max(scores, key=scores.get)
    if best == "gbm" and scores["gbm"] - scores["logreg"] < 0.002:
        best = "logreg"  # not worth the extra complexity
    return best, scores


def run(
    df: pd.DataFrame,
    *,
    kind: str = "auto",
    params: dict | None = None,
    tags: dict | None = None,
    register: bool = True,
    refit_full: bool = True,
    log=print,
) -> dict:
    """Train on ``df`` (time split), log to the active MLflow tracking URI, register."""
    import mlflow
    from mlflow.models import infer_signature

    params = params or {}
    train, test = time_split(df)
    val_scores = {}
    if kind == "auto":
        kind, val_scores = choose(train, params)
    model = build(kind, **params).fit(train[FEATURES], train[TARGET])
    metrics = evaluate(model, test)
    base = evaluate(CarrierRate().fit(train), test)
    if refit_full:
        # The holdout is the *latest* data, which after a drift is exactly the data the
        # new model must learn from. Metrics above come from the holdout; the model that
        # is registered is refitted on everything, and the gate then judges it on a
        # golden set neither version has seen.
        model = build(kind, **params).fit(df[FEATURES], df[TARGET])
    mlflow.set_experiment("freight-delay")
    with mlflow.start_run(tags=tags or {}) as r:
        mlflow.log_params(
            {"kind": kind, "rows_train": len(train), "rows_test": len(test), "refit_full": refit_full, **params}
        )
        mlflow.log_param("data_from", train["pickup_date"].min())
        mlflow.log_param("data_to", test["pickup_date"].max())
        mlflow.log_metrics({f"val_auc_{k}": v for k, v in val_scores.items()})
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "n"})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in base.items() if k in ("auc", "brier")})
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "reference.json"
            ref.write_text(json.dumps(monitor.reference(train)), "utf-8")
            mlflow.log_artifact(str(ref))
        sig = infer_signature(test[FEATURES].head(50), model.predict_proba(test[FEATURES].head(50))[:, 1])
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=sig,
            input_example=test[FEATURES].head(3),
            registered_model_name=MODEL_NAME if register else None,
            skops_trusted_types=SKOPS_TRUSTED,
        )
        version = None
        if register:
            client = mlflow.MlflowClient()
            version = str(info.registered_model_version)
            client.set_registered_model_alias(MODEL_NAME, "challenger", version)
        log(
            f"run {r.info.run_id[:8]}  {kind}  AUC {metrics['auc']:.3f} (carrier-rate baseline {base['auc']:.3f})"
            f"  Brier {metrics['brier']:.3f}  ECE {metrics['ece']:.3f}"
            + (f"  -> {MODEL_NAME} v{version} @challenger" if version else "")
        )
    return {
        "run_id": r.info.run_id,
        "version": version,
        "kind": kind,
        "val_auc": val_scores,
        "metrics": metrics,
        "baseline": base,
    }
