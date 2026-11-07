"""The whole model lifecycle in one offline run.

1. Train v1 on summer shipments (model family chosen on validation) -> gate -> champion.
2. A deliberately weak challenger (5 trees) is rejected by the gate.
3. Week A, winter: the API serves 3 000 shipments. Weather drifts hard, but the model
   has weather as an input, so once outcomes arrive the monitor sees performance and
   calibration holding: drift alert, no retrain.
4. Week B, winter + new subcontractors: carriers the model has never seen and that are
   less reliable. Calibration breaks (late rate under-predicted): retrain.
5. Retrain on everything labelled so far, gate on a fresh golden set from the new world,
   promote, and hot-reload the API.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from . import data, gate, monitor, train


class _Api:
    """The FastAPI app in-process, writing its prediction log to a temp dir."""

    def __init__(self, log_path: Path) -> None:
        os.environ["FDM_PREDICTION_LOG"] = str(log_path)
        os.environ["FDM_SKIP_LOAD"] = "1"
        import importlib

        from fastapi.testclient import TestClient

        from . import api

        self.api = importlib.reload(api)  # pick up the log path
        self.api.load()
        self.client = TestClient(self.api.app)
        self.log_path = log_path

    def predict(self, df: pd.DataFrame) -> list[dict]:
        payload = df[["shipment_id", *data.FEATURES]].to_dict(orient="records")
        out = []
        for i in range(0, len(payload), 500):
            r = self.client.post("/predict", json=payload[i : i + 500])
            r.raise_for_status()
            out += r.json()
        return out


def _week(api: _Api, df: pd.DataFrame, ref: dict, train_auc: float, log) -> dict:
    api.log_path.unlink(missing_ok=True)
    preds = api.predict(df)
    recent = monitor.read_log(api.log_path)
    rep = monitor.report(ref, recent, df[["shipment_id", "late"]], train_auc=train_auc)
    perf = rep["performance"]
    top = list(rep["feature_psi"].items())[:2]
    log(f"    {len(preds)} predictions by v{preds[0]['model_version']}")
    log("    drift: " + ", ".join(f"{k} PSI {v:.2f} ({monitor.status(v)})" for k, v in top))
    log(
        f"    outcomes: late {perf['observed_late_rate']:.1%} vs predicted {perf['predicted_late_rate']:.1%},"
        f" AUC {perf['auc']:.3f} (at training {train_auc:.3f})"
    )
    log(f"    retrain: {rep['retrain']}" + (f"  ({'; '.join(rep['reasons'])})" if rep["reasons"] else ""))
    return rep


def run(out_path: Path, log=print) -> dict:
    import mlflow

    res: dict = {}
    summer = data.generate(30000, seed=1, start="2026-05-01")
    log(f"[1] summer history: {len(summer)} shipments, {summer.late.mean():.1%} late")
    v1 = train.run(summer, tags={"stage": "initial"}, log=log)
    log(f"    family chosen on validation: {v1['kind']}  {v1['val_auc']}")
    _, golden_summer = train.time_split(summer)
    g1 = gate.run(golden_summer, log=log)
    res["v1"] = {k: v1[k] for k in ("kind", "val_auc", "metrics", "baseline")} | {"gate": g1["checks"]}

    log("[2] weak challenger (5 trees)")
    weak = train.run(summer, kind="gbm", params={"max_iter": 5}, tags={"stage": "weak"}, log=log)
    g2 = gate.run(golden_summer, log=log)
    res["weak"] = {"metrics": weak["metrics"], "promoted": g2["promoted"], "checks": g2["checks"]}

    ref = mlflow.artifacts.load_dict(f"runs:/{v1['run_id']}/reference.json")
    with tempfile.TemporaryDirectory() as tmp:
        api = _Api(Path(tmp) / "predictions.jsonl")
        log("[3] week A: winter")
        week_a = data.generate(3000, seed=2, start="2026-12-01", scenario="winter")
        res["week_a"] = _week(api, week_a, ref, v1["metrics"]["auc"], log)

        log("[4] week B: winter + new subcontractors")
        week_b = data.generate(3000, seed=4, start="2026-12-08", scenario="winter+new_carriers")
        res["week_b"] = _week(api, week_b, ref, v1["metrics"]["auc"], log)

        log("[5] retrain on everything labelled, gate on a fresh golden set from the new world")
        v3 = train.run(pd.concat([summer, week_a, week_b], ignore_index=True), tags={"stage": "retrain"}, log=log)
        golden = data.generate(4000, seed=5, start="2026-12-15", scenario="winter+new_carriers")
        g3 = gate.run(golden, log=log)
        reloaded = api.client.post("/reload").json() if g3["promoted"] else None
        if reloaded:
            log(f"    API reloaded: now serving v{reloaded['model_version']}")
        res["retrain"] = {"kind": v3["kind"], "metrics": v3["metrics"], "gate": g3, "api_after_reload": reloaded}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, default=str), "utf-8")
    log(f"-> {out_path}")
    return res
