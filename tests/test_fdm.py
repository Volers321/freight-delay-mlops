import os

import numpy as np
import pandas as pd
import pytest

from fdm import data, gate, monitor, train


@pytest.fixture(scope="module")
def summer():
    return data.generate(6000, seed=1)


def test_generator_is_deterministic_and_has_all_columns(summer):
    again = data.generate(6000, seed=1)
    pd.testing.assert_frame_equal(summer, again)
    assert set(data.FEATURES + [data.TARGET]) <= set(summer.columns)
    assert 0.1 < summer.late.mean() < 0.45


def test_scenarios_shift_the_world(summer):
    winter = data.generate(6000, seed=1, scenario="winter")
    assert winter.weather_severity.mean() > summer.weather_severity.mean() + 0.5
    assert winter.late.mean() > summer.late.mean()
    nc = data.generate(6000, seed=1, scenario="new_carriers")
    assert nc.carrier_id.nunique() > summer.carrier_id.nunique()
    with pytest.raises(ValueError):
        data.generate(10, scenario="mars")


def test_time_split_never_trains_on_the_future(summer):
    tr, te = train.time_split(summer)
    assert tr.pickup_date.max() <= te.pickup_date.min()


def test_model_beats_carrier_rate_baseline(summer):
    tr, te = train.time_split(summer)
    m = train.build("gbm").fit(tr[data.FEATURES], tr[data.TARGET])
    ours, base = train.evaluate(m, te), train.evaluate(train.CarrierRate().fit(tr), te)
    assert ours["auc"] > base["auc"] + 0.02
    assert ours["ece"] < 0.06


def test_unknown_carrier_does_not_crash(summer):
    tr, _ = train.time_split(summer)
    m = train.build("gbm").fit(tr[data.FEATURES], tr[data.TARGET])
    row = tr[data.FEATURES].head(1).assign(carrier_id="C999")
    assert 0 <= m.predict_proba(row)[0, 1] <= 1


def test_ece_is_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(int)
    assert train.ece(y, p) < 0.01
    assert train.ece(y, np.clip(p + 0.2, 0, 1)) > 0.1


def test_psi_flags_weather_in_winter(summer):
    ref = monitor.reference(summer)
    same = monitor.feature_psi(ref, data.generate(3000, seed=9))
    assert max(same.values()) < monitor.WATCH
    winter = monitor.feature_psi(ref, data.generate(3000, seed=9, scenario="winter"))
    assert winter["weather_severity"] > monitor.ACT
    nc = monitor.feature_psi(ref, data.generate(3000, seed=9, scenario="new_carriers"))
    assert nc["carrier_id"] > monitor.ACT


def test_drift_alone_does_not_retrain_when_outcomes_are_good(summer):
    ref = monitor.reference(summer)
    w = data.generate(3000, seed=9, scenario="winter")
    recent = w.assign(p_late=0.5 * w.late.mean() + 0.5 * w.late)  # informative, calibrated on average
    rep = monitor.report(ref, recent, w[["shipment_id", "late"]], train_auc=None)
    assert rep["drift_alerts"] and not rep["retrain"]
    rep_no_labels = monitor.report(ref, recent)
    assert rep_no_labels["retrain"]


def _m(auc, brier=0.15, ece=0.02, **slices):
    return {"auc": auc, "brier": brier, "ece": ece, **{f"auc_{k}": v for k, v in slices.items()}}


def test_gate_rules():
    ok, _ = gate.decide(_m(0.80), None)
    assert ok
    ok, _ = gate.decide(_m(0.65), None)
    assert not ok
    ok, _ = gate.decide(_m(0.81, ferry=0.80), _m(0.80, ferry=0.80))
    assert ok
    ok, checks = gate.decide(_m(0.81, ferry=0.70), _m(0.80, ferry=0.80))
    assert not ok and any("ferry" in c for c in checks)
    ok, _ = gate.decide(_m(0.80, brier=0.20), _m(0.80, brier=0.15))
    assert not ok
    ok, _ = gate.decide(_m(0.80, ece=0.09), _m(0.79))
    assert not ok


@pytest.fixture
def mlflow_tmp(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns.db'}")
    yield mlflow


def test_train_gate_serve_roundtrip(summer, mlflow_tmp, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    r = train.run(summer, log=lambda *_: None)
    assert r["version"] == "1"
    _, golden = train.time_split(summer)
    assert gate.run(golden, log=lambda *_: None)["promoted"]

    monkeypatch.setenv("FDM_PREDICTION_LOG", str(tmp_path / "pred.jsonl"))
    monkeypatch.setenv("FDM_SKIP_LOAD", "1")
    import importlib

    from fdm import api

    importlib.reload(api)
    api.load()
    c = TestClient(api.app)
    assert c.get("/health").json() == {"status": "ok", "model_version": "1"}
    rows = summer.head(5)[["shipment_id", *data.FEATURES]].to_dict(orient="records")
    out = c.post("/predict", json=rows).json()
    assert len(out) == 5 and all(0 <= o["p_late"] <= 1 for o in out)
    bad = dict(rows[0], load_type="banana")
    assert c.post("/predict", json=[bad]).status_code == 422
    logged = monitor.read_log(tmp_path / "pred.jsonl")
    assert len(logged) == 5 and set(data.FEATURES) <= set(logged.columns)
    assert os.path.exists(tmp_path / "pred.jsonl")
