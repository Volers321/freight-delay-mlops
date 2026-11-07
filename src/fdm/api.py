"""Prediction service.

Loads ``models:/freight-delay@champion`` at start-up (or on POST /reload, after the gate
promotes a new version), validates input with pydantic, and appends every prediction
with its features and model version to a JSONL log that the monitor reads.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import FEATURES
from .train import MODEL_NAME

LOG = Path(os.environ.get("FDM_PREDICTION_LOG", "logs/predictions.jsonl"))
ALIAS = os.environ.get("FDM_MODEL_ALIAS", "champion")


class Shipment(BaseModel):
    shipment_id: str
    origin: str = Field(min_length=2, max_length=2, examples=["PL"])
    destination: str = Field(min_length=2, max_length=2, examples=["DE"])
    distance_km: float = Field(gt=0, lt=6000)
    border_crossings: int = Field(ge=0, le=6)
    ferry: Literal["yes", "no"] = "no"
    load_type: Literal["FTL", "LTL", "reefer", "ADR"]
    weight_t: float = Field(gt=0, le=40)
    carrier_id: str
    weekday: Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pickup_hour: int = Field(ge=0, le=23)
    driver_hours_left: float = Field(ge=0, le=10)
    weather_severity: int = Field(ge=0, le=3)


class Prediction(BaseModel):
    shipment_id: str
    p_late: float
    risk: Literal["low", "medium", "high"]
    model_version: str


class _State:
    model = None
    version: str | None = None
    run_id: str | None = None
    lock = threading.Lock()


state = _State()


def load(alias: str = ALIAS) -> None:
    import mlflow

    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    with state.lock:
        state.model, state.version, state.run_id = model, str(mv.version), mv.run_id


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("FDM_SKIP_LOAD") != "1":
        load()
    yield


app = FastAPI(title="freight-delay", version="0.1.0", lifespan=lifespan)


def _risk(p: float) -> str:
    return "high" if p >= 0.5 else ("medium" if p >= 0.25 else "low")


def _log(rows: list[dict]) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@app.post("/predict", response_model=list[Prediction])
def predict(items: list[Shipment]) -> list[Prediction]:
    if state.model is None:
        raise HTTPException(503, "no model loaded")
    if not 1 <= len(items) <= 1000:
        raise HTTPException(422, "send 1-1000 shipments")
    df = pd.DataFrame([i.model_dump() for i in items])
    with state.lock:
        p = state.model.predict_proba(df[FEATURES])[:, 1]
        version = state.version
    now = time.time()
    _log(
        [
            {
                "ts": now,
                "shipment_id": i.shipment_id,
                "model_version": version,
                "p_late": float(pi),
                "features": i.model_dump(exclude={"shipment_id"}),
            }
            for i, pi in zip(items, p, strict=True)
        ]
    )
    return [
        Prediction(shipment_id=i.shipment_id, p_late=round(float(pi), 4), risk=_risk(pi), model_version=version)
        for i, pi in zip(items, p, strict=True)
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if state.model is not None else "no_model", "model_version": state.version}


@app.get("/metadata")
def metadata() -> dict:
    return {"model": MODEL_NAME, "alias": ALIAS, "version": state.version, "run_id": state.run_id, "features": FEATURES}


@app.post("/reload")
def reload() -> dict:
    load()
    return {"model_version": state.version}
