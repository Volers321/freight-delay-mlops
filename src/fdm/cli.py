"""Command line: ``fdm <command>``.

fdm demo                         # whole lifecycle offline: train, gate, serve, drift, retrain
fdm generate out.csv --n 30000 [--scenario winter --start 2026-12-01]
fdm train data.csv [--kind auto|gbm|logreg]   # logs to MLflow, registers @challenger
fdm gate golden.csv              # challenger vs champion on the golden set
fdm monitor logs/predictions.jsonl [--outcomes labels.csv]   # exit code 3 = retrain
fdm serve                        # uvicorn on :8000

MLflow location comes from MLFLOW_TRACKING_URI (default: sqlite:///mlruns.db).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from . import data


def _mlflow() -> None:
    import logging
    import warnings

    import mlflow

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    warnings.filterwarnings("ignore")
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    logging.getLogger("alembic").setLevel(logging.ERROR)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"))


def cmd_generate(a) -> None:
    df = data.generate(a.n, seed=a.seed, start=a.start, scenario=a.scenario)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} shipments, late {df.late.mean():.1%} -> {a.out}")


def cmd_train(a) -> None:
    from . import train

    _mlflow()
    train.run(pd.read_csv(a.data), kind=a.kind)


def cmd_gate(a) -> int:
    from . import gate

    _mlflow()
    return 0 if gate.run(pd.read_csv(a.golden))["promoted"] else 2


def _champion_reference():
    import mlflow

    client = mlflow.MlflowClient()
    from .train import MODEL_NAME

    mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    ref = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/reference.json")
    auc = client.get_run(mv.run_id).data.metrics.get("auc")
    return ref, auc


def cmd_monitor(a) -> int:
    from . import monitor

    _mlflow()
    ref, auc = _champion_reference()
    recent = monitor.read_log(Path(a.log))
    outcomes = pd.read_csv(a.outcomes) if a.outcomes else None
    rep = monitor.report(ref, recent, outcomes, train_auc=auc)
    print(json.dumps(rep, indent=1))
    return 3 if rep["retrain"] else 0


def cmd_serve(a) -> None:
    import uvicorn

    _mlflow()
    uvicorn.run("fdm.api:app", host="0.0.0.0", port=a.port)


def cmd_demo(a) -> int:
    from . import demo

    _mlflow()
    demo.run(Path(a.out))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fdm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("out")
    g.add_argument("--n", type=int, default=30000)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--start", default="2026-05-01")
    g.add_argument("--scenario", choices=["baseline", "winter", "new_carriers"], default="baseline")
    g.set_defaults(fn=cmd_generate)
    t = sub.add_parser("train")
    t.add_argument("data")
    t.add_argument("--kind", choices=["auto", "gbm", "logreg"], default="auto")
    t.set_defaults(fn=cmd_train)
    q = sub.add_parser("gate")
    q.add_argument("golden")
    q.set_defaults(fn=cmd_gate)
    m = sub.add_parser("monitor")
    m.add_argument("log")
    m.add_argument("--outcomes")
    m.set_defaults(fn=cmd_monitor)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    d = sub.add_parser("demo")
    d.add_argument("--out", default="results/demo.json")
    d.set_defaults(fn=cmd_demo)
    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
