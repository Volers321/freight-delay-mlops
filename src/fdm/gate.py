"""Promotion gate: challenger replaces champion only if it is not worse where it matters.

Both models are scored on the same *golden set*: the latest labelled shipments, which
neither model trained on. The rules are written down here, versioned with the code,
and the decision with its numbers is logged to the challenger's MLflow run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .train import MODEL_NAME, evaluate


@dataclass(frozen=True)
class Rules:
    min_auc: float = 0.70  # absolute floor
    max_auc_drop: float = 0.005  # vs champion, overall
    max_slice_drop: float = 0.02  # vs champion, any slice
    max_brier_ratio: float = 1.02  # calibration must not get worse by more than 2%
    max_ece: float = 0.05


def decide(challenger: dict, champion: dict | None, rules: Rules | None = None) -> tuple[bool, list[str]]:
    rules = rules or Rules()
    checks = []
    ok = challenger["auc"] >= rules.min_auc
    checks.append(f"{'PASS' if ok else 'FAIL'} AUC {challenger['auc']:.3f} >= floor {rules.min_auc}")
    e_ok = challenger["ece"] <= rules.max_ece
    checks.append(f"{'PASS' if e_ok else 'FAIL'} ECE {challenger['ece']:.3f} <= {rules.max_ece}")
    ok &= e_ok
    if champion is None:
        checks.append("PASS no champion yet")
        return ok, checks
    d = challenger["auc"] - champion["auc"]
    a_ok = d >= -rules.max_auc_drop
    checks.append(
        f"{'PASS' if a_ok else 'FAIL'} AUC {challenger['auc']:.3f} vs champion {champion['auc']:.3f} ({d:+.3f})"
    )
    b_ok = challenger["brier"] <= champion["brier"] * rules.max_brier_ratio
    checks.append(f"{'PASS' if b_ok else 'FAIL'} Brier {challenger['brier']:.4f} vs champion {champion['brier']:.4f}")
    ok &= a_ok & b_ok
    for k in sorted(champion):
        if k.startswith("auc_") and k in challenger:
            sd = challenger[k] - champion[k]
            s_ok = sd >= -rules.max_slice_drop
            if not s_ok:
                checks.append(f"FAIL slice {k[4:]}: {challenger[k]:.3f} vs {champion[k]:.3f}")
            ok &= s_ok
    if ok:
        checks.append(f"PASS no slice regressed by more than {rules.max_slice_drop}")
    return bool(ok), checks


def run(golden: pd.DataFrame, rules: Rules | None = None, log=print) -> dict:
    import mlflow

    rules = rules or Rules()
    client = mlflow.MlflowClient()
    ch = client.get_model_version_by_alias(MODEL_NAME, "challenger")
    challenger = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@challenger")
    m_ch = evaluate(challenger, golden)
    try:
        cp = client.get_model_version_by_alias(MODEL_NAME, "champion")
        m_cp = evaluate(mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion"), golden)
    except mlflow.exceptions.MlflowException:
        cp, m_cp = None, None
    if cp is not None and cp.version == ch.version:
        log(f"v{ch.version} is already champion")
        return {"promoted": False, "checks": ["challenger is champion"]}
    promoted, checks = decide(m_ch, m_cp, rules)
    for c in checks:
        log("  " + c)
    with mlflow.start_run(run_id=ch.run_id):
        mlflow.log_metrics({f"golden_{k}": v for k, v in m_ch.items() if k != "n"})
        mlflow.log_dict(
            {
                "rules": asdict(rules),
                "checks": checks,
                "promoted": promoted,
                "champion_version": cp.version if cp else None,
            },
            "gate.json",
        )
    if promoted:
        client.set_registered_model_alias(MODEL_NAME, "champion", ch.version)
        log(f"-> v{ch.version} promoted to @champion" + (f" (was v{cp.version})" if cp else ""))
    else:
        log(f"-> v{ch.version} rejected; champion stays v{cp.version if cp else '-'}")
    return {
        "promoted": promoted,
        "checks": checks,
        "challenger": m_ch,
        "champion": m_cp,
        "challenger_version": ch.version,
        "champion_version": cp.version if cp else None,
    }
