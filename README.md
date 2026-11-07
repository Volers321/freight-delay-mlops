# freight-delay-mlops

**Will this truck be late? — and how do you keep that model honest after it ships.** A late-delivery risk model for road freight, wrapped in the full lifecycle: MLflow tracking and model registry, a promotion gate with slice checks, a FastAPI service that logs every prediction, drift and performance monitoring, retraining, Docker and Kubernetes manifests. `fdm demo` runs the whole story offline in a few minutes.

🇵🇱 *Model ryzyka spóźnienia dostawy w transporcie drogowym z pełnym cyklem życia: śledzenie eksperymentów i rejestr modeli w MLflow, bramka promocji (challenger vs champion, także na wycinkach danych), API w FastAPI z logiem predykcji, monitoring dryfu (PSI) i jakości po spłynięciu wyników dostaw, retrening, Docker i Kubernetes.*

![CI](https://github.com/Volers321/freight-delay-mlops/actions/workflows/ci.yml/badge.svg)

> The shipments are **synthetic**, generated from causes a forwarder would recognise (see [Data](#data)). The project is about the machinery around a model, not about the model.

## The lifecycle in one run

```
$ fdm demo
[1] summer history: 30000 shipments, 26.1% late
run 9a93…  logreg  AUC 0.811 (carrier-rate baseline 0.760)  Brier 0.142  ECE 0.016  -> freight-delay v1 @challenger
    family chosen on validation: logreg  {'logreg': 0.831, 'gbm': 0.830}
  PASS AUC 0.811 >= floor 0.7 · PASS ECE 0.016 <= 0.05 · PASS no champion yet
-> v1 promoted to @champion
[2] weak challenger (5 trees)
  FAIL ECE 0.125 <= 0.05 · PASS AUC 0.809 vs champion 0.813 (-0.003) · FAIL Brier 0.1694 vs champion 0.1408
-> v2 rejected; champion stays v1
[3] week A: winter
    drift: weather_severity PSI 0.72 (act), carrier_id PSI 0.01 (ok)
    outcomes: late 34.1% vs predicted 34.4%, AUC 0.810 (at training 0.811)
    retrain: False
[4] week B: winter + new subcontractors
    drift: carrier_id PSI 3.50 (act), weather_severity PSI 0.76 (act)
    outcomes: late 46.6% vs predicted 33.1%, AUC 0.758 (at training 0.811)
    retrain: True  (AUC 0.758 vs 0.811 at training; late rate +13% vs predicted (calibration))
[5] retrain on everything labelled, gate on a fresh golden set from the new world
  PASS AUC 0.831 vs champion 0.765 (+0.066) · PASS Brier 0.1667 vs champion 0.2113
  PASS ECE 0.019 · PASS no slice regressed by more than 0.02
-> v3 promoted to @champion (was v1)
    API reloaded: now serving v3
```

(lightly shortened; full numbers in [`results/demo.json`](results/demo.json))

## What each step shows

**1. A baseline that means something.** The model is compared with what a forwarder already uses: each carrier's historical late rate (AUC 0.760). The model reaches 0.811. The family — logistic regression or gradient boosting — is chosen on a validation slice cut from the end of the training data; GBM was not better by the required margin (0.830 vs 0.831), so the simpler model ships.

**2. The gate stops a bad model.** A deliberately underfit challenger has nearly the same AUC (−0.003, within tolerance) but is badly calibrated (ECE 0.125, Brier +20%). An AUC-only check would have promoted it. The gate (`fdm/gate.py`) checks an AUC floor, AUC vs champion, Brier, calibration error, and AUC on five business slices (domestic, international, reefer, ferry, bad weather); the decision and its numbers are logged to the challenger's MLflow run.

**3. Drift is not the same as decay.** In winter the weather feature drifts far beyond the usual action threshold (PSI 0.72 > 0.25). But weather is an input; the model already knows that snow means delays. Once delivery outcomes arrive, AUC and calibration have held, so the monitor raises a drift alert and does **not** ask for a retrain.

**4. Decay.** Half the loads move to new subcontractors the model has never seen (carrier PSI 3.5). The model has no history for them and treats them as average; they are worse. Predicted late rate 33%, observed 47%; AUC down to 0.758. Retrain.

**5. Retrain, judged on new ground.** Both models are scored on a fresh golden set from the new world, which neither trained on: AUC 0.765 → 0.831, calibration error 0.115 → 0.019. Promoted; the API reloads without a restart.

### A bug worth keeping

The first retraining run was rejected by the gate, and the reason was mine, not the model's. Evaluation uses a **time split** — the latest 20% of shipments is held out. After a drift, the latest 20% *is* the new world, so the "retrained" model never saw a single new carrier. Now the registered model is refitted on all data after evaluation (`refit_full=True`, logged as a parameter), and the gate judges it on a golden set neither version has seen.

## Architecture

```
              ┌──────────── MLflow (tracking + registry) ────────────┐
 history.csv ─┤ fdm train  → run: params, metrics, slices,           │
              │              reference.json (feature distributions), │
              │              model + signature → vN @challenger      │
 golden.csv ──┤ fdm gate   → challenger vs champion → @champion      │
              └──────────────────────────┬───────────────────────────┘
                                         │ models:/freight-delay@champion
 TMS / client ── POST /predict ──► FastAPI (pydantic validation) ── predictions.jsonl
                                         ▲                                │
                                POST /reload                              ▼
                                         │              fdm monitor (CronJob, nightly)
                                         │              PSI vs reference · AUC, ECE once
                                         └── retrain ◄─ outcomes arrive · exit code 3
```

| Piece | Where |
|---|---|
| Data generator with scenarios | `src/fdm/data.py` |
| Training, baselines, metrics, MLflow logging | `src/fdm/train.py` |
| Promotion rules | `src/fdm/gate.py` |
| API: `/predict`, `/health`, `/metadata`, `/reload` | `src/fdm/api.py` |
| PSI drift, performance, retrain decision | `src/fdm/monitor.py` |
| Whole lifecycle | `src/fdm/demo.py` |
| Docker image (non-root, healthcheck), compose with MLflow server | `Dockerfile`, `docker-compose.yml` |
| Kubernetes: Deployment, Service, HPA, PVC, monitor CronJob | `k8s/` |
| CI: lint, tests, the full lifecycle, image build + health check | `.github/workflows/ci.yml` |

## Data

`fdm.data.generate` draws shipments (origin/destination country, distance, border crossings, ferry, load type, weight, carrier, weekday, pickup hour, driver hours left, weather) and decides lateness (> 2 h after the slot) from causes with known effects: distance, borders, ferries, weather, the carrier's hidden reliability, Friday traffic, a rest break forced by driver hours, reefer and ADR loads — plus interactions an additive model cannot see (a ferry in a storm, a reefer on a long haul, a weak carrier abroad). Scenarios `winter`, `new_carriers` and `winter+new_carriers` change the world after training.

Because the generator is mostly additive, logistic regression is as good as boosting here; on real data I would expect the opposite more often, which is why the choice is made per training run and not hard-coded.

## Run it

```bash
pip install -e ".[dev]"
fdm demo                     # whole lifecycle, local MLflow in ./mlruns.db
mlflow ui --backend-store-uri sqlite:///mlruns.db   # runs, registry, gate.json per run
pytest -q

# step by step
fdm generate data/history.csv --n 30000
fdm train data/history.csv
fdm generate data/golden.csv --n 4000 --seed 7 --start 2026-07-15
fdm gate data/golden.csv
fdm serve                    # http://localhost:8000/docs
fdm monitor logs/predictions.jsonl --outcomes data/outcomes.csv   # exit code 3 = retrain

# containers
docker compose up -d mlflow && docker compose run --rm trainer && docker compose up -d api
kubectl apply -f k8s/        # expects an MLflow service at mlflow.mlops.svc
```

Tested here: the Python package, the tests and the demo. CI also builds the Docker image and checks that it starts and answers `/health`. The Kubernetes manifests are a starting point; I have not run them on a cluster.

## What I would add next

- Delayed labels done properly: outcomes table keyed by `shipment_id`, joined on a schedule, with the label delay itself monitored.
- Shadow deployment: challenger scores live traffic next to the champion before the gate.
- Cost-based threshold: a forwarder acts on "high risk" (calls the carrier, adds buffer), so the cut-off should come from the cost of a call vs the cost of a late delivery.
- Terraform for the managed version (Azure ML or Vertex AI registry + Container Apps / Cloud Run).

## License

MIT
