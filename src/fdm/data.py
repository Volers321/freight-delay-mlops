"""Synthetic road-freight shipments with a known cause structure for late delivery.

A shipment is *late* when it arrives more than two hours after the agreed slot. The
generator draws the delay from causes a forwarder would recognise: distance, border
crossings, ferries, weather, the carrier's (hidden) reliability, Friday traffic, driver
hours left, refrigerated or ADR loads. Two scenarios change the world after a model has
been trained, which is what the monitoring has to catch:

* ``winter``: more snow and ice, so weather severity goes up and so do delays;
* ``new_carriers``: half of the loads go to new, cheaper subcontractors the model has
  never seen (and which are less reliable).

Scenarios combine with ``+``, e.g. ``winter+new_carriers``. Some causes interact (a
ferry in a storm, a reefer on a long haul, a weak carrier on an international lane), so
an additive model cannot capture everything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COUNTRIES = {"PL": 0, "DE": 1, "CZ": 1, "SK": 1, "LT": 1, "NL": 2, "FR": 2, "IT": 2, "SE": 2, "ES": 3}
LOAD_TYPES = ["FTL", "LTL", "reefer", "ADR"]
NUMERIC = ["distance_km", "border_crossings", "weight_t", "driver_hours_left", "weather_severity", "pickup_hour"]
CATEGORICAL = ["origin", "destination", "load_type", "carrier_id", "weekday", "ferry"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "late"


def _carriers(rng: np.random.Generator, n: int, offset: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "carrier_id": [f"C{offset + i:03d}" for i in range(n)],
            "reliability": rng.normal(0, 0.8, n),  # hidden: higher = more delays
            "share": rng.dirichlet(np.full(n, 1.5)),
        }
    )


def generate(
    n: int,
    seed: int = 0,
    start: str = "2026-05-01",
    scenario: str = "baseline",
    n_carriers: int = 40,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    carriers = _carriers(np.random.default_rng(1000), n_carriers)  # same fleet across calls
    scenarios = set(scenario.split("+"))
    if not scenarios <= {"baseline", "winter", "new_carriers"}:
        raise ValueError(f"unknown scenario {scenario!r}")
    if "new_carriers" in scenarios:
        new = _carriers(np.random.default_rng(2000), 15, offset=n_carriers)
        new["reliability"] += 0.6  # cheaper subcontractors, less reliable
        new["share"] *= 0.5
        carriers = carriers.assign(share=carriers["share"] * 0.5)
        carriers = pd.concat([carriers, new], ignore_index=True)
    c = carriers.iloc[rng.choice(len(carriers), size=n, p=carriers["share"] / carriers["share"].sum())]

    names = list(COUNTRIES)
    origin = np.where(rng.random(n) < 0.7, "PL", rng.choice(names, n))
    dest = rng.choice(names, n, p=np.array([0.3, 0.2, 0.05, 0.03, 0.04, 0.1, 0.1, 0.08, 0.05, 0.05]))
    ring = np.array([COUNTRIES[o] for o in origin]), np.array([COUNTRIES[d] for d in dest])
    hops = np.abs(ring[0] - ring[1]) + (origin != dest)
    distance = np.where(origin == dest, rng.uniform(60, 650, n), 150 + 380 * hops + rng.normal(0, 120, n))
    distance = np.clip(distance, 40, None)
    ferry = (dest == "SE") | (origin == "SE") | (rng.random(n) < 0.02)
    borders = np.where(origin == dest, 0, np.maximum(np.abs(ring[0] - ring[1]), 1)).astype(int)
    load = rng.choice(LOAD_TYPES, n, p=[0.55, 0.25, 0.12, 0.08])
    weight = np.where(load == "LTL", rng.uniform(0.5, 8, n), rng.uniform(8, 24, n))
    dates = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, 90, n), unit="D")
    weekday = dates.day_name().str[:3].to_numpy()
    hour = rng.integers(5, 20, n)
    hours_left = np.round(rng.uniform(1, 9, n), 1)
    wmean = 1.6 if "winter" in scenarios else 0.5
    weather = np.clip(rng.poisson(wmean, n), 0, 3)

    drive_h = distance / 65
    z = (
        -3.7
        + 0.35 * np.log1p(distance / 300)
        + 0.25 * borders
        + 0.6 * ferry
        + 0.8 * ferry * (weather >= 2)  # storms cancel sailings
        + 0.55 * weather
        + c["reliability"].to_numpy() * (1 + 0.6 * (origin != dest))  # weak carriers struggle abroad
        + 0.45 * (weekday == "Fri")
        + 0.2 * (load == "reefer")
        + 0.9 * (load == "reefer") * (distance > 1000)  # temperature checks, long haul
        + 0.3 * (load == "ADR")
        + 0.8 * (drive_h > hours_left + 2)  # will need a rest break before delivery
        + 0.2 * (hour >= 16)
        + rng.normal(0, 0.6, n)
    )
    late = rng.random(n) < 1 / (1 + np.exp(-z))
    return pd.DataFrame(
        {
            "shipment_id": [f"S{seed:02d}-{i:06d}" for i in range(n)],
            "pickup_date": dates.strftime("%Y-%m-%d"),
            "origin": origin,
            "destination": dest,
            "distance_km": np.round(distance, 0),
            "border_crossings": borders,
            "ferry": np.where(ferry, "yes", "no"),
            "load_type": load,
            "weight_t": np.round(weight, 1),
            "carrier_id": c["carrier_id"].to_numpy(),
            "weekday": weekday,
            "pickup_hour": hour,
            "driver_hours_left": hours_left,
            "weather_severity": weather,
            TARGET: late.astype(int),
        }
    )
