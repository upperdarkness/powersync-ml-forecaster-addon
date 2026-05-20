#!/usr/bin/env python3
"""
Fast synthetic reference test for v4.2.1 origin-based training.
No Home Assistant required. This intentionally uses a simple ridge-like least squares
model so it runs quickly on small systems; the deployed AppDaemon app uses
HistGradientBoostingRegressor.
"""
import sys
import math
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))


def install_app_import_stubs() -> None:
    try:
        import requests  # noqa: F401
    except ModuleNotFoundError:
        requests_stub = types.ModuleType("requests")
        requests_stub.get = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("requests is unavailable"))
        sys.modules["requests"] = requests_stub

    try:
        import joblib  # noqa: F401
    except ModuleNotFoundError:
        joblib_stub = types.ModuleType("joblib")
        joblib_stub.dump = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("joblib is unavailable"))
        joblib_stub.load = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("joblib is unavailable"))
        sys.modules["joblib"] = joblib_stub

    try:
        import sklearn.ensemble  # noqa: F401
        import sklearn.metrics  # noqa: F401
    except ModuleNotFoundError:
        sklearn_stub = types.ModuleType("sklearn")
        ensemble_stub = types.ModuleType("sklearn.ensemble")
        metrics_stub = types.ModuleType("sklearn.metrics")

        class UnavailableHistGradientBoostingRegressor:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("scikit-learn is unavailable")

        def mean_absolute_error(y_true, y_pred):
            return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))

        ensemble_stub.HistGradientBoostingRegressor = UnavailableHistGradientBoostingRegressor
        metrics_stub.mean_absolute_error = mean_absolute_error
        sklearn_stub.ensemble = ensemble_stub
        sklearn_stub.metrics = metrics_stub
        sys.modules["sklearn"] = sklearn_stub
        sys.modules["sklearn.ensemble"] = ensemble_stub
        sys.modules["sklearn.metrics"] = metrics_stub


install_app_import_stubs()

from powersync_ml_forecaster import DEFAULT_CFG, PowerSyncMLForecaster  # noqa: E402

rng = np.random.default_rng(42)
interval_min = 5
days = 35
points_per_day = 24 * 60 // interval_min
n = days * points_per_day
hours = (np.arange(n) * interval_min / 60.0) % 24
day_index = np.arange(n) // points_per_day
dow = day_index % 7
weekend = (dow >= 5).astype(float)
temp = 22 + 8 * np.sin(2 * np.pi * (hours - 14) / 24) + rng.normal(0, 1.2, n)
baseline = 0.55 + 0.45 * ((hours >= 6) & (hours <= 8)) + 0.8 * ((hours >= 17) & (hours <= 21)) + 0.2 * weekend
baseline = baseline + 0.04 * np.maximum(temp - 26, 0) + rng.normal(0, 0.06, n)
ev = np.zeros(n)
for d in range(6, days, 8):
    start = d * points_per_day + 19 * 60 // interval_min
    ev[start:start + 36] = 7.0
raw = baseline + ev
baseline_excluded = np.maximum(raw - np.where(ev >= 0.5, ev, 0), 0)

origin_step = 120 // interval_min
lead_step = 120 // interval_min
horizon_steps = 48 * 60 // interval_min
min_origin = 7 * points_per_day
max_origin = n - horizon_steps - 1
rows = []
y = []
for origin in range(min_origin, max_origin, origin_step):
    hist = baseline_excluded[:origin+1]
    roll24 = hist[-points_per_day:].mean()
    roll168 = hist[-7*points_per_day:].mean()
    for lead in range(lead_step, horizon_steps + 1, lead_step):
        target = origin + lead
        h = hours[target]
        d = dow[target]
        lag24_valid = target - points_per_day <= origin
        lag48_valid = target - 2*points_per_day <= origin
        lag168 = baseline_excluded[target - 7*points_per_day]
        row = [
            1.0,
            math.sin(2*math.pi*h/24), math.cos(2*math.pi*h/24),
            math.sin(2*math.pi*d/7), math.cos(2*math.pi*d/7),
            float(d >= 5),
            lead * interval_min,
            roll24, roll168,
            baseline_excluded[target - points_per_day] if lag24_valid else 0.0,
            float(lag24_valid),
            baseline_excluded[target - 2*points_per_day] if lag48_valid else 0.0,
            float(lag48_valid),
            lag168,
            temp[target],
        ]
        rows.append(row)
        y.append(baseline_excluded[target])
X = np.asarray(rows, dtype=float)
y = np.asarray(y, dtype=float)
cut = int(len(y) * 0.7)
X_train, X_val = X[:cut], X[cut:]
y_train, y_val = y[:cut], y[cut:]
# Ridge-like least squares for the reference harness only.
alpha = 0.1
coef = np.linalg.solve(X_train.T @ X_train + alpha * np.eye(X_train.shape[1]), X_train.T @ y_train)
pred = np.clip(X_val @ coef, 0, None)
mae = np.mean(np.abs(y_val - pred))
p168 = np.mean(np.abs(y_val - X_val[:, 13]))
print(f"Rows: train={len(y_train):,} validation={len(y_val):,}")
print(f"Reference model MAE: {mae:.3f} kW")
print(f"Persistence 168h MAE: {p168:.3f} kW")
print(f"Delta vs persistence: {(mae - p168) / p168 * 100:.1f}%")
print("Synthetic reference completed")


def run_odd_minute_origin_training_test() -> None:
    odd_start = datetime(2026, 1, 1, 10, 16, tzinfo=timezone.utc)
    test_days = 12
    periods = test_days * points_per_day
    timestamps = [odd_start + timedelta(minutes=interval_min * i) for i in range(periods)]
    test_hours = (np.arange(periods) * interval_min / 60.0 + odd_start.hour + odd_start.minute / 60.0) % 24
    test_day_index = np.arange(periods) // points_per_day
    test_weekend = ((test_day_index % 7) >= 5).astype(float)
    test_load = (
        0.65
        + 0.35 * ((test_hours >= 6) & (test_hours <= 9))
        + 0.75 * ((test_hours >= 17) & (test_hours <= 21))
        + 0.15 * test_weekend
        + rng.normal(0, 0.03, periods)
    )
    history = [
        {"last_changed": ts.isoformat(), "state": f"{float(load):.5f}"}
        for ts, load in zip(timestamps, test_load)
    ]

    app = PowerSyncMLForecaster.__new__(PowerSyncMLForecaster)
    app.cfg = dict(DEFAULT_CFG)
    app.cfg.update({
        "load_sensor": "sensor.synthetic_odd_minute_load",
        "load_unit_override": "kW",
        "timezone": "UTC",
        "base_interval_minutes": interval_min,
        "training_origin_interval_minutes": 60,
        "training_lead_interval_minutes": 15,
        "horizon_hours": 48,
        "weather_enabled": False,
        "public_holiday_region": None,
        "additional_feature_sensors": [],
        "max_training_rows": 500000,
    })
    app.tz = timezone.utc
    app.holiday_cal = None
    app.log = lambda *_args, **_kwargs: None
    app.get_state = lambda *_args, **_kwargs: "kW"

    end = timestamps[-1]
    base = app._build_base_frame({"load": history}, odd_start, end)
    expected_first_index = pd.Timestamp("2026-01-01T10:15:00Z")
    assert base.index[0] == expected_first_index, f"base index was not interval aligned: {base.index[0]}"

    validation_cutoff = odd_start + timedelta(days=8)
    train_rows, val_rows = app._build_origin_training_table(base, odd_start, end, validation_cutoff)
    assert not train_rows.empty, "odd-minute synthetic history produced no training rows"
    assert not val_rows.empty, "odd-minute synthetic history produced no validation rows"

    shifted_base = base.copy()
    shifted_base.index = shifted_base.index + pd.Timedelta(minutes=1)
    shifted_train_rows, shifted_val_rows = app._build_origin_training_table(shifted_base, odd_start, end, validation_cutoff)
    assert not shifted_train_rows.empty, "nearest origin matching produced no training rows"
    assert not shifted_val_rows.empty, "nearest origin matching produced no validation rows"

    late_cutoff = end
    fallback_train_rows, fallback_val_rows = app._build_origin_training_table(base, odd_start, end, late_cutoff)
    assert not fallback_train_rows.empty, "late validation cutoff fallback produced no training rows"
    assert not fallback_val_rows.empty, "late validation cutoff fallback produced no validation rows"
    print(f"Odd-minute origin rows: train={len(train_rows):,} validation={len(val_rows):,}")


run_odd_minute_origin_training_test()
