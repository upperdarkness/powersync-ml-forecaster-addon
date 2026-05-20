#!/usr/bin/env python3
"""
Fast synthetic reference test for v4.3.0 origin-based training.
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
from standalone_runner import StandaloneForecaster  # noqa: E402

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


def run_statistics_history_fallback_test() -> None:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=30)
    detail_start = end - timedelta(days=10)
    detailed_points = 10 * points_per_day
    detailed_times = [detail_start + timedelta(minutes=interval_min * i) for i in range(detailed_points)]
    detailed_load = 1.0 + 0.25 * np.sin(2 * np.pi * np.arange(detailed_points) / points_per_day)
    detailed_history = [
        {"last_changed": ts.isoformat(), "state": f"{float(value):.5f}"}
        for ts, value in zip(detailed_times, detailed_load)
    ]
    stats_records = []
    hourly_points = 30 * 24
    for i in range(hourly_points):
        ts = start + timedelta(hours=i)
        value = 0.85 + 0.2 * math.sin(2 * math.pi * (ts.hour / 24.0))
        stats_records.append({"start": int(ts.timestamp() * 1000), "mean": value})

    app = PowerSyncMLForecaster.__new__(PowerSyncMLForecaster)
    app.cfg = dict(DEFAULT_CFG)
    app.cfg.update({
        "load_sensor": "sensor.synthetic_load",
        "load_unit_override": "kW",
        "minimum_training_days": 14,
        "base_interval_minutes": interval_min,
        "timezone": "UTC",
    })
    app.history_diagnostics = {}
    app.get_state = lambda *_args, **_kwargs: "kW"
    app.log = lambda *_args, **_kwargs: None

    base = app._build_base_frame({
        "load": detailed_history,
        "load_short_term_statistics": [],
        "load_long_term_statistics": stats_records,
    }, start, end)
    required_points = app.cfg["minimum_training_days"] * 24 * 60 // app.cfg["base_interval_minutes"]
    assert base["baseline_load_kw"].notna().sum() >= required_points
    counts = app.history_diagnostics["history_source_counts"]
    assert counts.get("detailed_state", 0) > 0
    assert counts.get("long_term_statistics", 0) > 0
    stats_rows = base["history_source"] == "long_term_statistics"
    assert (base.loc[stats_rows, "ev_excluded_kw"] == 0).all()
    assert np.allclose(
        base.loc[stats_rows, "baseline_load_kw"].head(20),
        base.loc[stats_rows, "raw_load_kw"].head(20),
    )
    print(
        "Statistics fallback test: "
        f"usable_days={app.history_diagnostics['usable_baseline_days']} "
        f"sources={app.history_diagnostics['history_source_counts']}"
    )


run_statistics_history_fallback_test()


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")


class FakeSession:
    def __init__(self):
        self.posts = []

    def get(self, *_args, **_kwargs):
        return FakeResponse(404, None, "")

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse(201, {"entity_id": "sensor.hafo_power_sync_home_load_forecast"}, "")


class ConstantModel:
    def predict(self, frame):
        return np.full(len(frame), 1.234, dtype=float)


def run_training_cycle_publish_test() -> None:
    app = StandaloneForecaster.__new__(StandaloneForecaster)
    app.args = {}
    app.cfg = dict(DEFAULT_CFG)
    app.cfg.update({
        "load_sensor": "sensor.synthetic_load",
        "hafo_entity_id": "sensor.hafo_power_sync_home_load_forecast",
        "friendly_name": "HAFO Load Forecast",
        "timezone": "UTC",
        "base_interval_minutes": 5,
        "training_lead_interval_minutes": 15,
        "publish_interval_minutes": 5,
        "horizon_hours": 48,
        "weather_enabled": False,
    })
    app.tz = timezone.utc
    app.holiday_cal = None
    app.model = None
    app.feature_cols = []
    app.training_frame = None
    app.validation_metrics = {"model_mae_kw": 0.123}
    app.history_diagnostics = {}
    app.last_train_time = None
    app.last_error = None
    app._session = FakeSession()
    app._headers = {"Authorization": "Bearer fake", "Content-Type": "application/json"}
    app._ha_url = "http://supervisor/core"
    app.logs = []
    app.log = lambda msg, level="INFO", **_kwargs: app.logs.append((level, msg))
    app._save_model_cache = lambda: None
    app._write_latest_forecast = lambda _forecast: None
    app._append_snapshot = lambda _forecast: None
    app._fetch_weather_live = lambda _forecast_days: pd.DataFrame()

    def successful_training():
        idx = pd.date_range("2026-01-01T00:00:00Z", periods=8 * points_per_day, freq="5min")
        app.training_frame = pd.DataFrame({
            "baseline_load_kw": np.full(len(idx), 0.9, dtype=float),
            "ev_charging_flag": np.zeros(len(idx), dtype=float),
        }, index=idx)
        app.model = ConstantModel()
        app.feature_cols = ["lead_minutes"]

    app._train_validate_refit_and_forecast = successful_training

    app._train_callback({})

    forecast_posts = [
        post for post in app._session.posts
        if post["url"] == "http://supervisor/core/api/states/sensor.hafo_power_sync_home_load_forecast"
        and post["json"]["attributes"].get("forecast")
    ]
    assert forecast_posts, (
        "training cycle did not publish the HAFO forecast sensor; "
        f"posts={app._session.posts!r} logs={app.logs!r} last_error={app.last_error!r}"
    )
    payload = forecast_posts[-1]["json"]
    attrs = payload["attributes"]
    assert payload["state"] == "1.234", f"unexpected published state: {payload['state']}"
    assert attrs["status"] == "ok"
    assert attrs["source"] == "powersync_ml_forecaster"
    assert attrs["unit_of_measurement"] == "kW"
    assert attrs["friendly_name"] == "HAFO Load Forecast"
    assert len(attrs["forecast"]) == 576, f"unexpected forecast length: {len(attrs['forecast'])}"
    log_text = "\n".join(msg for _level, msg in app.logs)
    assert "Training cycle complete" in log_text
    assert "Generating forecast for sensor.hafo_power_sync_home_load_forecast" in log_text
    assert "Publishing forecast sensor sensor.hafo_power_sync_home_load_forecast" in log_text
    assert "Published forecast sensor sensor.hafo_power_sync_home_load_forecast status=201" in log_text
    print(f"Training publish test: posts={len(app._session.posts)} forecast_entries={len(attrs['forecast'])}")


run_training_cycle_publish_test()
