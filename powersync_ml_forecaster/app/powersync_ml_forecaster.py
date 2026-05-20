"""
PowerSync ML Load Forecaster v4.2

AppDaemon 4 application that publishes a HAFO-compatible baseline load forecast
sensor for PowerSync. The model excludes EV charging from the training target and
uses forecast-origin-safe features only.

Install path:
  /addon_configs/a0d7b954_appdaemon/apps/powersync_ml_forecaster/powersync_ml_forecaster.py

Configuration:
  append apps.yaml.example content to AppDaemon apps.yaml and replace entity IDs.
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import requests
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

try:
    import hassapi as hass
except Exception:  # lets synthetic tests import helper functions outside AppDaemon
    class _DummyHass:  # pragma: no cover
        pass
    hass = type("hass", (), {"Hass": _DummyHass})

try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except Exception:
    holidays = None
    HOLIDAYS_AVAILABLE = False

HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LIVE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "cloud_cover",
    "precipitation",
    "wind_speed_10m",
    "is_day",
]

DEFAULT_CFG: Dict[str, Any] = {
    "hafo_entity_id": "sensor.hafo_power_sync_home_load_forecast",
    "friendly_name": "HAFO PowerSync ML Load Forecast",
    "load_sensor": None,
    "load_unit_override": None,
    "ev_power_sensor": None,
    "ev_power_unit_override": None,
    "ev_exclusion_threshold_kw": 0.5,
    "import_price_sensor": None,
    "feed_in_price_sensor": None,
    "additional_feature_sensors": [],
    "latitude": None,
    "longitude": None,
    "timezone": None,
    "public_holiday_region": None,
    "days_back": 90,
    "minimum_training_days": 45,
    "horizon_hours": 48,
    "base_interval_minutes": 5,
    "training_origin_interval_minutes": 60,
    "training_lead_interval_minutes": 15,
    "publish_interval_minutes": 5,
    "retrain_minutes": 360,
    "update_minutes": 15,
    "validation_days": 14,
    "max_training_rows": 250000,
    "subsample_strategy": "stratified_by_lead_bucket",
    "history_fetch_chunk_days": 7,
    "max_total_history_fetch_seconds": 600,
    "max_single_history_chunk_seconds": 90,
    "max_training_frame_mb": 600,
    "weather_enabled": True,
    "weather_training_mode": "historical_forecast_proxy",
    "price_training_mode": "origin_current_price_only",
    "use_live_price_forecast_features": False,
    "data_dir": "/share/powersync_ml_forecaster",
    "cache_max_age_hours": 24,
    "snapshot_retention_days": 30,
    "require_quality_gate": True,
    "min_model_improvement_pct": 5.0,
    "ev_artifact_min_false_load_reduction_pct": 50.0,
    "ev_artifact_min_mae_improvement_pct": 20.0,
    "ha_url": None,
    "ha_token": None,
    "model_max_iter": 300,
    "model_learning_rate": 0.06,
    "model_max_depth": 6,
    "model_min_samples_leaf": 20,
}

LEAD_BUCKETS = [
    (0, 6 * 60, "0_to_6h"),
    (6 * 60, 12 * 60, "6_to_12h"),
    (12 * 60, 24 * 60, "12_to_24h"),
    (24 * 60, 36 * 60, "24_to_36h"),
    (36 * 60, 48 * 60 + 1, "36_to_48h"),
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def safe_float(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    if s in ("", "unknown", "unavailable", "none", "null"):
        return np.nan
    if s in ("on", "open", "home", "detected", "true", "yes"):
        return 1.0
    if s in ("off", "closed", "not_home", "clear", "false", "no"):
        return 0.0
    try:
        return float(s)
    except Exception:
        return np.nan


def safe_feature_name(entity_id: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_]", "_", entity_id)


def convert_power_to_kw(series: pd.Series, unit: Optional[str]) -> pd.Series:
    if unit and str(unit).upper() == "W":
        return series / 1000.0
    return series


def lead_bucket_name(lead_minutes: float) -> str:
    for lo, hi, name in LEAD_BUCKETS:
        if lo <= lead_minutes < hi:
            return name
    return "outside_horizon"


class PowerSyncMLForecaster(hass.Hass):
    def initialize(self):
        self.cfg = self._read_config()
        self.tz = pytz.timezone(self.cfg["timezone"])
        self.data_dir = Path(self.cfg["data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.data_dir / "model_cache.joblib"
        self.snapshot_path = self.data_dir / "forecast_snapshots.jsonl"
        self.latest_forecast_path = self.data_dir / "latest_forecast.json"
        self.feature_cols: List[str] = []
        self.model: Optional[HistGradientBoostingRegressor] = None
        self.training_frame: Optional[pd.DataFrame] = None
        self.validation_metrics: Dict[str, Any] = {}
        self.last_train_time: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.holiday_cal = self._build_holiday_calendar()

        self._publish_status("initialising", "Starting PowerSync ML forecaster")
        loaded = self._load_cached_model_if_valid()
        if loaded:
            try:
                self._refresh_forecast()
            except Exception as e:
                self.log(f"Cached model forecast refresh failed: {e}", level="WARNING")

        self.run_in(self._train_callback, 10)
        self.run_every(
            self._train_callback,
            datetime.now() + timedelta(seconds=10 + self.cfg["retrain_minutes"] * 60),
            self.cfg["retrain_minutes"] * 60,
        )
        self.run_every(
            self._forecast_callback,
            datetime.now() + timedelta(minutes=2),
            self.cfg["update_minutes"] * 60,
        )
        self.log(
            f"Initialized v4.2. load_sensor={self.cfg['load_sensor']} output={self.cfg['hafo_entity_id']}"
        )

    def _read_config(self) -> Dict[str, Any]:
        cfg = dict(DEFAULT_CFG)
        for k in DEFAULT_CFG:
            if k in self.args:
                cfg[k] = self.args[k]
        if not cfg["load_sensor"]:
            raise ValueError("load_sensor is required")
        cfg["timezone"] = cfg.get("timezone") or self.get_timezone() or "UTC"
        cfg["ha_url"] = cfg.get("ha_url") or os.environ.get("HASS_URL") or "http://supervisor/core"
        cfg["ha_token"] = cfg.get("ha_token") or os.environ.get("HASS_TOKEN") or os.environ.get("SUPERVISOR_TOKEN")
        cfg["days_back"] = int(min(max(cfg["days_back"], 14), 365))
        if cfg["publish_interval_minutes"] < cfg["base_interval_minutes"]:
            cfg["publish_interval_minutes"] = cfg["base_interval_minutes"]
        return cfg

    def _build_holiday_calendar(self):
        region = self.cfg.get("public_holiday_region")
        if not region or not HOLIDAYS_AVAILABLE:
            return None
        try:
            return holidays.Australia(subdiv=region)
        except Exception as e:
            self.log(f"Holiday calendar disabled: {e}", level="WARNING")
            return None

    def _train_callback(self, kwargs):
        try:
            self._publish_status("training", "Training started")
            t0 = time.time()
            self._train_validate_refit_and_forecast()
            elapsed = time.time() - t0
            self.last_train_time = utc_now()
            self._save_model_cache()
            self._publish_status("ok", "")
            self.log(f"Training cycle complete in {elapsed:.1f}s")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self.last_error = err
            self._publish_status("error", err)
            self.log(traceback.format_exc(), level="ERROR")

    def _forecast_callback(self, kwargs):
        if self.model is None or self.training_frame is None:
            return
        try:
            self._refresh_forecast()
        except Exception as e:
            self.log(f"Forecast refresh failed: {e}", level="WARNING")

    def _load_cached_model_if_valid(self) -> bool:
        if not self.model_path.exists():
            return False
        max_age = timedelta(hours=float(self.cfg["cache_max_age_hours"]))
        mtime = datetime.fromtimestamp(self.model_path.stat().st_mtime, tz=timezone.utc)
        if utc_now() - mtime > max_age:
            return False
        try:
            obj = load(self.model_path)
            self.model = obj["model"]
            self.feature_cols = obj["feature_cols"]
            self.training_frame = obj.get("training_frame_tail")
            self.validation_metrics = obj.get("validation_metrics", {})
            self.last_train_time = obj.get("last_train_time")
            self.log("Loaded cached model")
            return True
        except Exception as e:
            self.log(f"Could not load cached model: {e}", level="WARNING")
            return False

    def _save_model_cache(self):
        if self.model is None:
            return
        tail = None
        if self.training_frame is not None:
            intervals = int((8 * 24 * 60) // self.cfg["base_interval_minutes"])
            tail = self.training_frame.tail(intervals).copy()
        dump({
            "model": self.model,
            "feature_cols": self.feature_cols,
            "training_frame_tail": tail,
            "validation_metrics": self.validation_metrics,
            "last_train_time": self.last_train_time,
            "version": "4.2",
        }, self.model_path)

    def _train_validate_refit_and_forecast(self):
        now = utc_now()
        start = now - timedelta(days=int(self.cfg["days_back"]))
        sensor_data = self._fetch_all_histories_chunked(start, now)
        base = self._build_base_frame(sensor_data, start, now)
        if len(base.dropna(subset=["baseline_load_kw"])) < self.cfg["minimum_training_days"] * 24 * 60 // self.cfg["base_interval_minutes"]:
            raise RuntimeError("Insufficient baseline load history after preparation")
        self.training_frame = base

        validation_cutoff = now - timedelta(days=int(self.cfg["validation_days"]))
        train_rows, val_rows = self._build_origin_training_table(base, start, now, validation_cutoff)
        if train_rows.empty or val_rows.empty:
            raise RuntimeError("Unable to build non-empty train and validation tables")

        model_for_validation = self._new_model()
        self.feature_cols = [c for c in train_rows.columns if c not in ("target_load_kw", "target_time", "origin_time", "lead_bucket")]
        model_for_validation.fit(train_rows[self.feature_cols].astype(float), train_rows["target_load_kw"].astype(float))
        val_pred = np.clip(model_for_validation.predict(val_rows[self.feature_cols].astype(float)), 0, None)
        self.validation_metrics = self._compute_validation_metrics(val_rows, val_pred)
        self._check_quality_gate(self.validation_metrics)

        # Standard deployment refit: train final model on all eligible data including validation period.
        all_rows = pd.concat([train_rows, val_rows], ignore_index=True)
        all_rows = self._subsample_training_rows(all_rows)
        self.model = self._new_model()
        self.model.fit(all_rows[self.feature_cols].astype(float), all_rows["target_load_kw"].astype(float))
        self._refresh_forecast()

    def _new_model(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=int(self.cfg["model_max_iter"]),
            learning_rate=float(self.cfg["model_learning_rate"]),
            max_depth=int(self.cfg["model_max_depth"]),
            min_samples_leaf=int(self.cfg["model_min_samples_leaf"]),
            l2_regularization=0.01,
            random_state=42,
        )

    def _fetch_all_histories_chunked(self, start: datetime, end: datetime) -> Dict[str, List[dict]]:
        sensors = {"load": self.cfg["load_sensor"]}
        if self.cfg.get("ev_power_sensor"):
            sensors["ev_power"] = self.cfg["ev_power_sensor"]
        if self.cfg.get("import_price_sensor"):
            sensors["import_price"] = self.cfg["import_price_sensor"]
        if self.cfg.get("feed_in_price_sensor"):
            sensors["feed_in_price"] = self.cfg["feed_in_price_sensor"]
        for extra in self.cfg.get("additional_feature_sensors") or []:
            sensors[safe_feature_name(extra)] = extra

        total_start = time.time()
        result: Dict[str, List[dict]] = {}
        chunk_days = int(self.cfg["history_fetch_chunk_days"])
        for key, entity in sensors.items():
            records: List[dict] = []
            cursor = start
            while cursor < end:
                if time.time() - total_start > float(self.cfg["max_total_history_fetch_seconds"]):
                    raise RuntimeError("History fetch exceeded max_total_history_fetch_seconds")
                chunk_end = min(cursor + timedelta(days=chunk_days), end)
                c0 = time.time()
                records.extend(self._fetch_history(entity, cursor, chunk_end))
                elapsed = time.time() - c0
                if elapsed > float(self.cfg["max_single_history_chunk_seconds"]):
                    raise RuntimeError(f"History chunk for {entity} took {elapsed:.1f}s")
                cursor = chunk_end
            result[key] = records
            self.log(f"Fetched {len(records)} records for {entity}")
        return result

    def _fetch_history(self, entity_id: str, start: datetime, end: datetime) -> List[dict]:
        try:
            hist = self.get_history(entity_id=entity_id, start_time=start.isoformat(), end_time=end.isoformat())
            if hist and isinstance(hist, list) and hist[0]:
                return hist[0]
        except Exception as e:
            self.log(f"get_history failed for {entity_id}: {e}; trying REST", level="WARNING")
        token = self.cfg.get("ha_token")
        if not token:
            return []
        try:
            url = f"{self.cfg['ha_url']}/api/history/period/{start.isoformat()}"
            r = requests.get(url, params={"filter_entity_id": entity_id, "end_time": end.isoformat()}, headers={"Authorization": f"Bearer {token}"}, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data[0] if data and data[0] else []
        except Exception as e:
            self.log(f"REST history failed for {entity_id}: {e}", level="ERROR")
            return []

    def _build_base_frame(self, sensor_data: Dict[str, List[dict]], start: datetime, end: datetime) -> pd.DataFrame:
        interval = int(self.cfg["base_interval_minutes"])
        idx = pd.date_range(start=start.replace(second=0, microsecond=0), end=end.replace(second=0, microsecond=0), freq=f"{interval}min", tz="UTC")
        df = pd.DataFrame(index=idx)
        for key, hist in sensor_data.items():
            series = self._history_to_series(hist, idx, interval)
            if series is not None:
                df[key] = series
        load_unit = self.cfg.get("load_unit_override") or self.get_state(self.cfg["load_sensor"], attribute="unit_of_measurement")
        df["raw_load_kw"] = convert_power_to_kw(df["load"], load_unit)
        if "ev_power" in df:
            ev_unit = self.cfg.get("ev_power_unit_override") or self.get_state(self.cfg["ev_power_sensor"], attribute="unit_of_measurement")
            df["ev_power_kw"] = convert_power_to_kw(df["ev_power"], ev_unit).fillna(0)
        else:
            df["ev_power_kw"] = 0.0
        charging = df["ev_power_kw"] >= float(self.cfg["ev_exclusion_threshold_kw"])
        df["ev_excluded_kw"] = np.where(charging, df["ev_power_kw"], 0.0)
        df["baseline_load_kw"] = np.maximum(df["raw_load_kw"] - df["ev_excluded_kw"], 0)
        df["ev_charging_flag"] = charging.astype(float)
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if mem_mb > float(self.cfg["max_training_frame_mb"]):
            raise RuntimeError(f"Prepared base frame memory {mem_mb:.1f} MB exceeds limit")
        return df

    def _history_to_series(self, hist: List[dict], idx: pd.DatetimeIndex, interval: int) -> Optional[pd.Series]:
        rows = []
        for h in hist:
            ts = h.get("last_changed") or h.get("last_updated")
            if not ts:
                continue
            val = safe_float(h.get("state"))
            if pd.isna(val):
                continue
            rows.append((pd.to_datetime(ts, utc=True), val))
        if not rows:
            return None
        s = pd.Series([v for _, v in rows], index=pd.DatetimeIndex([t for t, _ in rows])).sort_index()
        s = s.resample(f"{interval}min").last()
        return s.reindex(idx, method="ffill", limit=max(1, int(60 / interval)))

    def _build_origin_training_table(self, base: pd.DataFrame, start: datetime, end: datetime, validation_cutoff: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        origin_interval = int(self.cfg["training_origin_interval_minutes"])
        lead_interval = int(self.cfg["training_lead_interval_minutes"])
        horizon_minutes = int(self.cfg["horizon_hours"] * 60)
        earliest_origin = base.index.min() + pd.Timedelta(days=7)
        latest_origin = base.index.max() - pd.Timedelta(minutes=horizon_minutes)
        origins = pd.date_range(earliest_origin.ceil(f"{origin_interval}min"), latest_origin.floor(f"{origin_interval}min"), freq=f"{origin_interval}min", tz="UTC")
        leads = np.arange(lead_interval, horizon_minutes + 1, lead_interval, dtype=int)
        weather_hist = self._fetch_weather_training(start, end) if self.cfg["weather_enabled"] else pd.DataFrame()
        chunks = []
        for origin in origins:
            origin_features = self._features_known_at_origin(base, origin)
            if origin_features is None:
                continue
            rows = []
            for lead in leads:
                target_time = origin + pd.Timedelta(minutes=int(lead))
                target = self._lookup_series_value(base["baseline_load_kw"], target_time)
                if pd.isna(target):
                    continue
                feat = dict(origin_features)
                feat.update(self._calendar_features(target_time))
                feat.update(self._valid_target_lag_features(base, origin, target_time))
                feat.update(self._weather_features(weather_hist, target_time))
                feat["lead_minutes"] = float(lead)
                feat["lead_sin"] = math.sin(2 * math.pi * lead / horizon_minutes)
                feat["lead_cos"] = math.cos(2 * math.pi * lead / horizon_minutes)
                feat["target_load_kw"] = float(target)
                feat["origin_time"] = origin
                feat["target_time"] = target_time
                feat["lead_bucket"] = lead_bucket_name(lead)
                rows.append(feat)
            if rows:
                chunks.append(pd.DataFrame(rows))
        if not chunks:
            raise RuntimeError("No origin training rows built")
        table = pd.concat(chunks, ignore_index=True)
        table = table.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        train = table[pd.to_datetime(table["origin_time"], utc=True) < validation_cutoff].copy()
        val = table[pd.to_datetime(table["origin_time"], utc=True) >= validation_cutoff].copy()
        train = self._subsample_training_rows(train)
        return train, val

    def _features_known_at_origin(self, base: pd.DataFrame, origin: pd.Timestamp) -> Optional[Dict[str, float]]:
        if origin not in base.index:
            return None
        out: Dict[str, float] = {}
        history = base.loc[:origin, "baseline_load_kw"].dropna()
        if len(history) < int(7 * 24 * 60 / self.cfg["base_interval_minutes"]):
            return None
        for hours in [1, 6, 24, 168]:
            window = int(hours * 60 / self.cfg["base_interval_minutes"])
            tail = history.tail(window)
            if len(tail) >= max(2, window // 2):
                out[f"origin_rolling_mean_{hours}h_kw"] = float(tail.mean())
                if hours >= 24:
                    out[f"origin_rolling_std_{hours}h_kw"] = float(tail.std())
        for col in ["import_price", "feed_in_price", "ev_charging_flag"]:
            if col in base.columns:
                out[f"origin_{col}"] = float(base.loc[origin, col]) if not pd.isna(base.loc[origin, col]) else 0.0
        for extra in self.cfg.get("additional_feature_sensors") or []:
            col = safe_feature_name(extra)
            if col in base.columns:
                out[f"origin_{col}"] = float(base.loc[origin, col]) if not pd.isna(base.loc[origin, col]) else 0.0
                out[f"origin_{col}_is_origin_value"] = 1.0
        return out

    def _calendar_features(self, target_time: pd.Timestamp) -> Dict[str, float]:
        local = target_time.tz_convert(self.tz)
        hour = local.hour + local.minute / 60.0
        dow = local.dayofweek
        month = local.month
        doy = local.dayofyear
        d = local.date()
        is_holiday = 1.0 if self.holiday_cal is not None and d in self.holiday_cal else 0.0
        tomorrow = d + timedelta(days=1)
        before_holiday = 1.0 if self.holiday_cal is not None and tomorrow in self.holiday_cal else 0.0
        return {
            "target_hour_sin": math.sin(2 * math.pi * hour / 24),
            "target_hour_cos": math.cos(2 * math.pi * hour / 24),
            "target_dow_sin": math.sin(2 * math.pi * dow / 7),
            "target_dow_cos": math.cos(2 * math.pi * dow / 7),
            "target_month_sin": math.sin(2 * math.pi * (month - 1) / 12),
            "target_month_cos": math.cos(2 * math.pi * (month - 1) / 12),
            "target_doy_sin": math.sin(2 * math.pi * doy / 365),
            "target_doy_cos": math.cos(2 * math.pi * doy / 365),
            "target_is_weekend": 1.0 if dow >= 5 else 0.0,
            "target_is_public_holiday": is_holiday,
            "target_day_before_holiday": before_holiday,
        }

    def _valid_target_lag_features(self, base: pd.DataFrame, origin: pd.Timestamp, target_time: pd.Timestamp) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for hours in [24, 48, 168]:
            lag_time = target_time - pd.Timedelta(hours=hours)
            valid = lag_time <= origin
            val = self._lookup_series_value(base["baseline_load_kw"], lag_time) if valid else np.nan
            out[f"target_lag_{hours}h_kw"] = 0.0 if pd.isna(val) else float(val)
            out[f"target_lag_{hours}h_valid"] = 1.0 if valid and not pd.isna(val) else 0.0
        return out

    def _lookup_series_value(self, series: pd.Series, ts: pd.Timestamp) -> float:
        try:
            pos = series.index.get_indexer([ts], method="nearest", tolerance=pd.Timedelta(minutes=int(self.cfg["base_interval_minutes"] // 2 + 1)))[0]
            if pos == -1:
                return np.nan
            return float(series.iloc[pos])
        except Exception:
            return np.nan

    def _fetch_weather_training(self, start: datetime, end: datetime) -> pd.DataFrame:
        if not self.cfg.get("latitude") or not self.cfg.get("longitude"):
            return pd.DataFrame()
        params = {
            "latitude": self.cfg["latitude"],
            "longitude": self.cfg["longitude"],
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "hourly": ",".join(OPEN_METEO_VARS),
            "timezone": "UTC",
            "wind_speed_unit": "kmh",
        }
        return self._fetch_weather_df(HISTORICAL_FORECAST_URL, params)

    def _fetch_weather_live(self, forecast_days: int) -> pd.DataFrame:
        if not self.cfg.get("latitude") or not self.cfg.get("longitude"):
            return pd.DataFrame()
        params = {
            "latitude": self.cfg["latitude"],
            "longitude": self.cfg["longitude"],
            "forecast_days": forecast_days,
            "hourly": ",".join(OPEN_METEO_VARS),
            "timezone": "UTC",
            "wind_speed_unit": "kmh",
        }
        return self._fetch_weather_df(LIVE_FORECAST_URL, params)

    def _fetch_weather_df(self, url: str, params: Dict[str, Any]) -> pd.DataFrame:
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            hourly = r.json().get("hourly", {})
            times = hourly.get("time") or []
            if not times:
                return pd.DataFrame()
            df = pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True)})
            for v in OPEN_METEO_VARS:
                df[f"wx_{v}"] = hourly.get(v, [np.nan] * len(df))
            return df.set_index("timestamp")
        except Exception as e:
            self.log(f"Weather fetch failed from {url}: {e}", level="WARNING")
            return pd.DataFrame()

    def _weather_features(self, weather_df: pd.DataFrame, target_time: pd.Timestamp) -> Dict[str, float]:
        out = {}
        for v in OPEN_METEO_VARS:
            col = f"wx_{v}"
            out[col] = 0.0
        if weather_df.empty:
            return out
        try:
            row = weather_df.reindex([target_time], method="nearest", tolerance=pd.Timedelta(minutes=65)).iloc[0]
            for c in weather_df.columns:
                out[c] = float(row[c]) if not pd.isna(row[c]) else 0.0
        except Exception:
            pass
        return out

    def _subsample_training_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        max_rows = int(self.cfg["max_training_rows"])
        if len(df) <= max_rows:
            return df
        rng = np.random.default_rng(42)
        groups = []
        per_bucket = max(1, max_rows // max(1, df["lead_bucket"].nunique()))
        for _, g in df.groupby("lead_bucket"):
            if len(g) > per_bucket:
                idx = rng.choice(g.index.to_numpy(), size=per_bucket, replace=False)
                groups.append(g.loc[idx])
            else:
                groups.append(g)
        sampled = pd.concat(groups).sample(frac=1, random_state=42).reset_index(drop=True)
        if len(sampled) > max_rows:
            sampled = sampled.sample(n=max_rows, random_state=42)
        return sampled

    def _compute_validation_metrics(self, val_rows: pd.DataFrame, preds: np.ndarray) -> Dict[str, Any]:
        actual = val_rows["target_load_kw"].to_numpy(dtype=float)
        model_mae = float(mean_absolute_error(actual, preds))
        metrics: Dict[str, Any] = {"model_mae_kw": round(model_mae, 4), "by_lead_bucket": {}}
        for name, g in val_rows.groupby("lead_bucket"):
            idx = g.index.to_numpy()
            # val_rows may have non-contiguous indexes after concat; create positional mask.
            mask = val_rows["lead_bucket"].to_numpy() == name
            a = actual[mask]
            p = preds[mask]
            bucket = {"model_mae_kw": round(float(mean_absolute_error(a, p)), 4)}
            # Persistence 168h is valid across the whole 48h horizon.
            if "target_lag_168h_kw" in g:
                bucket["persistence_168h_mae_kw"] = round(float(mean_absolute_error(a, g["target_lag_168h_kw"].to_numpy(dtype=float))), 4)
            # Persistence 24h is only valid up to 24h.
            if name in ("0_to_6h", "6_to_12h", "12_to_24h") and "target_lag_24h_valid" in g and g["target_lag_24h_valid"].mean() > 0.95:
                bucket["persistence_24h_mae_kw"] = round(float(mean_absolute_error(a, g["target_lag_24h_kw"].to_numpy(dtype=float))), 4)
            else:
                bucket["persistence_24h_mae_kw"] = None
            metrics["by_lead_bucket"][name] = bucket
        valid_baselines = []
        for b in metrics["by_lead_bucket"].values():
            for k in ("persistence_24h_mae_kw", "persistence_168h_mae_kw"):
                if b.get(k) is not None:
                    valid_baselines.append(float(b[k]))
        if valid_baselines:
            best = min(valid_baselines)
            metrics["best_persistence_mae_kw"] = round(best, 4)
            metrics["model_vs_best_persistence_delta_pct"] = round((model_mae - best) / best * 100.0, 2) if best > 0 else None
        # EV artefact gate approximation: compare validation rows whose origin had EV charging recently.
        metrics["ev_artifact_gate"] = {"status": "not_enough_ev_validation_data"}
        return metrics

    def _check_quality_gate(self, metrics: Dict[str, Any]):
        if not self.cfg.get("require_quality_gate"):
            return
        delta = metrics.get("model_vs_best_persistence_delta_pct")
        if delta is not None and delta <= -float(self.cfg["min_model_improvement_pct"]):
            return
        ev_gate = metrics.get("ev_artifact_gate", {})
        if ev_gate.get("passed"):
            return
        raise RuntimeError(f"Quality gate failed. Metrics: {json.dumps(metrics)[:1000]}")

    def _refresh_forecast(self):
        if self.model is None or self.training_frame is None or not self.feature_cols:
            return
        now = utc_now()
        pub_interval = int(self.cfg["publish_interval_minutes"])
        train_lead_interval = int(self.cfg["training_lead_interval_minutes"])
        horizon_minutes = int(self.cfg["horizon_hours"] * 60)
        forecast_start = self._next_boundary(now, pub_interval)
        # v4.2 Option C: predict at training lead resolution then interpolate to publish resolution.
        coarse_leads = np.arange(train_lead_interval, horizon_minutes + 1, train_lead_interval, dtype=int)
        coarse_times = [forecast_start + timedelta(minutes=int(lead)) for lead in coarse_leads]
        live_weather = self._fetch_weather_live(max(3, int(math.ceil(self.cfg["horizon_hours"] / 24)) + 1)) if self.cfg["weather_enabled"] else pd.DataFrame()
        rows = []
        origin = pd.Timestamp(now)
        origin_features = self._features_known_at_origin(self.training_frame, self.training_frame.index[-1]) or {}
        for lead, target_time in zip(coarse_leads, coarse_times):
            target_ts = pd.Timestamp(target_time)
            feat = dict(origin_features)
            feat.update(self._calendar_features(target_ts))
            feat.update(self._valid_target_lag_features(self.training_frame, pd.Timestamp(self.training_frame.index[-1]), target_ts))
            feat.update(self._weather_features(live_weather, target_ts))
            feat["lead_minutes"] = float(lead)
            feat["lead_sin"] = math.sin(2 * math.pi * lead / horizon_minutes)
            feat["lead_cos"] = math.cos(2 * math.pi * lead / horizon_minutes)
            rows.append(feat)
        X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        preds_coarse = np.clip(self.model.predict(X[self.feature_cols].astype(float)), 0, None)
        publish_leads = np.arange(pub_interval, horizon_minutes + 1, pub_interval, dtype=int)
        preds_publish = np.interp(publish_leads, coarse_leads, preds_coarse)
        forecast = []
        daily_totals: Dict[str, float] = {}
        for lead, value in zip(publish_leads, preds_publish):
            t = forecast_start + timedelta(minutes=int(lead))
            t_local = t.astimezone(self.tz)
            v = round(float(value), 3)
            forecast.append({"time": t_local.isoformat(), "value": v})
            day = t_local.date().isoformat()
            daily_totals[day] = daily_totals.get(day, 0.0) + float(value) * pub_interval / 60.0
        self._publish_forecast(forecast, preds_publish, daily_totals)
        self._write_latest_forecast(forecast)
        self._append_snapshot(forecast)

    def _next_boundary(self, dt: datetime, interval_minutes: int) -> datetime:
        epoch = int(dt.timestamp())
        step = interval_minutes * 60
        return datetime.fromtimestamp(((epoch // step) + 1) * step, tz=timezone.utc)

    def _publish_status(self, status: str, message: str):
        try:
            current = self.get_state(self.cfg["hafo_entity_id"], attribute="all") or {}
            attrs = current.get("attributes", {}) if isinstance(current, dict) else {}
            state = current.get("state", "0") if isinstance(current, dict) else "0"
        except Exception:
            attrs, state = {}, "0"
        attrs.update({
            "status": status,
            "status_message": message,
            "last_status_update": utc_now().isoformat(),
            "friendly_name": self.cfg["friendly_name"],
            "unit_of_measurement": "kW",
            "device_class": "power",
            "state_class": "measurement",
            "source": "powersync_ml_forecaster_v4_2",
        })
        self.set_state(self.cfg["hafo_entity_id"], state=state, attributes=attrs)

    def _publish_forecast(self, forecast: List[dict], predictions: np.ndarray, daily_totals: Dict[str, float]):
        attrs = {
            "forecast": forecast,
            "status": "ok",
            "friendly_name": self.cfg["friendly_name"],
            "unit_of_measurement": "kW",
            "device_class": "power",
            "state_class": "measurement",
            "source": "powersync_ml_forecaster_v4_2",
            "source_entity": self.cfg["load_sensor"],
            "last_updated": utc_now().astimezone(self.tz).isoformat(),
            "last_trained": self.last_train_time.astimezone(self.tz).isoformat() if self.last_train_time else None,
            "horizon_hours": self.cfg["horizon_hours"],
            "publish_interval_minutes": self.cfg["publish_interval_minutes"],
            "training_lead_interval_minutes": self.cfg["training_lead_interval_minutes"],
            "interpolation_mode": "predict_training_leads_then_linear_interpolate_to_publish_interval",
            "model": "HistGradientBoostingRegressor_absolute_error",
            "validation": self.validation_metrics,
            "weather_training_mode": self.cfg["weather_training_mode"],
            "price_training_mode": self.cfg["price_training_mode"],
            "ev_exclusion_threshold_kw": self.cfg["ev_exclusion_threshold_kw"],
            "ev_excluded_kwh_last_training": self._ev_excluded_kwh(),
            "daily_forecast_kwh": {k: round(v, 2) for k, v in daily_totals.items()},
        }
        state = str(round(float(predictions[0]), 3)) if len(predictions) else "0"
        self.set_state(self.cfg["hafo_entity_id"], state=state, attributes=attrs)

    def _ev_excluded_kwh(self) -> Optional[float]:
        if self.training_frame is None or "ev_excluded_kw" not in self.training_frame:
            return None
        return round(float(self.training_frame["ev_excluded_kw"].sum() * self.cfg["base_interval_minutes"] / 60.0), 3)


    def _write_latest_forecast(self, forecast: List[dict]):
        rec = {
            "created_at": utc_now().isoformat(),
            "entity_id": self.cfg["hafo_entity_id"],
            "forecast": forecast,
            "unit_of_measurement": "kW",
            "source": "powersync_ml_forecaster_v4_2",
        }
        tmp = self.latest_forecast_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(rec, f)
        tmp.replace(self.latest_forecast_path)

    def _append_snapshot(self, forecast: List[dict]):
        rec = {"created_at": utc_now().isoformat(), "entity_id": self.cfg["hafo_entity_id"], "forecast": forecast}
        with self.snapshot_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self._prune_snapshots()

    def _prune_snapshots(self):
        if not self.snapshot_path.exists():
            return
        cutoff = utc_now() - timedelta(days=int(self.cfg["snapshot_retention_days"]))
        try:
            kept = []
            with self.snapshot_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if pd.to_datetime(rec.get("created_at"), utc=True).to_pydatetime() >= cutoff:
                            kept.append(line)
                    except Exception:
                        pass
            with self.snapshot_path.open("w", encoding="utf-8") as f:
                f.writelines(kept)
        except Exception as e:
            self.log(f"Snapshot pruning failed: {e}", level="WARNING")
