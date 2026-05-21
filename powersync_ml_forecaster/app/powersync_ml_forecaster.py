"""
PowerSync ML Load Forecaster v4.5.0

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

FORECASTER_VERSION = "4.5.0"

DEFAULT_CFG: Dict[str, Any] = {
    "hafo_entity_id": "sensor.hafo_power_sync_home_load_forecast",
    "friendly_name": "HAFO Load Forecast",
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
    "training_origin_interval_minutes": 120,
    "training_lead_interval_minutes": 30,
    "publish_interval_minutes": 5,
    "retrain_minutes": 360,
    "update_minutes": 15,
    "validation_days": 14,
    "max_training_rows": 75000,
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
    "model_max_iter": 120,
    "model_learning_rate": 0.06,
    "model_max_depth": 6,
    "model_min_samples_leaf": 20,
    "load_anomaly_filter_enabled": True,
    "load_anomaly_min_residual_kw": 1.5,
    "load_anomaly_mad_multiplier": 6.0,
    "load_anomaly_min_duration_minutes": 30,
    "load_anomaly_buffer_before_minutes": 15,
    "load_anomaly_buffer_after_minutes": 60,
    "load_anomaly_max_daily_fraction": 0.35,
    "load_anomaly_replacement": "time_of_week_median",
    "clean_contaminated_lags": True,
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


def aligned_interval_bounds(start: datetime, end: datetime, interval_minutes: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    freq = f"{int(interval_minutes)}min"
    aligned_start = pd.to_datetime(start, utc=True).floor(freq)
    aligned_end = pd.to_datetime(end, utc=True).ceil(freq)
    if aligned_end < aligned_start:
        aligned_end = aligned_start
    return aligned_start, aligned_end


class PowerSyncMLForecaster(hass.Hass):
    def initialize(self):
        self.cfg = self._read_config()
        self.tz = pytz.timezone(self.cfg["timezone"])
        self.data_dir = Path(self.cfg["data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.data_dir / "model_cache.joblib"
        self.base_frame_cache_path = self.data_dir / "base_frame_cache.parquet"
        self.weather_cache_path = self.data_dir / "weather_cache.parquet"
        self.snapshot_path = self.data_dir / "forecast_snapshots.jsonl"
        self.latest_forecast_path = self.data_dir / "latest_forecast.json"
        self.feature_cols: List[str] = []
        self.model: Optional[HistGradientBoostingRegressor] = None
        self.training_frame: Optional[pd.DataFrame] = None
        self.validation_metrics: Dict[str, Any] = {}
        self.history_diagnostics: Dict[str, Any] = {}
        self.load_anomaly_diagnostics: Dict[str, Any] = {}
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
            f"Initialized v{FORECASTER_VERSION}. load_sensor={self.cfg['load_sensor']} output={self.cfg['hafo_entity_id']}"
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
        if cfg.get("friendly_name") == "HAFO PowerSync ML Load Forecast":
            cfg["friendly_name"] = DEFAULT_CFG["friendly_name"]
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
            try:
                self._publish_status("training", "Training started")
            except Exception:
                self.log("Failed to publish training status:\n" + traceback.format_exc(), level="ERROR")
            t0 = time.time()
            self._train_validate_refit_and_forecast()
            elapsed = time.time() - t0
            self.last_train_time = utc_now()
            self._save_model_cache()
            self.log(f"Training cycle complete in {elapsed:.1f}s")
            self._refresh_forecast()
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self.last_error = err
            try:
                self._publish_status("error", err)
            except Exception:
                self.log("Failed to publish error status:\n" + traceback.format_exc(), level="ERROR")
            self.log(traceback.format_exc(), level="ERROR")

    def _forecast_callback(self, kwargs):
        if self.model is None or self.training_frame is None:
            return
        try:
            self._refresh_forecast()
        except Exception:
            self.last_error = traceback.format_exc()
            self.log("Forecast refresh failed:\n" + self.last_error, level="ERROR")

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
            "version": FORECASTER_VERSION,
        }, self.model_path)

    def _train_validate_refit_and_forecast(self):
        stage_t0 = time.time()
        now = utc_now()
        start = now - timedelta(days=int(self.cfg["days_back"]))
        base = self._load_or_build_base_frame(start, now)
        self._log_stage_elapsed("history fetch + base frame construction", stage_t0)
        stage_t0 = time.time()
        required_points = self.cfg["minimum_training_days"] * 24 * 60 // self.cfg["base_interval_minutes"]
        if len(base.dropna(subset=["baseline_load_kw"])) < required_points:
            raise RuntimeError(self._insufficient_history_message(base))
        self.training_frame = base
        self.log(
            "Prepared training base frame: "
            f"rows={len(base)} usable_baseline_rows={int(base['baseline_load_kw'].notna().sum())} "
            f"usable_baseline_days={self.history_diagnostics.get('usable_baseline_days')}"
        )

        validation_cutoff = now - timedelta(days=int(self.cfg["validation_days"]))
        stage_t0 = time.time()
        train_rows, val_rows = self._build_origin_training_table(base, start, now, validation_cutoff)
        self._log_stage_elapsed("origin training table construction", stage_t0)
        if train_rows.empty or val_rows.empty:
            raise RuntimeError("Unable to build non-empty train and validation tables")
        self.log(f"Built origin training rows: train={len(train_rows)} validation={len(val_rows)}")

        model_for_validation = self._new_model()
        self.feature_cols = [c for c in train_rows.columns if c not in ("target_load_kw", "target_time", "origin_time", "lead_bucket")]
        self.log(f"Fitting validation model: rows={len(train_rows)} features={len(self.feature_cols)}")
        stage_t0 = time.time()
        model_for_validation.fit(train_rows[self.feature_cols].astype(float), train_rows["target_load_kw"].astype(float))
        val_pred = np.clip(model_for_validation.predict(val_rows[self.feature_cols].astype(float)), 0, None)
        self._log_stage_elapsed("validation fit", stage_t0)
        self.validation_metrics = self._compute_validation_metrics(val_rows, val_pred)
        self._check_quality_gate(self.validation_metrics)

        # Standard deployment refit: train final model on all eligible data including validation period.
        all_rows = pd.concat([train_rows, val_rows], ignore_index=True)
        all_rows = self._subsample_training_rows(all_rows)
        self.log(f"Fitting final model: rows={len(all_rows)} features={len(self.feature_cols)}")
        stage_t0 = time.time()
        self.model = self._new_model()
        self.model.fit(all_rows[self.feature_cols].astype(float), all_rows["target_load_kw"].astype(float))
        self._log_stage_elapsed("final fit", stage_t0)
        self.log("Final model fit complete")

    def _log_stage_elapsed(self, stage_name: str, t0: float):
        self.log(f"Training stage '{stage_name}' elapsed_seconds={time.time() - t0:.3f}")

    def _load_or_build_base_frame(self, start: datetime, end: datetime) -> pd.DataFrame:
        t0 = time.time()
        cached = self._read_base_frame_cache()
        sensor_data: Dict[str, List[dict]]
        if cached is not None and not cached.empty:
            latest_cached = cached.index.max().to_pydatetime()
            fetch_start = max(start, latest_cached - timedelta(minutes=int(self.cfg["base_interval_minutes"]) * 2))
            sensor_data = self._fetch_all_histories_chunked(fetch_start, end)
            self._log_stage_elapsed("history fetch", t0)
            t1 = time.time()
            new_base = self._build_base_frame(sensor_data, fetch_start, end)
            merged = pd.concat([cached[cached.index < new_base.index.min()], new_base]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            merged = merged[merged.index >= pd.to_datetime(start, utc=True)]
            self._write_base_frame_cache(merged)
            self._log_stage_elapsed("base frame construction", t1)
            return merged
        sensor_data = self._fetch_all_histories_chunked(start, end)
        self._log_stage_elapsed("history fetch", t0)
        t1 = time.time()
        base = self._build_base_frame(sensor_data, start, end)
        self._write_base_frame_cache(base)
        self._log_stage_elapsed("base frame construction", t1)
        return base

    def _read_base_frame_cache(self) -> Optional[pd.DataFrame]:
        if not self.base_frame_cache_path.exists():
            return None
        try:
            cached = pd.read_parquet(self.base_frame_cache_path)
            cached.index = pd.to_datetime(cached.index, utc=True)
            return cached.sort_index()
        except Exception as e:
            self.log(f"Base frame cache read failed: {e}", level="WARNING")
            return None

    def _write_base_frame_cache(self, df: pd.DataFrame):
        try:
            df.to_parquet(self.base_frame_cache_path)
        except Exception as e:
            self.log(f"Base frame cache write failed: {e}", level="WARNING")

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
        load_entity = self.cfg["load_sensor"]
        result["load_short_term_statistics"] = self._fetch_load_statistics(load_entity, start, end, "5minute")
        result["load_long_term_statistics"] = self._fetch_load_statistics(load_entity, start, end, "hour")
        return result

    def _fetch_load_statistics(self, entity_id: str, start: datetime, end: datetime, period: str) -> List[dict]:
        try:
            stats = self.get_statistics(entity_id, start.isoformat(), end.isoformat(), period=period, statistic_types=["mean"])
            records = self._extract_statistics_records(stats, entity_id)
            self.log(f"Fetched {len(records)} {period} statistics records for {entity_id}")
            return records
        except AttributeError:
            self.log("Recorder statistics unavailable in this runtime; detailed history only", level="WARNING")
        except Exception as e:
            self.log(f"Recorder statistics fetch failed for {entity_id} period={period}: {e}", level="WARNING")
        return []

    def _extract_statistics_records(self, stats: Any, entity_id: str) -> List[dict]:
        if not stats:
            return []
        if isinstance(stats, list):
            return [r for r in stats if isinstance(r, dict)]
        if isinstance(stats, dict):
            if entity_id in stats and isinstance(stats[entity_id], list):
                return [r for r in stats[entity_id] if isinstance(r, dict)]
            result = stats.get("result")
            if isinstance(result, dict) and entity_id in result and isinstance(result[entity_id], list):
                return [r for r in result[entity_id] if isinstance(r, dict)]
        return []

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
        idx_start, idx_end = aligned_interval_bounds(start, end, interval)
        idx = pd.date_range(start=idx_start, end=idx_end, freq=f"{interval}min")
        df = pd.DataFrame(index=idx)
        load_unit = self.cfg.get("load_unit_override") or self.get_state(self.cfg["load_sensor"], attribute="unit_of_measurement")
        load_detailed = self._history_to_series(sensor_data.get("load", []), idx, interval)
        load_short = self._statistics_to_series(sensor_data.get("load_short_term_statistics", []), idx, interval, "short_term_statistics")
        load_long = self._statistics_to_series(sensor_data.get("load_long_term_statistics", []), idx, interval, "long_term_statistics")
        if load_detailed is not None:
            df["load"] = load_detailed
        else:
            df["load"] = np.nan
        df["raw_load_kw"] = convert_power_to_kw(df["load"], load_unit)
        df["history_source"] = pd.Series(index=idx, dtype="object")
        df.loc[df["raw_load_kw"].notna(), "history_source"] = "detailed_state"
        for source_name, series in [("short_term_statistics", load_short), ("long_term_statistics", load_long)]:
            if series is None:
                continue
            stats_kw = convert_power_to_kw(series, load_unit)
            fill_mask = df["raw_load_kw"].isna() & stats_kw.notna()
            df.loc[fill_mask, "raw_load_kw"] = stats_kw.loc[fill_mask]
            df.loc[fill_mask, "history_source"] = source_name
        for key, hist in sensor_data.items():
            if key in ("load", "load_short_term_statistics", "load_long_term_statistics"):
                continue
            series = self._history_to_series(hist, idx, interval)
            if series is not None:
                df[key] = series
        if "ev_power" in df:
            ev_unit = self.cfg.get("ev_power_unit_override") or self.get_state(self.cfg["ev_power_sensor"], attribute="unit_of_measurement")
            df["ev_power_kw"] = convert_power_to_kw(df["ev_power"], ev_unit).fillna(0)
        else:
            df["ev_power_kw"] = 0.0
        detailed_load = df["history_source"] == "detailed_state"
        charging = detailed_load & (df["ev_power_kw"] >= float(self.cfg["ev_exclusion_threshold_kw"]))
        df["ev_excluded_kw"] = np.where(charging, df["ev_power_kw"], 0.0)
        df["baseline_load_kw"] = np.where(detailed_load, np.maximum(df["raw_load_kw"] - df["ev_excluded_kw"], 0), df["raw_load_kw"])
        df["ev_charging_flag"] = charging.astype(float)
        t0 = time.time()
        df = self._apply_load_anomaly_filter(df)
        self._log_stage_elapsed("load anomaly filtering", t0)
        self.history_diagnostics = self._history_diagnostics(df, start, end)
        self.log(f"History diagnostics: {json.dumps(self.history_diagnostics, sort_keys=True)}")
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if mem_mb > float(self.cfg["max_training_frame_mb"]):
            raise RuntimeError(f"Prepared base frame memory {mem_mb:.1f} MB exceeds limit")
        return df

    def _apply_load_anomaly_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        self.load_anomaly_diagnostics = {
            "load_anomaly_filter_enabled": bool(self.cfg.get("load_anomaly_filter_enabled", True)),
            "load_anomaly_events_last_training": 0,
            "load_anomaly_intervals_masked": 0,
            "load_anomaly_kwh_masked": 0.0,
            "load_anomaly_replacement": self.cfg.get("load_anomaly_replacement", "time_of_week_median"),
            "contaminated_lag_values_replaced": 0,
        }
        df["load_contaminated"] = False
        df["cleaned_baseline_load_kw"] = df["baseline_load_kw"]
        if not self.cfg.get("load_anomaly_filter_enabled", True):
            return df
        local = df.index.tz_convert(self.tz)
        slots_per_day = max(1, int(round(1440 / int(self.cfg["base_interval_minutes"]))))
        slot = local.hour * 60 + local.minute
        keys = pd.MultiIndex.from_arrays([local.dayofweek, slot])
        med_map = df.groupby(keys)["baseline_load_kw"].median()
        expected = keys.map(med_map).astype(float)
        residual = (df["baseline_load_kw"] - expected).fillna(0.0)
        mad_map = residual.groupby(keys).apply(lambda s: float(np.nanmedian(np.abs(s - np.nanmedian(s)))) if len(s) else 0.0)
        mad = keys.map(mad_map).astype(float)
        min_residual = float(self.cfg["load_anomaly_min_residual_kw"])
        mad_mult = float(self.cfg["load_anomaly_mad_multiplier"])
        threshold = np.maximum(min_residual, mad_mult * mad)
        candidate = residual >= threshold
        min_points = max(1, int(math.ceil(float(self.cfg["load_anomaly_min_duration_minutes"]) / float(self.cfg["base_interval_minutes"]))))
        grp = (candidate != candidate.shift(fill_value=False)).cumsum()
        run_len = candidate.groupby(grp).transform("sum")
        sustained = candidate & (run_len >= min_points)
        before = max(0, int(math.ceil(float(self.cfg["load_anomaly_buffer_before_minutes"]) / float(self.cfg["base_interval_minutes"]))))
        after = max(0, int(math.ceil(float(self.cfg["load_anomaly_buffer_after_minutes"]) / float(self.cfg["base_interval_minutes"]))))
        mask = sustained.copy()
        for i in range(1, before + 1):
            mask |= sustained.shift(-i, fill_value=False)
        for i in range(1, after + 1):
            mask |= sustained.shift(i, fill_value=False)
        max_daily_fraction = float(self.cfg["load_anomaly_max_daily_fraction"])
        max_daily_points = int(math.floor(max_daily_fraction * slots_per_day))
        dates = pd.Series(local.date, index=df.index)
        for d in dates.unique():
            day_mask = dates == d
            if int(mask[day_mask].sum()) > max_daily_points > 0:
                mask.loc[day_mask] = False
        df["load_contaminated"] = mask.astype(bool)
        replacement = expected
        df.loc[df["load_contaminated"], "cleaned_baseline_load_kw"] = replacement[df["load_contaminated"]]
        masked_kwh = float(df.loc[df["load_contaminated"], "baseline_load_kw"].fillna(0.0).sum() * int(self.cfg["base_interval_minutes"]) / 60.0)
        self.load_anomaly_diagnostics["load_anomaly_intervals_masked"] = int(df["load_contaminated"].sum())
        self.load_anomaly_diagnostics["load_anomaly_kwh_masked"] = round(masked_kwh, 3)
        event_groups = (df["load_contaminated"] != df["load_contaminated"].shift(fill_value=False)).cumsum()
        events = []
        for _, g in df[df["load_contaminated"]].groupby(event_groups):
            start_ts = g.index.min()
            end_ts = g.index.max()
            duration = int((end_ts - start_ts).total_seconds() / 60) + int(self.cfg["base_interval_minutes"])
            peak_kw = float(g["baseline_load_kw"].max())
            est_kwh = float(g["baseline_load_kw"].sum() * int(self.cfg["base_interval_minutes"]) / 60.0)
            events.append((start_ts, end_ts, duration, peak_kw, est_kwh))
            self.log(f"Load anomaly event start={start_ts.isoformat()} end={end_ts.isoformat()} duration_minutes={duration} peak_kw={peak_kw:.3f} estimated_masked_kwh={est_kwh:.3f}")
        self.load_anomaly_diagnostics["load_anomaly_events_last_training"] = len(events)
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

    def _statistics_to_series(self, records: List[dict], idx: pd.DatetimeIndex, interval: int, source: str) -> Optional[pd.Series]:
        rows = []
        for rec in records or []:
            ts = rec.get("start") or rec.get("start_time") or rec.get("start_ts")
            if ts is None:
                continue
            if isinstance(ts, (int, float)):
                unit = "ms" if float(ts) > 100000000000 else "s"
                ts = pd.to_datetime(float(ts), unit=unit, utc=True)
            else:
                ts = pd.to_datetime(ts, utc=True)
            val = safe_float(rec.get("mean"))
            if pd.isna(val):
                continue
            rows.append((ts, val))
        if not rows:
            return None
        s = pd.Series([v for _, v in rows], index=pd.DatetimeIndex([t for t, _ in rows])).sort_index()
        if source == "long_term_statistics":
            limit = max(1, int(60 / interval) - 1)
            return s.reindex(idx, method="ffill", limit=limit)
        return s.resample(f"{interval}min").mean().reindex(idx, method="nearest", tolerance=pd.Timedelta(minutes=float(interval) / 2.0))

    def _history_diagnostics(self, base: pd.DataFrame, start: datetime, end: datetime) -> Dict[str, Any]:
        counts = {str(k): int(v) for k, v in base["history_source"].fillna("missing").value_counts().to_dict().items()}
        detailed_days = self._span_days(base.index[base["history_source"] == "detailed_state"])
        stats_days = self._span_days(base.index[base["history_source"].isin(["short_term_statistics", "long_term_statistics"])])
        usable_points = int(base["cleaned_baseline_load_kw"].notna().sum())
        usable_days = round(usable_points * int(self.cfg["base_interval_minutes"]) / 1440.0, 2)
        return {
            "detailed_history_days_available": detailed_days,
            "statistics_history_days_available": stats_days,
            "history_source_counts": counts,
            "usable_baseline_days": usable_days,
        }

    def _span_days(self, index: pd.DatetimeIndex) -> float:
        if len(index) == 0:
            return 0.0
        if len(index) == 1:
            return round(int(self.cfg["base_interval_minutes"]) / 1440.0, 2)
        return round((index.max() - index.min()).total_seconds() / 86400.0, 2)

    def _insufficient_history_message(self, base: pd.DataFrame) -> str:
        diag = self.history_diagnostics or self._history_diagnostics(base, utc_now(), utc_now())
        usable = diag.get("usable_baseline_days", 0)
        detailed = diag.get("detailed_history_days_available", 0)
        stats = diag.get("statistics_history_days_available", 0)
        counts = diag.get("history_source_counts", {})
        reason = "recorder history retention"
        if stats <= 0:
            reason = "missing recorder statistics"
        elif usable < float(self.cfg["minimum_training_days"]):
            reason = "recorder history retention and insufficient usable recorder statistics"
        return (
            "Insufficient baseline load history after preparation: "
            f"minimum_training_days={self.cfg['minimum_training_days']} "
            f"usable_baseline_days={usable} "
            f"detailed_history_days_available={detailed} "
            f"statistics_history_days_available={stats} "
            f"history_source_counts={counts} "
            f"limitation={reason}"
        )

    def _build_origin_training_table(self, base: pd.DataFrame, start: datetime, end: datetime, validation_cutoff: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        origin_interval = int(self.cfg["training_origin_interval_minutes"])
        lead_interval = int(self.cfg["training_lead_interval_minutes"])
        horizon_minutes = int(self.cfg["horizon_hours"] * 60)
        earliest_origin = base.index.min() + pd.Timedelta(days=7)
        latest_origin = base.index.max() - pd.Timedelta(minutes=horizon_minutes)
        origins = pd.date_range(earliest_origin.ceil(f"{origin_interval}min"), latest_origin.floor(f"{origin_interval}min"), freq=f"{origin_interval}min", tz="UTC")
        leads = np.arange(lead_interval, horizon_minutes + 1, lead_interval, dtype=int)
        t0 = time.time()
        weather_hist = self._fetch_weather_training(start, end) if self.cfg["weather_enabled"] else pd.DataFrame()
        self._log_stage_elapsed("historical weather fetch", t0)
        self.log(
            "Building origin training table: "
            f"candidate_origins={len(origins)} leads_per_origin={len(leads)} "
            f"estimated_max_rows={len(origins) * len(leads)}"
        )
        origin_df = pd.DataFrame({"origin_time": np.repeat(origins.to_numpy(), len(leads)), "lead_minutes": np.tile(leads, len(origins)).astype(float)})
        origin_feat = pd.DataFrame(index=origins)
        for hours in [1, 6, 24, 168]:
            window = max(1, int(hours * 60 / self.cfg["base_interval_minutes"]))
            origin_feat[f"origin_rolling_mean_{hours}h_kw"] = base["cleaned_baseline_load_kw"].rolling(window=window, min_periods=max(2, window // 2)).mean().reindex(origins)
            if hours >= 24:
                origin_feat[f"origin_rolling_std_{hours}h_kw"] = base["cleaned_baseline_load_kw"].rolling(window=window, min_periods=max(2, window // 2)).std().reindex(origins)
        for col in ["import_price", "feed_in_price", "ev_charging_flag"]:
            if col in base.columns:
                origin_feat[f"origin_{col}"] = base[col].reindex(origins)
        for extra in self.cfg.get("additional_feature_sensors") or []:
            col = safe_feature_name(extra)
            if col in base.columns:
                origin_feat[f"origin_{col}"] = base[col].reindex(origins)
                origin_feat[f"origin_{col}_is_origin_value"] = 1.0
        origin_feat["origin_history_ok"] = (
            base["cleaned_baseline_load_kw"].notna().rolling(window=max(1, int(7 * 24 * 60 / self.cfg["base_interval_minutes"])), min_periods=1).sum().reindex(origins)
            >= int(7 * 24 * 60 / self.cfg["base_interval_minutes"])
        )
        origin_df = origin_df.merge(origin_feat.reset_index().rename(columns={"index": "origin_time"}), on="origin_time", how="left")
        origin_df = origin_df[origin_df["origin_history_ok"]].drop(columns=["origin_history_ok"])
        origin_df["target_time"] = pd.to_datetime(origin_df["origin_time"], utc=True) + pd.to_timedelta(origin_df["lead_minutes"], unit="m")
        target_map = base[["cleaned_baseline_load_kw"]].rename(columns={"cleaned_baseline_load_kw": "target_load_kw"})
        table = origin_df.merge(target_map, left_on="target_time", right_index=True, how="left")
        table = table[table["target_load_kw"].notna()].copy()
        local = table["target_time"].dt.tz_convert(self.tz)
        hour = local.dt.hour + local.dt.minute / 60.0
        dow = local.dt.dayofweek
        month = local.dt.month
        doy = local.dt.dayofyear
        table["target_hour_sin"] = np.sin(2 * math.pi * hour / 24.0)
        table["target_hour_cos"] = np.cos(2 * math.pi * hour / 24.0)
        table["target_dow_sin"] = np.sin(2 * math.pi * dow / 7.0)
        table["target_dow_cos"] = np.cos(2 * math.pi * dow / 7.0)
        table["target_month_sin"] = np.sin(2 * math.pi * (month - 1) / 12.0)
        table["target_month_cos"] = np.cos(2 * math.pi * (month - 1) / 12.0)
        table["target_doy_sin"] = np.sin(2 * math.pi * doy / 365.0)
        table["target_doy_cos"] = np.cos(2 * math.pi * doy / 365.0)
        table["target_is_weekend"] = (dow >= 5).astype(float)
        date_vals = local.dt.date
        if self.holiday_cal is not None:
            table["target_is_public_holiday"] = date_vals.map(lambda d: 1.0 if d in self.holiday_cal else 0.0)
            table["target_day_before_holiday"] = date_vals.map(lambda d: 1.0 if (d + timedelta(days=1)) in self.holiday_cal else 0.0)
        else:
            table["target_is_public_holiday"] = 0.0
            table["target_day_before_holiday"] = 0.0
        series_col = "cleaned_baseline_load_kw" if self.cfg.get("clean_contaminated_lags", True) and "cleaned_baseline_load_kw" in base else "baseline_load_kw"
        for hours in [24, 48, 168]:
            lagged = base[[series_col, "load_contaminated"]].copy()
            lagged.index = lagged.index + pd.Timedelta(hours=hours)
            lagged = lagged.rename(columns={series_col: f"target_lag_{hours}h_kw", "load_contaminated": f"lag_{hours}h_contaminated"})
            table = table.merge(lagged, left_on="target_time", right_index=True, how="left")
            valid = (table["target_time"] - pd.Timedelta(hours=hours)) <= table["origin_time"]
            table[f"target_lag_{hours}h_valid"] = (valid & table[f"target_lag_{hours}h_kw"].notna()).astype(float)
            table.loc[~valid, f"target_lag_{hours}h_kw"] = np.nan
            table[f"lag_{hours}h_contaminated"] = np.where(valid & table[f"lag_{hours}h_contaminated"].fillna(0).ge(0.5), 1.0, 0.0)
        if not weather_hist.empty:
            wx = weather_hist.reset_index().rename(columns={"timestamp": "weather_time"})
            table = pd.merge_asof(table.sort_values("target_time"), wx.sort_values("weather_time"), left_on="target_time", right_on="weather_time", direction="nearest", tolerance=pd.Timedelta(minutes=65))
        for v in OPEN_METEO_VARS:
            col = f"wx_{v}"
            if col not in table.columns:
                table[col] = 0.0
        table["lead_sin"] = np.sin(2 * math.pi * table["lead_minutes"] / horizon_minutes)
        table["lead_cos"] = np.cos(2 * math.pi * table["lead_minutes"] / horizon_minutes)
        table["lead_bucket"] = table["lead_minutes"].map(lead_bucket_name)
        table = table.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        train, val = self._split_train_validation_rows(table, validation_cutoff)
        if train.empty or val.empty:
            self.log(
                "Unable to build non-empty train and validation tables: "
                f"total_rows={len(table)} "
                f"unique_origins={pd.to_datetime(table['origin_time'], utc=True).nunique()} "
                f"requested_validation_cutoff={self._format_optional_timestamp(pd.to_datetime(validation_cutoff, utc=True))} "
                f"first_row_origin={self._format_optional_timestamp(pd.to_datetime(table['origin_time'], utc=True).min())} "
                f"last_row_origin={self._format_optional_timestamp(pd.to_datetime(table['origin_time'], utc=True).max())}",
                level="ERROR",
            )
        train = self._subsample_training_rows(train)
        return train, val

    def _split_train_validation_rows(self, table: pd.DataFrame, validation_cutoff: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        origin_times = pd.to_datetime(table["origin_time"], utc=True)
        requested_cutoff = pd.to_datetime(validation_cutoff, utc=True)
        train_mask = origin_times < requested_cutoff
        val_mask = origin_times >= requested_cutoff
        if train_mask.any() and val_mask.any():
            return table[train_mask].copy(), table[val_mask].copy()

        unique_origins = pd.DatetimeIndex(origin_times.drop_duplicates().sort_values())
        if len(unique_origins) < 2:
            return table[train_mask].copy(), table[val_mask].copy()

        if not val_mask.any():
            effective_cutoff = unique_origins[-1]
        elif not train_mask.any():
            effective_cutoff = unique_origins[1]
        else:
            effective_cutoff = requested_cutoff

        self.log(
            "Adjusted validation cutoff to keep non-empty train and validation tables: "
            f"requested_validation_cutoff={self._format_optional_timestamp(requested_cutoff)} "
            f"effective_validation_cutoff={self._format_optional_timestamp(effective_cutoff)} "
            f"first_row_origin={self._format_optional_timestamp(unique_origins[0])} "
            f"last_row_origin={self._format_optional_timestamp(unique_origins[-1])} "
            f"unique_origins={len(unique_origins)}",
            level="WARNING",
        )
        train_mask = origin_times < effective_cutoff
        val_mask = origin_times >= effective_cutoff
        return table[train_mask].copy(), table[val_mask].copy()

    def _features_known_at_origin(self, base: pd.DataFrame, origin: pd.Timestamp) -> Optional[Dict[str, float]]:
        matched_origin = self._nearest_index_timestamp(base.index, origin)
        if matched_origin is None:
            return None
        out: Dict[str, float] = {}
        history = base.loc[:matched_origin, "cleaned_baseline_load_kw"].dropna()
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
                out[f"origin_{col}"] = float(base.loc[matched_origin, col]) if not pd.isna(base.loc[matched_origin, col]) else 0.0
        for extra in self.cfg.get("additional_feature_sensors") or []:
            col = safe_feature_name(extra)
            if col in base.columns:
                out[f"origin_{col}"] = float(base.loc[matched_origin, col]) if not pd.isna(base.loc[matched_origin, col]) else 0.0
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
            lag_flag = self._lookup_series_value(base["load_contaminated"].astype(float), lag_time) if valid and "load_contaminated" in base else 0.0
            series_col = "cleaned_baseline_load_kw" if self.cfg.get("clean_contaminated_lags", True) and "cleaned_baseline_load_kw" in base else "baseline_load_kw"
            val = self._lookup_series_value(base[series_col], lag_time) if valid else np.nan
            out[f"target_lag_{hours}h_kw"] = 0.0 if pd.isna(val) else float(val)
            out[f"target_lag_{hours}h_valid"] = 1.0 if valid and not pd.isna(val) else 0.0
            out[f"lag_{hours}h_contaminated"] = 1.0 if valid and float(lag_flag) >= 0.5 else 0.0
        return out

    def _lookup_series_value(self, series: pd.Series, ts: pd.Timestamp) -> float:
        try:
            matched_ts = self._nearest_index_timestamp(series.index, ts)
            if matched_ts is None:
                return np.nan
            return float(series.loc[matched_ts])
        except Exception:
            return np.nan

    def _nearest_index_timestamp(self, index: pd.DatetimeIndex, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
        if len(index) == 0:
            return None
        tolerance = pd.Timedelta(minutes=float(self.cfg["base_interval_minutes"]) / 2.0)
        try:
            target = pd.to_datetime(ts, utc=True)
            pos = index.get_indexer([target], method="nearest", tolerance=tolerance)[0]
            if pos == -1:
                return None
            return index[pos]
        except Exception:
            return None

    def _format_optional_timestamp(self, ts: Optional[pd.Timestamp]) -> Optional[str]:
        if ts is None or pd.isna(ts):
            return None
        return pd.Timestamp(ts).isoformat()

    def _fetch_weather_training(self, start: datetime, end: datetime) -> pd.DataFrame:
        if not self.cfg.get("latitude") or not self.cfg.get("longitude"):
            return pd.DataFrame()
        cached = self._read_weather_cache()
        if cached is not None and not cached.empty:
            cache_slice = cached[(cached.index >= pd.to_datetime(start, utc=True)) & (cached.index <= pd.to_datetime(end, utc=True))]
            if not cache_slice.empty and cache_slice.index.min() <= pd.to_datetime(start, utc=True) and cache_slice.index.max() >= pd.to_datetime(end, utc=True):
                return cache_slice
        params = {
            "latitude": self.cfg["latitude"],
            "longitude": self.cfg["longitude"],
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "hourly": ",".join(OPEN_METEO_VARS),
            "timezone": "UTC",
            "wind_speed_unit": "kmh",
        }
        fetched = self._fetch_weather_df(HISTORICAL_FORECAST_URL, params)
        if fetched.empty:
            return fetched
        if cached is not None and not cached.empty:
            fetched = pd.concat([cached, fetched]).sort_index()
            fetched = fetched[~fetched.index.duplicated(keep="last")]
        self._write_weather_cache(fetched)
        return fetched[(fetched.index >= pd.to_datetime(start, utc=True)) & (fetched.index <= pd.to_datetime(end, utc=True))]

    def _read_weather_cache(self) -> Optional[pd.DataFrame]:
        if not self.weather_cache_path.exists():
            return None
        try:
            cached = pd.read_parquet(self.weather_cache_path)
            cached.index = pd.to_datetime(cached.index, utc=True)
            return cached.sort_index()
        except Exception as e:
            self.log(f"Weather cache read failed: {e}", level="WARNING")
            return None

    def _write_weather_cache(self, df: pd.DataFrame):
        try:
            df.to_parquet(self.weather_cache_path)
        except Exception as e:
            self.log(f"Weather cache write failed: {e}", level="WARNING")

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
        stage_t0 = time.time()
        entity_id = self.cfg["hafo_entity_id"]
        self.log(f"Generating forecast for {entity_id}")
        if self.model is None:
            raise RuntimeError("Cannot generate forecast: model is not trained")
        if self.training_frame is None:
            raise RuntimeError("Cannot generate forecast: training frame is unavailable")
        if not self.feature_cols:
            raise RuntimeError("Cannot generate forecast: feature columns are unavailable")
        now = utc_now()
        pub_interval = int(self.cfg["publish_interval_minutes"])
        train_lead_interval = int(self.cfg["training_lead_interval_minutes"])
        horizon_minutes = int(self.cfg["horizon_hours"] * 60)
        forecast_start = self._next_boundary(now, pub_interval)
        # Predict at training lead resolution then interpolate to publish resolution.
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
        if len(preds_publish) == 0:
            raise RuntimeError("Cannot publish forecast: generated forecast is empty")
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
        self._log_stage_elapsed("forecast generation", stage_t0)

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
            "source": "powersync_ml_forecaster",
        })
        self.set_state(self.cfg["hafo_entity_id"], state=state, attributes=attrs)

    def _publish_forecast(self, forecast: List[dict], predictions: np.ndarray, daily_totals: Dict[str, float]):
        entity_id = self.cfg["hafo_entity_id"]
        attrs = {
            "forecast": forecast,
            "status": "ok",
            "friendly_name": self.cfg["friendly_name"],
            "unit_of_measurement": "kW",
            "device_class": "power",
            "state_class": "measurement",
            "source": "powersync_ml_forecaster",
            "source_entity": self.cfg["load_sensor"],
            "last_updated": utc_now().astimezone(self.tz).isoformat(),
            "last_trained": self.last_train_time.astimezone(self.tz).isoformat() if self.last_train_time else None,
            "horizon_hours": self.cfg["horizon_hours"],
            "interval_minutes": self.cfg["publish_interval_minutes"],
            "publish_interval_minutes": self.cfg["publish_interval_minutes"],
            "training_lead_interval_minutes": self.cfg["training_lead_interval_minutes"],
            "interpolation_mode": "predict_training_leads_then_linear_interpolate_to_publish_interval",
            "model": "HistGradientBoostingRegressor_absolute_error",
            "validation": self.validation_metrics,
            "history_diagnostics": getattr(self, "history_diagnostics", {}),
            "weather_training_mode": self.cfg["weather_training_mode"],
            "price_training_mode": self.cfg["price_training_mode"],
            "ev_exclusion_threshold_kw": self.cfg["ev_exclusion_threshold_kw"],
            "ev_excluded_kwh_last_training": self._ev_excluded_kwh(),
            "load_anomaly_filter_enabled": self.load_anomaly_diagnostics.get("load_anomaly_filter_enabled"),
            "load_anomaly_events_last_training": self.load_anomaly_diagnostics.get("load_anomaly_events_last_training"),
            "load_anomaly_intervals_masked": self.load_anomaly_diagnostics.get("load_anomaly_intervals_masked"),
            "load_anomaly_kwh_masked": self.load_anomaly_diagnostics.get("load_anomaly_kwh_masked"),
            "load_anomaly_replacement": self.load_anomaly_diagnostics.get("load_anomaly_replacement"),
            "contaminated_lag_values_replaced": self._contaminated_lag_values_replaced(),
            "daily_forecast_kwh": {k: round(v, 2) for k, v in daily_totals.items()},
        }
        state = str(round(float(predictions[0]), 3)) if len(predictions) else "0"
        self.log(f"Publishing forecast sensor {entity_id}")
        status_code = self.set_state(entity_id, state=state, attributes=attrs)
        if status_code is None:
            self.log(f"Published forecast sensor {entity_id} status=unknown")
        else:
            self.log(f"Published forecast sensor {entity_id} status={status_code}")

    def _ev_excluded_kwh(self) -> Optional[float]:
        if self.training_frame is None or "ev_excluded_kw" not in self.training_frame:
            return None
        return round(float(self.training_frame["ev_excluded_kw"].sum() * self.cfg["base_interval_minutes"] / 60.0), 3)

    def _contaminated_lag_values_replaced(self) -> int:
        if self.training_frame is None:
            return 0
        cols = [c for c in ["lag_24h_contaminated", "lag_48h_contaminated", "lag_168h_contaminated"] if c in self.training_frame.columns]
        if not cols:
            return 0
        return int(self.training_frame[cols].sum().sum())


    def _write_latest_forecast(self, forecast: List[dict]):
        rec = {
            "created_at": utc_now().isoformat(),
            "entity_id": self.cfg["hafo_entity_id"],
            "forecast": forecast,
            "unit_of_measurement": "kW",
            "source": "powersync_ml_forecaster",
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
