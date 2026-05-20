"""Standalone Home Assistant add-on runner for PowerSync ML Load Forecaster v4.2.3.

This wrapper lets the AppDaemon-oriented forecaster run as a normal Home
Assistant add-on. It provides the small subset of AppDaemon methods used by the
forecaster via the Home Assistant Core REST API.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from powersync_ml_forecaster import PowerSyncMLForecaster

OPTIONS_PATH = Path("/data/options.json")

OPTION_KEYS = {
    "hafo_entity_id",
    "friendly_name",
    "load_sensor",
    "load_unit_override",
    "ev_power_sensor",
    "ev_power_unit_override",
    "ev_exclusion_threshold_kw",
    "import_price_sensor",
    "feed_in_price_sensor",
    "additional_feature_sensors",
    "latitude",
    "longitude",
    "timezone",
    "public_holiday_region",
    "days_back",
    "minimum_training_days",
    "horizon_hours",
    "base_interval_minutes",
    "training_origin_interval_minutes",
    "training_lead_interval_minutes",
    "publish_interval_minutes",
    "retrain_minutes",
    "update_minutes",
    "validation_days",
    "max_training_rows",
    "history_fetch_chunk_days",
    "max_total_history_fetch_seconds",
    "max_single_history_chunk_seconds",
    "max_training_frame_mb",
    "weather_enabled",
    "weather_training_mode",
    "price_training_mode",
    "use_live_price_forecast_features",
    "data_dir",
    "cache_max_age_hours",
    "snapshot_retention_days",
    "require_quality_gate",
    "min_model_improvement_pct",
    "ev_artifact_min_false_load_reduction_pct",
    "ev_artifact_min_mae_improvement_pct",
    "model_max_iter",
    "model_learning_rate",
    "model_max_depth",
    "model_min_samples_leaf",
}


def _clean_options(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert add-on UI values into forecaster config values."""
    cleaned: Dict[str, Any] = {}
    for key, value in raw.items():
        if key not in OPTION_KEYS:
            continue
        if isinstance(value, str):
            value = value.strip()
            if value == "" and key not in {"load_sensor", "hafo_entity_id", "friendly_name", "timezone", "data_dir"}:
                value = None
        if key == "additional_feature_sensors":
            value = [str(v).strip() for v in (value or []) if str(v).strip()]
        cleaned[key] = value

    # Use Supervisor/Core API by default. The add-on receives SUPERVISOR_TOKEN
    # automatically when homeassistant_api: true is set in config.yaml.
    cleaned["ha_url"] = os.environ.get("HA_URL") or "http://supervisor/core"
    cleaned["ha_token"] = os.environ.get("HASS_TOKEN") or os.environ.get("SUPERVISOR_TOKEN")
    return cleaned


def load_options() -> Dict[str, Any]:
    if not OPTIONS_PATH.exists():
        raise FileNotFoundError("/data/options.json was not found. Is this running as a Home Assistant add-on?")
    raw = json.loads(OPTIONS_PATH.read_text())
    return _clean_options(raw)


class StandaloneForecaster(PowerSyncMLForecaster):
    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self._session = requests.Session()
        self._stop_requested = False
        token = args.get("ha_token") or os.environ.get("SUPERVISOR_TOKEN")
        if not token:
            raise RuntimeError("No Home Assistant API token available. Ensure the add-on has homeassistant_api: true.")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self._ha_url = args.get("ha_url") or "http://supervisor/core"
        token_source = "configured"
        if token == os.environ.get("SUPERVISOR_TOKEN"):
            token_source = "SUPERVISOR_TOKEN"
        elif token == os.environ.get("HASS_TOKEN"):
            token_source = "HASS_TOKEN"
        self.log(f"Home Assistant Core API base URL: {self._ha_url.rstrip('/')}; token_source={token_source}")

    # AppDaemon compatibility methods -------------------------------------------------
    def log(self, msg: str, level: str = "INFO", **_: Any) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        print(f"{ts} [{level}] {msg}", flush=True)

    def get_timezone(self) -> str:
        return self.args.get("timezone") or "UTC"

    def run_in(self, *args: Any, **kwargs: Any) -> None:
        # Scheduling is handled by this runner's main loop.
        return None

    def run_every(self, *args: Any, **kwargs: Any) -> None:
        # Scheduling is handled by this runner's main loop.
        return None

    def _url(self, path: str) -> str:
        return f"{self._ha_url.rstrip('/')}/{path.lstrip('/')}"

    def get_state(self, entity_id: str, attribute: Optional[str] = None) -> Any:
        r = self._session.get(self._url(f"/api/states/{entity_id}"), headers=self._headers, timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        if attribute == "all":
            return data
        if attribute:
            return data.get("attributes", {}).get(attribute)
        return data.get("state")

    def set_state(self, entity_id: str, state: str, attributes: Optional[Dict[str, Any]] = None) -> int:
        payload = {"state": state, "attributes": attributes or {}}
        url = self._url(f"/api/states/{entity_id}")
        r = self._session.post(url, headers=self._headers, json=payload, timeout=20)
        if r.status_code >= 400:
            body = (r.text or "")[:500]
            self.log(
                f"Home Assistant state publish failed for {entity_id}: status={r.status_code} body={body}",
                level="ERROR",
            )
            r.raise_for_status()
        return int(r.status_code)

    def get_history(self, entity_id: str, start_time: str, end_time: str, **_: Any) -> Any:
        params = {
            "filter_entity_id": entity_id,
            "end_time": end_time,
            "minimal_response": "true",
        }
        r = self._session.get(
            self._url(f"/api/history/period/{start_time}"),
            headers=self._headers,
            params=params,
            timeout=90,
        )
        r.raise_for_status()
        return r.json()

    # Standalone lifecycle -----------------------------------------------------------
    def stop(self, *_: Any) -> None:
        self._stop_requested = True

    def serve_forever(self) -> None:
        self.initialize()

        # The AppDaemon version schedules the first training callback. In add-on mode,
        # trigger it directly so the first useful sensor state appears as soon as possible.
        self._train_callback({})

        next_train = time.monotonic() + int(self.cfg["retrain_minutes"]) * 60
        next_refresh = time.monotonic() + int(self.cfg["update_minutes"]) * 60

        while not self._stop_requested:
            now = time.monotonic()
            try:
                if now >= next_train:
                    self._train_callback({})
                    next_train = now + int(self.cfg["retrain_minutes"]) * 60
                    # Training publishes a forecast; avoid immediate duplicate refresh.
                    next_refresh = now + int(self.cfg["update_minutes"]) * 60
                elif now >= next_refresh:
                    self._forecast_callback({})
                    next_refresh = now + int(self.cfg["update_minutes"]) * 60
            except Exception:
                self.log(traceback.format_exc(), level="ERROR")
            time.sleep(5)


def main() -> int:
    args = load_options()
    app = StandaloneForecaster(args)
    signal.signal(signal.SIGTERM, app.stop)
    signal.signal(signal.SIGINT, app.stop)
    app.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
