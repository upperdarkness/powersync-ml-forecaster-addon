#!/usr/bin/env python3
"""Publish a mock HAFO-compatible sensor to prove PowerSync will consume it."""
import os, sys, requests
from datetime import datetime, timedelta, timezone

HA_URL = os.environ.get("HA_URL")
HA_TOKEN = os.environ.get("HA_TOKEN")
ENTITY_ID = os.environ.get("HAFO_ENTITY_ID", "sensor.hafo_power_sync_home_load_forecast")
INTERVAL_MINUTES = int(os.environ.get("INTERVAL_MINUTES", "5"))
HORIZON_HOURS = int(os.environ.get("HORIZON_HOURS", "48"))
if not HA_URL or not HA_TOKEN:
    sys.exit("Set HA_URL and HA_TOKEN")

now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
forecast = []
steps = HORIZON_HOURS * 60 // INTERVAL_MINUTES
for i in range(1, steps + 1):
    t = now + timedelta(minutes=i * INTERVAL_MINUTES)
    h = t.hour
    value = 2.5 if 6 <= h <= 8 or 17 <= h <= 21 else 0.8 if 9 <= h <= 16 else 0.4
    forecast.append({"time": t.isoformat(), "value": value})
payload = {
    "state": str(forecast[0]["value"]),
    "attributes": {
        "forecast": forecast,
        "unit_of_measurement": "kW",
        "device_class": "power",
        "state_class": "measurement",
        "friendly_name": "HAFO ML Mock Load Forecast",
        "source": "powersync_ml_mock_compatibility_test",
        "status": "ok",
        "horizon_hours": HORIZON_HOURS,
        "publish_interval_minutes": INTERVAL_MINUTES,
    },
}
r = requests.post(f"{HA_URL.rstrip('/')}/api/states/{ENTITY_ID}", headers={"Authorization": f"Bearer {HA_TOKEN}"}, json=payload, timeout=10)
print(f"HTTP {r.status_code}")
print(r.text[:500])
