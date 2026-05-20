#!/usr/bin/env python3
"""Validate the published HAFO-compatible forecast sensor schema."""
import os, sys, requests
from datetime import datetime

HA_URL = os.environ.get("HA_URL")
HA_TOKEN = os.environ.get("HA_TOKEN")
ENTITY_ID = os.environ.get("HAFO_ENTITY_ID", "sensor.hafo_power_sync_home_load_forecast")
if not HA_URL or not HA_TOKEN:
    sys.exit("Set HA_URL and HA_TOKEN")
r = requests.get(f"{HA_URL.rstrip('/')}/api/states/{ENTITY_ID}", headers={"Authorization": f"Bearer {HA_TOKEN}"}, timeout=10)
r.raise_for_status()
s = r.json()
attrs = s.get("attributes", {})
forecast = attrs.get("forecast", [])
fail = []
try: float(s.get("state"));
except Exception: fail.append("state is not numeric")
if attrs.get("status") != "ok": fail.append("status is not ok")
if not isinstance(forecast, list) or not forecast: fail.append("forecast is missing or empty")
else:
    if not all("time" in f and "value" in f for f in forecast[:5]): fail.append("forecast entries lack time/value")
    try:
        times = [datetime.fromisoformat(f["time"]) for f in forecast]
        span = (times[-1] - times[0]).total_seconds() / 3600
        if not (47 <= span <= 49): fail.append(f"forecast span is {span:.1f}h, expected about 48h")
    except Exception as e:
        fail.append(f"forecast times are not parseable: {e}")
if attrs.get("unit_of_measurement") != "kW": fail.append("unit_of_measurement is not kW")
if fail:
    print("FAILED")
    for f in fail: print(f" - {f}")
    sys.exit(1)
print("PASSED")
print(f"{ENTITY_ID}: {len(forecast)} forecast points, state={s.get('state')} kW")
