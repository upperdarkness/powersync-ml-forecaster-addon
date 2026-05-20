#!/usr/bin/env python3
"""Compare ML forecast with PowerSync LP load forecast where attributes are parseable."""
import os, sys, requests, math
import numpy as np

HA_URL = os.environ.get("HA_URL")
HA_TOKEN = os.environ.get("HA_TOKEN")
ML_ENTITY_ID = os.environ.get("HAFO_ENTITY_ID", "sensor.hafo_power_sync_home_load_forecast")
PS_ENTITY_ID = os.environ.get("POWERSYNC_LP_ENTITY_ID", "sensor.power_sync_lp_load_forecast")
if not HA_URL or not HA_TOKEN:
    sys.exit("Set HA_URL and HA_TOKEN")
headers = {"Authorization": f"Bearer {HA_TOKEN}"}
ml = requests.get(f"{HA_URL.rstrip('/')}/api/states/{ML_ENTITY_ID}", headers=headers, timeout=10).json()
ps = requests.get(f"{HA_URL.rstrip('/')}/api/states/{PS_ENTITY_ID}", headers=headers, timeout=10).json()
ml_values = [float(f["value"]) for f in ml.get("attributes", {}).get("forecast", [])[:48]]
attrs = ps.get("attributes", {})
ps_attr = attrs.get("forecast_load") or attrs.get("forecast") or attrs.get("load_forecast") or []
ps_values = []
if isinstance(ps_attr, list):
    for item in ps_attr[:48]:
        if isinstance(item, dict):
            v = item.get("value") or item.get("load") or item.get("power")
        else:
            v = item
        try: ps_values.append(float(v))
        except Exception: pass
if len(ml_values) >= 6 and len(ps_values) >= 6:
    n = min(len(ml_values), len(ps_values))
    corr = float(np.corrcoef(ml_values[:n], ps_values[:n])[0,1]) if n > 1 else float("nan")
    print(f"Compared {n} points. Correlation={corr:.3f}")
    if corr >= 0.90:
        print("PASSED: PowerSync forecast shape matches ML forecast")
    else:
        print("CHECK MANUALLY: correlation below 0.90")
else:
    print("Could not parse enough PowerSync forecast points.")
    print("Open the PowerSync LP forecast chart and compare it visually to the ML mock or ML forecast.")
    print(f"PowerSync state: {ps.get('state')}")
    print(f"PowerSync attributes available: {sorted(attrs.keys())}")
