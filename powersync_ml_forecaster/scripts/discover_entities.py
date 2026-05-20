#!/usr/bin/env python3
"""List candidate Home Assistant entities for the PowerSync ML forecaster."""
import os, sys, json, requests

HA_URL = os.environ.get("HA_URL")
HA_TOKEN = os.environ.get("HA_TOKEN")
if not HA_URL or not HA_TOKEN:
    sys.exit("Set HA_URL and HA_TOKEN")

r = requests.get(f"{HA_URL.rstrip('/')}/api/states", headers={"Authorization": f"Bearer {HA_TOKEN}"}, timeout=30)
r.raise_for_status()
states = r.json()
patterns = {
    "load": ["load", "home_load", "consumption", "power_consumption", "grid_power"],
    "ev": ["ocular", "ev", "charger", "wallbox", "charge_power"],
    "amber_import": ["amber", "general", "import", "price"],
    "amber_export": ["amber", "feed", "export", "fit", "price"],
    "solar": ["fronius", "solar", "pv", "inverter"],
    "battery_soc": ["powerwall", "battery", "soc", "charge"],
    "occupancy": ["home", "presence", "person", "occupancy"],
}
for cat, keys in patterns.items():
    print(f"\n=== {cat} candidates ===")
    for s in states:
        eid = s.get("entity_id", "")
        if not eid.startswith(("sensor.", "binary_sensor.", "person.")):
            continue
        low = eid.lower()
        if all(k in low for k in keys) or any(k in low for k in keys[:2]):
            attrs = s.get("attributes", {})
            unit = attrs.get("unit_of_measurement", "")
            print(f"{eid:70s} state={str(s.get('state'))[:16]:16s} unit={unit}")
