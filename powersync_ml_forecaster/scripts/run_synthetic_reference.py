#!/usr/bin/env python3
"""
Fast synthetic reference test for v4.2 origin-based training.
No Home Assistant required. This intentionally uses a simple ridge-like least squares
model so it runs quickly on small systems; the deployed AppDaemon app uses
HistGradientBoostingRegressor.
"""
import math
import numpy as np

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
