# PowerSync ML Load Forecaster — Specification v4.2
## Implementation ready specification for Codex, Claude Code or Claude Cowork

---

## 1. Purpose

Build a Home Assistant AppDaemon application that publishes a HAFO compatible load forecast sensor for PowerSync.

The forecaster predicts **baseline household load in kW**, excluding EV charging load, so PowerSync can schedule EV charging on top of the baseline instead of treating a previous EV charging session as normal household consumption.

Version 4.2 builds on v4.1 and closes the remaining implementation gaps:

1. Weather source modes are explicit and the Historical Forecast API proxy is the default.
2. Price features default to origin current price only unless stored forecast snapshots exist.
3. Training lead resolution and published forecast resolution have an explicit interpolation rule.
4. PowerSync fallback has a minimal custom integration entity contract.
5. Model startup and cache loading behaviour are specified.
6. Recorder and memory safety abort limits are specified.
7. Overall MAE quality gates are supplemented with an EV artefact gate.
8. A runnable synthetic reference skeleton is included so coding agents have a concrete implementation target.
9. Dependency ranges are tightened.
10. The specification distinguishes confirmed design constraints from configurable approximations.

---

## 2. Scope

### In scope

- AppDaemon 4 application.
- One published Home Assistant sensor containing a HAFO style `forecast` attribute.
- Baseline household load forecast in kW.
- EV charging exclusion from the training target.
- Forecast origin based training and validation.
- Weather features using Open Meteo historical forecast and live forecast APIs.
- Amber import and feed in price features where Home Assistant forecast attributes are parseable.
- Optional contextual sensors, but only as origin time features.
- Forecast snapshots for live error measurement.
- PowerSync compatibility test before the ML model is deployed.
- Diagnostics that make stale or fallback data obvious.

### Out of scope

- Forecasting planned EV charging sessions.
- Replacing PowerSync's optimiser.
- Forecasting solar generation unless a separate solar forecast source is added.
- Forecasting battery SOC.
- Editing PowerSync source code as the default path.
- Creating a native Home Assistant custom integration as the default path.

If the compatibility test proves that PowerSync will not consume an AppDaemon-created state entity, the fallback options are:

1. Configure PowerSync to use the exact entity if it exposes a setting for this.
2. Patch PowerSync's sensor discovery logic to accept the ML sensor.
3. Build a minimal Home Assistant custom integration to create an entity registry backed sensor using the entity contract in section 8.5.
4. Stop the AppDaemon deployment and keep HAFO or PowerSync persistence until one of the above is done.

Do not proceed on the assumption that a transient state entity is enough. The mock sensor test is the evidence gate.

---

## 3. Required user inputs

The implementing agent must obtain these before making changes:

1. Home Assistant URL, for example `http://homeassistant.local:8123`.
2. Long lived Home Assistant access token.
3. SSH, Samba or File Editor access to the Home Assistant filesystem.
4. Confirmation that the AppDaemon 4 add on is installed and running.
5. Confirmation that PowerSync is installed and has run at least once.
6. Household load sensor entity ID and unit.
7. EV charger power sensor entity ID and unit, if available.
8. Amber import price sensor entity ID, if available.
9. Amber feed in price sensor entity ID, if available.
10. Latitude and longitude for weather.
11. Time zone, for example `Australia/Adelaide`.
12. Australian state for public holidays, for example `SA`.
13. Preferred output HAFO entity ID after compatibility testing.

Do not assume entity IDs. Discover candidates and ask the user to confirm them.

---

## 4. Success criteria

The implementation is successful when all of these are true:

```text
1. A HAFO compatible forecast entity exists in Home Assistant.
2. The entity state is numeric and represents the first forecast point in kW.
3. The entity has status: ok.
4. The entity attribute forecast is a list of objects containing time and value.
5. The forecast covers approximately 48 hours.
6. PowerSync's LP load forecast matches the ML forecast with correlation >= 0.90 over comparable points, or the plotted shape and magnitude visibly match when the LP attribute cannot be parsed.
7. EV charging intervals are excluded from the training target.
8. The model publishes MAE by forecast lead bucket.
9. The model beats the best valid persistence baseline by at least 5 percent overall or in PowerSync relevant lead buckets, or it passes the EV artefact gate.
10. On validation days following EV charging, the model reduces false repeated EV load by at least 50 percent or beats persistence by at least 20 percent during the corresponding charge window.
11. Forecast snapshots are written under /share/powersync_ml_forecaster/.
```

---

## 5. Implementation phases

```text
Phase 1: Discovery
  Find actual Home Assistant entity IDs and confirm units.

Phase 2: PowerSync compatibility test
  Publish a mock HAFO forecast sensor and confirm PowerSync consumes it.

Phase 3: Install AppDaemon files
  Add the application file, config, dependencies, and /share data directory.

Phase 4: First training run
  Train, validate, refit on full data, publish forecast, and inspect diagnostics.

Phase 5: PowerSync handoff
  Confirm PowerSync LP forecast uses the ML forecast.

Phase 6: Live validation
  Compare forecast origin validation, live forecast error, and EV exclusion behaviour.

Phase 7: Tuning
  Adjust sensor selection, horizon resolution, training days, and retrain cadence.
```

---

## 6. Architecture

```text
Home Assistant recorder history
        |
        v
Entity discovery and unit normalisation
        |
        v
5 minute UTC time grid
        |
        v
raw_load_kw and ev_power_kw
        |
        v
baseline_load_kw = raw_load_kw minus EV charging load where ev_power_kw >= threshold
        |
        v
origin based training table:
  one row = forecast_origin_time + target_time + lead_minutes
        |
        v
features known at origin time only:
  calendar at target time
  public holiday at target time
  historical load lags valid at origin time
  rolling load statistics up to origin time
  weather forecast for target time as it would have been known from the origin
  price forecast for target time if parseable from HA attributes
  optional contextual sensors measured at origin time only
        |
        v
HistGradientBoostingRegressor with absolute_error loss
        |
        v
walk forward validation by lead bucket
        |
        v
refit final model on all eligible data
        |
        v
publish HAFO compatible sensor and snapshot JSONL
```

---

## 7. Non negotiable design rules

### 7.1 No target time leakage

For every training row:

```text
origin_time = T0
target_time = T
lead_minutes = T - T0
```

A feature is allowed only if it would have been known at `T0`.

Target time values from ordinary sensors are not allowed unless they are genuine forecasts available at `T0`.

### 7.2 Optional contextual sensors are origin features only

Do not train with `sensor_value[target_time]` and deploy with `sensor_value[now]`. That recreates the same distribution mismatch v4.0 was designed to avoid.

Correct handling:

```text
origin_indoor_temperature = indoor_temperature[origin_time]
origin_someone_home = someone_home[origin_time]
origin_pool_pump_state = pool_pump_state[origin_time]
```

These values are then repeated across all lead rows for that origin. The model never sees actual future values for these sensors during training.

Optional stale flags are allowed but must exist in training as well as deployment:

```text
origin_indoor_temperature_is_origin_value = 1
origin_someone_home_is_origin_value = 1
```

Do not use a flag named `_is_stale_future_value` unless the training set contains the same condition. Prefer `_is_origin_value`.

### 7.3 Valid lag rule

A lag feature for target `T` and lag duration `D` is valid only if:

```text
T - D <= T0
```

Equivalently:

```text
D >= lead_minutes
```

Examples:

| Lead | `lag_24h` valid? | `lag_48h` valid? | `lag_168h` valid? |
|---:|---|---|---|
| 6h | yes | yes | yes |
| 18h | yes | yes | yes |
| 30h | no | yes | yes |
| 42h | no | yes | yes |
| 48h | no or boundary only | yes or boundary only | yes |

Invalid lags must be set to `NaN` or omitted for that row. Do not backfill them from future actual load.

### 7.4 Rolling features are origin time features

Rolling load statistics are computed using baseline load up to, but not after, `origin_time`.

Allowed examples:

```text
origin_rolling_mean_1h
origin_rolling_mean_6h
origin_rolling_mean_24h
origin_rolling_mean_168h
origin_rolling_std_24h
origin_rolling_std_168h
origin_load_point_24h_before_origin
```

The point feature above is deliberately named `origin_load_point_24h_before_origin` to avoid implying that it is an aggregate.

### 7.5 EV load is excluded from the target

If the EV charger power sensor is available:

```text
if ev_power_kw >= ev_exclusion_threshold_kw:
    baseline_load_kw = max(raw_load_kw - ev_power_kw, 0)
else:
    baseline_load_kw = raw_load_kw
```

Default threshold:

```yaml
ev_exclusion_threshold_kw: 0.5
```

Publish diagnostics:

```text
ev_exclusion_enabled
ev_excluded_intervals
ev_excluded_kwh
ev_exclusion_threshold_kw
ev_power_unit
```

Do not require EV power to appear as an important model feature. EV power should primarily affect the target cleanup, not become a predictor of baseline household load.

---

## 8. PowerSync compatibility test

Do this before installing the ML application.

### 8.1 Determine the expected entity ID

Possible output IDs:

```text
sensor.hafo_load_forecast
sensor.hafo_<source_sensor_object_id>_forecast
sensor.hafo_power_sync_home_load_forecast
```

Use PowerSync settings and logs to identify what it tries to read. If PowerSync lets the user configure the forecast entity, use that configured value.

### 8.2 Publish a mock sensor

Publish a simple 48 hour forecast with obvious morning and evening peaks.

The state must be the first forecast value. The `forecast` attribute must be a list of:

```json
{"time":"2026-05-19T10:00:00+09:30","value":1.23}
```

### 8.3 Pass condition

PowerSync passes compatibility if its LP load forecast matches the mock forecast.

Quantitative pass condition where comparable arrays are available:

```text
correlation >= 0.90
median_absolute_ratio between 0.80 and 1.25
```

Manual pass condition where arrays cannot be parsed:

```text
PowerSync's LP forecast chart visibly follows the mock peaks and troughs within one optimisation cycle.
```

### 8.4 Failure handling

If the mock sensor is not consumed:

1. Check the exact entity ID in PowerSync logs.
2. Check whether PowerSync requires hourly resolution instead of 5 minute resolution.
3. Check whether it expects W rather than kW.
4. Check whether it requires a Home Assistant entity registry entry rather than a transient state entity.
5. If an entity registry entry is required, stop the AppDaemon path and either patch PowerSync or build a minimal custom integration.

Do not deploy the ML forecaster until one of these compatibility paths succeeds.


### 8.5 Minimal custom integration fallback entity contract

If PowerSync requires a registered Home Assistant entity rather than an AppDaemon-created state, the fallback custom integration must expose this sensor contract:

```text
platform: sensor
entity_id: sensor.hafo_power_sync_home_load_forecast, or the exact entity ID confirmed in Phase 2
native_unit_of_measurement: kW
device_class: power
state_class: measurement
native_value: first forecast value in kW
extra_state_attributes:
  forecast: list[{time: ISO8601 local time string, value: float kW}]
  source: powersync_ml_forecaster
  status: ok|training|error|stale
  forecast_created_at: ISO8601 UTC time string
```

The custom integration may read the latest forecast from `/share/powersync_ml_forecaster/latest_forecast.json` if the ML trainer remains in AppDaemon. That separates entity registration from model training and avoids rewriting the whole forecaster as a native integration.

---

## 9. Training data design

### 9.1 Time grid

Use a UTC indexed base frame.

Recommended live forecast interval:

```yaml
interval_minutes: 5
```

Recommended training construction interval:

```yaml
training_origin_interval_minutes: 60
training_lead_interval_minutes: 15
```

This keeps the origin by lead training set much smaller than the live output grid.

### 9.1.1 Published interval versus training lead interval

Default rule:

```yaml
forecast_output_mode: predict_training_grid_then_interpolate
```

The model predicts at `training_lead_interval_minutes`, then the published HAFO forecast is linearly interpolated to `interval_minutes` if PowerSync needs finer resolution.

Do not ask a tree model to predict lead values it never saw during training unless `training_lead_interval_minutes == interval_minutes`.

Allowed modes:

| Mode | Behaviour | When to use |
|---|---|---|
| `same_as_training` | Published forecast interval equals training lead interval | Lowest risk and best for hourly PowerSync inputs |
| `predict_training_grid_then_interpolate` | Predict at training lead interval, interpolate to published interval | Recommended default for 15 minute training and 5 minute output |
| `train_at_output_interval` | Train and publish at the same fine interval | Use only on hardware that can handle the row count |


### 9.2 Candidate row count

With 90 days of history:

```text
Origins every 60 minutes: approximately 2160 origins
Leads every 15 minutes across 48h: 192 leads
Candidate rows: approximately 414,720
```

This is acceptable on NUC class hardware but may be heavy on a Raspberry Pi class host once features are added.

### 9.3 Low power hardware profile

For low power hardware, use:

```yaml
training_origin_interval_minutes: 60
training_lead_interval_minutes: 60
max_training_rows: 50000
```

or:

```yaml
interval_minutes: 60
training_origin_interval_minutes: 90
training_lead_interval_minutes: 90
max_training_rows: 50000
```

The live published forecast can still be 5 minute resolution if the host can afford prediction cost, but training should be coarser if memory pressure appears.

### 9.4 Chunked construction requirement

Do not build a large fully featured origin by lead cross product in one unbounded step.

Required approach:

1. Build and cache origin level features once per origin.
2. Build lead level calendar, holiday, weather, price and lag features in chunks.
3. Broadcast origin level features across lead rows.
4. Append chunk rows to a list or temporary frame.
5. Drop intermediate objects after each chunk.
6. Apply subsampling before fitting if row count exceeds `max_training_rows`.

Recommended chunk size:

```yaml
training_origin_chunk_size: 168
```

This is roughly one week of hourly origins per chunk.

---

## 10. Feature specification

### 10.1 Required features

```text
lead_minutes
lead_bucket
hour_sin
hour_cos
dow_sin
dow_cos
month_sin
month_cos
doy_sin
doy_cos
is_weekend
is_public_holiday
day_before_public_holiday
lag_24h_target_kw, only where valid
lag_48h_target_kw, only where valid
lag_168h_target_kw, only where valid
origin_rolling_mean_1h_kw
origin_rolling_mean_6h_kw
origin_rolling_mean_24h_kw
origin_rolling_mean_168h_kw
origin_rolling_std_24h_kw
origin_rolling_std_168h_kw
origin_load_point_24h_before_origin_kw
origin_load_point_168h_before_origin_kw
```

### 10.2 Weather features

Weather features are target time forecast features, not actual observed future conditions.

Use the forecast for the target time that would have been available at the origin time where practical. If exact forecast issue time reconstruction is not available, use Open Meteo Historical Forecast API for training and live Forecast API for deployment. Do not use Archive API as the default training source.

Recommended weather fields:

```text
wx_temperature_2m
wx_apparent_temperature
wx_relative_humidity_2m
wx_cloud_cover
wx_precipitation
wx_wind_speed_10m
wx_is_day
```

Publish diagnostics:

```text
weather_source: historical_forecast_api|forecast_api|disabled|error
weather_points
weather_last_error
```

### 10.3 Price features

Default training mode:

```yaml
price_training_mode: origin_current_price_only
```

The training row may use the import and feed in price known at `origin_time`. It must not silently use the realised price at `target_time` unless that price was genuinely known at `origin_time`.

Preferred mode, if enough snapshots exist:

```yaml
price_training_mode: stored_forecast_snapshots
```

This uses previously stored Amber forecast attributes captured at forecast origins. It best matches live deployment.

Allowed but caveated mode:

```yaml
price_training_mode: historical_target_price_proxy
```

This may be useful for testing, but it must publish a diagnostic caveat because it can leak target time information.

For deployment, parse Amber forecast attributes where available. If parsing fails, fallback to last known price only if this is explicitly labelled.

Publish diagnostics:

```text
import_price_forecast_source: parsed|origin_current_price|stored_snapshot|historical_target_price_proxy|fallback_last_value|disabled|error
feed_in_price_forecast_source: parsed|origin_current_price|stored_snapshot|historical_target_price_proxy|fallback_last_value|disabled|error
import_price_forecast_points
feed_in_price_forecast_points
price_training_mode
price_last_error
price_caveat
```

### 10.4 Optional contextual sensors

Optional sensors must be declared explicitly.

Allowed configuration:

```yaml
additional_feature_sensors:
  - entity_id: binary_sensor.someone_home
    name: someone_home
    unit: bool
    feature_mode: origin_value
  - entity_id: sensor.indoor_temperature
    name: indoor_temperature
    unit: numeric
    feature_mode: origin_value
```

Allowed generated features:

```text
origin_someone_home
origin_someone_home_is_origin_value
origin_indoor_temperature
origin_indoor_temperature_is_origin_value
```

Forbidden generated features:

```text
someone_home_at_target_time
indoor_temperature_at_target_time
indoor_temperature_is_stale_future_value, unless identical stale rows exist in training
```

### 10.5 Features excluded by default

Exclude these unless a genuine future trajectory is available:

```text
solar_power
battery_soc
battery_charge_power
battery_discharge_power
instantaneous_grid_import
instantaneous_grid_export
```

These are system response variables, not stable exogenous predictors.

---

## 11. Model

Default model:

```text
sklearn.ensemble.HistGradientBoostingRegressor
loss = absolute_error
```

Recommended defaults:

```yaml
model_max_iter: 300
model_learning_rate: 0.05
model_max_leaf_nodes: 31
model_min_samples_leaf: 30
```

Rationale:

- Handles non linear calendar and weather interactions.
- Tolerates `NaN` values in invalid lag features.
- Fast enough for AppDaemon on modest hardware.
- Absolute error loss is less sensitive to spikes.

---

## 12. Baseline comparison models

### 12.1 Valid persistence baselines

Calculate persistence errors only when the baseline is valid at the forecast origin.

```text
persistence_24h = baseline_load[target_time - 24h]
valid only where lead_minutes <= 24h

persistence_48h = baseline_load[target_time - 48h]
valid only where lead_minutes <= 48h

persistence_168h = baseline_load[target_time - 168h]
valid across all 48h leads
```

For lead buckets beyond 24 hours, `persistence_24h_mae` must be `null`, `N/A`, or omitted. Do not fill it with future data.

### 12.2 Published baseline metrics

Publish:

```text
model_mae_by_lead_bucket
persistence_24h_mae_by_lead_bucket
persistence_48h_mae_by_lead_bucket
persistence_168h_mae_by_lead_bucket
best_valid_persistence_mae_by_lead_bucket
model_vs_best_persistence_delta_pct
```

Definition:

```text
model_vs_best_persistence_delta_pct = 100 * (model_mae - best_valid_persistence_mae) / best_valid_persistence_mae
```

A negative value means the model is better.

### 12.3 Deployment gates

Overall quality gate:

```text
model_vs_best_persistence_delta_pct <= -5
```

EV artefact gate:

```text
On validation days following EV charging:
  model MAE during the corresponding previous charge window beats best valid persistence by >= 20 percent
  OR forecasted false repeated EV load kWh is reduced by >= 50 percent.
```

The model is suitable for PowerSync handoff if it passes the overall quality gate or the EV artefact gate. If it passes only the EV artefact gate, publish:

```text
model_acceptance_reason: ev_artefact_gate
model_quality_caveat: aggregate MAE did not beat persistence by the configured margin
```

If neither gate passes, keep the sensor in `test_mode` and do not hand it to PowerSync unless the user explicitly overrides the gate.

---

## 13. Validation and refit semantics

### 13.1 Walk forward validation

Use forecast origin simulation. Do not use ordinary one step ahead validation as the primary metric.

Recommended split:

```yaml
validation_days: 14
```

Process:

```text
1. Reserve recent origins inside the validation period.
2. Train model on origins before the validation period.
3. Predict each validation origin across the configured horizon.
4. Compare predicted baseline load at target times with actual baseline load.
5. Aggregate MAE by lead bucket.
6. Compare against valid persistence baselines only.
```

### 13.2 Lead buckets

Publish MAE for:

```text
0_to_6h
6_to_12h
12_to_24h
24_to_36h
36_to_48h
overall
```


### 13.2.1 EV artefact validation

Identify validation origins where an EV charging block occurred in the previous 24 to 48 hours. For each such origin:

1. Identify the previous EV charging window from `ev_power_kw >= ev_exclusion_threshold_kw`.
2. Compare the model forecast for the corresponding future local time window against actual baseline load.
3. Compare the same window against valid persistence baselines.
4. Calculate false repeated EV load:

```text
false_ev_kwh = max(forecast_kw - baseline_actual_kw, 0) integrated over the matching charge window
```

Publish:

```text
ev_artefact_validation_days
ev_artefact_model_mae
ev_artefact_best_persistence_mae
ev_artefact_mae_delta_pct
ev_false_repeated_kwh_model
ev_false_repeated_kwh_persistence
ev_false_repeated_kwh_reduction_pct
ev_artefact_gate_passed
```

### 13.3 Final refit for deployment

After validation metrics are calculated, train the deployed model again on **all eligible training rows**, including the validation period.

Publish both timestamps:

```text
validation_trained_until
final_refit_trained_until
```

The model used for live prediction must be the final refit model, not the validation fold model.

### 13.4 Live forecast validation

Store each published forecast snapshot so real error can be calculated later.

Snapshot directory:

```text
/share/powersync_ml_forecaster/
```

Snapshot file:

```text
/share/powersync_ml_forecaster/forecast_snapshots.jsonl
```

This path is outside the AppDaemon code directory and should survive add on restarts and updates.

Each line:

```json
{"created_at":"2026-05-19T10:00:00+00:00","entity_id":"sensor.hafo_power_sync_home_load_forecast","forecast":[{"time":"2026-05-19T10:05:00+09:30","value":1.23}]}
```

Retention:

```yaml
snapshot_retention_days: 30
```

---

## 14. Training row subsampling

If candidate rows exceed `max_training_rows`, subsample deterministically with a fixed seed.

Required strategy:

1. Assign every candidate row to a lead bucket.
2. Sample within each lead bucket.
3. Prefer uniform bucket allocation so long lead performance is not drowned by short lead rows.
4. If a bucket has fewer rows than its allocation, keep all rows and redistribute unused allocation to the other buckets.
5. Preserve chronological validation rows. Subsampling applies to model fitting rows, not to validation target rows.

Default:

```yaml
max_training_rows: 250000
subsample_strategy: uniform_by_lead_bucket
subsample_random_seed: 42
```

For low power hardware:

```yaml
max_training_rows: 50000
training_lead_interval_minutes: 60
```

---

## 15. Weather data

### 15.1 Historical training weather

Use explicit weather modes.

Recommended default:

```yaml
weather_training_mode: historical_forecast_proxy
```

Allowed modes:

| Mode | Endpoint | Meaning | Diagnostic |
|---|---|---|---|
| `origin_faithful_previous_runs` | Stored previous forecast snapshots, if implemented | Uses the forecast captured at each origin | `weather_source: origin_faithful_previous_runs` |
| `historical_forecast_proxy` | Open Meteo Historical Forecast API | Uses archived forecast model outputs as the practical training proxy | `weather_source: historical_forecast_api` |
| `archive_api_fallback` | Open Meteo Archive or Historical Weather API | Uses actual/reanalysis style weather, not forecast noise | `weather_source: archive_api_fallback` and caveat required |
| `disabled` | none | No weather features | `weather_source: disabled` |

Default endpoint for `historical_forecast_proxy`:

```text
https://historical-forecast-api.open-meteo.com/v1/forecast
```

Default endpoint for live forecasts:

```text
https://api.open-meteo.com/v1/forecast
```

Do not use Archive API as the default. It may be used only as an explicit fallback with this diagnostic flag:

```text
weather_source: archive_api_fallback
weather_source_caveat: archive weather is actual/reanalysis style data and may overstate live model performance
```

### 15.2 Live future weather

Use Open Meteo Forecast API for live target time weather features.

### 15.3 Weather failure behaviour

If weather fetch fails:

1. Publish `weather_source: error`.
2. Keep the forecast running if enough non weather features exist.
3. Publish `weather_last_error`.
4. Do not silently fill all weather features with zero.

---

## 16. Published sensor schema

Example:

```yaml
state: "1.234"
attributes:
  forecast:
    - time: "2026-05-19T10:05:00+09:30"
      value: 1.234
    - time: "2026-05-19T10:10:00+09:30"
      value: 1.198

  unit_of_measurement: kW
  device_class: power
  state_class: measurement
  friendly_name: HAFO Load Forecast
  source: powersync_ml_forecaster
  status: ok

  forecast_id: "20260519T003000Z"
  forecast_created_at: "2026-05-19T00:30:00+00:00"
  source_entity: sensor.power_sync_home_load
  horizon_hours: 48
  interval_minutes: 5

  model: hist_gradient_boosting
  model_mae_by_lead_bucket:
    0_to_6h: 0.24
    6_to_12h: 0.28
    12_to_24h: 0.33
    24_to_36h: 0.41
    36_to_48h: 0.46
    overall: 0.35
  persistence_24h_mae_by_lead_bucket:
    0_to_6h: 0.30
    6_to_12h: 0.34
    12_to_24h: 0.40
    24_to_36h: null
    36_to_48h: null
    overall_valid_only: 0.35
  persistence_48h_mae_by_lead_bucket:
    0_to_6h: 0.35
    6_to_12h: 0.38
    12_to_24h: 0.42
    24_to_36h: 0.49
    36_to_48h: 0.52
    overall_valid_only: 0.43
  persistence_168h_mae_by_lead_bucket:
    0_to_6h: 0.36
    6_to_12h: 0.39
    12_to_24h: 0.44
    24_to_36h: 0.51
    36_to_48h: 0.55
    overall: 0.45
  best_valid_persistence_mae_by_lead_bucket:
    0_to_6h: 0.30
    6_to_12h: 0.34
    12_to_24h: 0.40
    24_to_36h: 0.49
    36_to_48h: 0.52
    overall: 0.42
  model_vs_best_persistence_delta_pct: -16.7
  model_acceptance_reason: overall_quality_gate
  ev_artefact_gate_passed: true
  ev_false_repeated_kwh_reduction_pct: 72.4

  validation_trained_until: "2026-05-05T00:00:00+00:00"
  final_refit_trained_until: "2026-05-19T00:00:00+00:00"

  ev_exclusion_enabled: true
  ev_excluded_intervals: 238
  ev_excluded_kwh: 52.4
  ev_exclusion_threshold_kw: 0.5

  weather_source: historical_forecast_api
  weather_training_mode: historical_forecast_proxy
  import_price_forecast_source: parsed
  feed_in_price_forecast_source: parsed
  price_training_mode: origin_current_price_only

  snapshot_path: /share/powersync_ml_forecaster/forecast_snapshots.jsonl
```

---

## 17. AppDaemon installation

### 17.1 App directory

Common AppDaemon paths:

| Installation type | AppDaemon app directory |
|---|---|
| HA OS add on | `/addon_configs/a0d7b954_appdaemon/apps/` |
| Older HA add on layout | `/config/appdaemon/apps/` |
| Container | mapped AppDaemon `apps/` volume |
| Core | AppDaemon config `apps/` directory |

App code layout:

```text
apps/
  powersync_ml_forecaster/
    powersync_ml_forecaster.py
  apps.yaml
```

### 17.2 Persistent data directory

Create:

```bash
mkdir -p /share/powersync_ml_forecaster
```

Data layout:

```text
/share/powersync_ml_forecaster/
  forecast_snapshots.jsonl
  model_cache.joblib       optional
  training_cache.parquet   optional
```

Do not write snapshots under `apps/`.


### 17.2.1 Startup model cache behaviour

On AppDaemon startup:

```text
if model_cache.joblib exists and age <= model_cache_max_age_hours:
    load cached model, feature list and diagnostics
    publish status: ok_cache_loaded
    generate a fresh forecast using current origin data
    schedule full retrain
else:
    publish status: training
    train and validate before the first live ML forecast
```

If cache loading fails, publish `status: training` and continue with a fresh training run. Do not publish stale predictions as if they were fresh.

Recommended defaults:

```yaml
model_cache_enabled: true
model_cache_max_age_hours: 24
```

### 17.3 Dependencies

Prefer compatible ranges unless exact versions are tested on the target AppDaemon Python version.

```yaml
python_packages:
  - "numpy>=1.26,<2.0"
  - "pandas>=2.2,<3.0"
  - "scikit-learn>=1.4,<2.0"
  - "requests>=2.31,<3.0"
  - "holidays>=0.45,<1.0"
```

Restart the AppDaemon add on after changing packages.

---

## 18. apps.yaml template

```yaml
powersync_ml_forecaster:
  module: powersync_ml_forecaster.powersync_ml_forecaster
  class: PowerSyncMLForecaster

  # Required: confirmed during discovery
  load_sensor: sensor.power_sync_home_load
  load_unit_override: null

  # Required: confirmed during PowerSync compatibility test
  hafo_entity_id: sensor.hafo_power_sync_home_load_forecast
  friendly_name: HAFO Load Forecast

  # Location and calendar
  latitude: -35.07
  longitude: 138.54
  timezone: Australia/Adelaide
  public_holiday_region: SA

  # EV exclusion
  ev_power_sensor: sensor.ocular_charger_power
  ev_power_unit_override: null
  ev_exclusion_threshold_kw: 0.5

  # Price features
  import_price_sensor: sensor.amber_general_price
  feed_in_price_sensor: sensor.amber_feed_in_price
  price_unit: auto
  price_training_mode: origin_current_price_only
  price_fallback_mode: fallback_last_value

  # Optional contextual sensors. Remove examples unless they exist.
  additional_feature_sensors: []
  # Example:
  # additional_feature_sensors:
  #   - entity_id: binary_sensor.someone_home
  #     name: someone_home
  #     unit: bool
  #     feature_mode: origin_value

  # Forecast output
  horizon_hours: 48
  interval_minutes: 5
  update_minutes: 15
  forecast_output_mode: predict_training_grid_then_interpolate

  # Training
  days_back: 90
  minimum_training_days: 45
  retrain_minutes: 360
  validation_days: 14
  training_origin_interval_minutes: 60
  training_lead_interval_minutes: 15
  training_origin_chunk_size: 168
  max_training_rows: 250000
  subsample_strategy: uniform_by_lead_bucket
  subsample_random_seed: 42

  # Low power override, if needed
  # training_lead_interval_minutes: 60
  # max_training_rows: 50000

  # Model
  model_max_iter: 300
  model_learning_rate: 0.05
  model_max_leaf_nodes: 31
  model_min_samples_leaf: 30

  # Weather source
  weather_training_mode: historical_forecast_proxy

  # Operational safety
  history_fetch_chunk_days: 7
  max_total_history_fetch_seconds: 600
  max_single_history_chunk_seconds: 90
  max_training_frame_memory_mb: 512
  data_dir: /share/powersync_ml_forecaster
  snapshot_retention_days: 30
  model_cache_enabled: true
  model_cache_max_age_hours: 24

  # Quality gates
  required_model_improvement_pct: 5
  ev_artefact_required_mae_improvement_pct: 20
  ev_artefact_required_false_kwh_reduction_pct: 50
```

---

## 19. Reference implementation contract

The implementation must include these functions or equivalent methods. Names may differ, but behaviour must match.

```python
@dataclass
class TrainingRow:
    origin_time: pd.Timestamp
    target_time: pd.Timestamp
    lead_minutes: int
    target_kw: float
```

Required functions:

```python
build_base_history_frame() -> pd.DataFrame
construct_baseline_load(raw_load_kw, ev_power_kw, threshold_kw) -> pd.Series
build_origin_features(base_df, origin_times) -> pd.DataFrame
build_target_features(origin_time, target_times, base_df, weather, prices) -> pd.DataFrame
is_lag_valid(lead_minutes, lag_hours) -> bool
lookup_valid_lag(base_df, target_time, origin_time, lag_hours) -> float | np.nan
build_training_rows_chunked(base_df, weather, prices, origins, leads) -> pd.DataFrame
subsample_training_rows(df, max_rows, strategy="uniform_by_lead_bucket") -> pd.DataFrame
walk_forward_validate(df, validation_days) -> dict
fit_final_model_on_all_rows(df) -> model
build_live_forecast_features(now, model_features) -> pd.DataFrame
publish_hafo_sensor(forecast, diagnostics) -> None
write_forecast_snapshot(forecast, data_dir) -> None
```

### 19.1 Lag validity reference code

```python
import numpy as np
import pandas as pd


def is_lag_valid(lead_minutes: int, lag_hours: int) -> bool:
    return lag_hours * 60 >= lead_minutes


def lookup_valid_lag(target: pd.Series, origin_time: pd.Timestamp, target_time: pd.Timestamp, lag_hours: int):
    lead_minutes = int((target_time - origin_time).total_seconds() // 60)
    if not is_lag_valid(lead_minutes, lag_hours):
        return np.nan
    lag_time = target_time - pd.Timedelta(hours=lag_hours)
    if lag_time > origin_time:
        return np.nan
    try:
        return float(target.loc[lag_time])
    except KeyError:
        return float(target.reindex([lag_time], method="nearest", tolerance=pd.Timedelta(minutes=3)).iloc[0])
```

### 19.2 Optional sensor origin feature reference code

```python

def build_optional_origin_features(base_df: pd.DataFrame, origin_time: pd.Timestamp, sensors: list[dict]) -> dict:
    features = {}
    for sensor in sensors:
        name = sensor["name"]
        col = sensor["column"]
        mode = sensor.get("feature_mode", "origin_value")
        if mode != "origin_value":
            raise ValueError(f"Unsupported feature_mode for {name}: {mode}")
        value = base_df[col].reindex([origin_time], method="ffill", tolerance=pd.Timedelta(minutes=60)).iloc[0]
        features[f"origin_{name}"] = value
        features[f"origin_{name}_is_origin_value"] = 1.0
    return features
```

### 19.3 Stratified subsampling reference code

```python

def subsample_uniform_by_lead_bucket(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df

    buckets = sorted(df["lead_bucket"].dropna().unique())
    per_bucket = max_rows // len(buckets)
    remainder = max_rows % len(buckets)

    sampled_parts = []
    spare = 0
    rng_seed = seed

    for i, bucket in enumerate(buckets):
        part = df[df["lead_bucket"] == bucket]
        target_n = per_bucket + (1 if i < remainder else 0) + spare
        if len(part) <= target_n:
            sampled_parts.append(part)
            spare = target_n - len(part)
        else:
            sampled_parts.append(part.sample(n=target_n, random_state=rng_seed))
            spare = 0
        rng_seed += 1

    out = pd.concat(sampled_parts).sort_values(["origin_time", "target_time"])
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed).sort_values(["origin_time", "target_time"])
    return out
```

### 19.4 Walk forward validation and final refit reference sequence

```python
validation_cutoff = all_rows["origin_time"].max() - pd.Timedelta(days=validation_days)
fit_rows = all_rows[all_rows["origin_time"] < validation_cutoff]
validation_rows = all_rows[all_rows["origin_time"] >= validation_cutoff]

validation_model = fit_model(fit_rows)
validation_metrics = evaluate_by_lead_bucket(validation_model, validation_rows)

# Deployment model: refit on all eligible rows after metrics are recorded.
final_model = fit_model(all_rows)
```


### 19.5 Runnable synthetic reference skeleton

This skeleton must run locally without Home Assistant. It proves the forecast origin construction, lag validity rule, lead buckets, validation/refit sequence and interpolation contract. The production AppDaemon implementation can replace the synthetic data and publishing stubs with Home Assistant calls, but should preserve the same data boundaries.

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


LEAD_BUCKETS = [
    (0, 6 * 60, "0_to_6h"),
    (6 * 60, 12 * 60, "6_to_12h"),
    (12 * 60, 24 * 60, "12_to_24h"),
    (24 * 60, 36 * 60, "24_to_36h"),
    (36 * 60, 48 * 60 + 1, "36_to_48h"),
]


def lead_bucket(lead_minutes: int) -> str:
    for lo, hi, name in LEAD_BUCKETS:
        if lo <= lead_minutes < hi:
            return name
    return "out_of_range"


def is_lag_valid(lead_minutes: int, lag_hours: int) -> bool:
    return lag_hours * 60 >= lead_minutes


def make_synthetic_base_frame(days: int = 90, interval_minutes: int = 5) -> pd.DataFrame:
    idx = pd.date_range(
        "2026-01-01", periods=days * 24 * 60 // interval_minutes,
        freq=f"{interval_minutes}min", tz="UTC"
    )
    local = idx.tz_convert("Australia/Adelaide")
    hour = local.hour + local.minute / 60
    dow = local.dayofweek

    daily = 0.7 + 0.35 * np.sin(2 * np.pi * (hour - 7) / 24) ** 2
    evening = np.where((hour >= 17) & (hour <= 22), 0.8, 0)
    weekend = np.where(dow >= 5, 0.25, 0)
    weather = 24 + 8 * np.sin(2 * np.pi * (local.dayofyear - 15) / 365)
    heat_load = np.clip(weather - 26, 0, None) * 0.12
    noise = np.random.default_rng(42).normal(0, 0.08, len(idx))

    ev_kw = np.zeros(len(idx))
    # Repeat a few EV sessions so the EV artefact gate has data.
    for day in range(20, days, 11):
        start = day * 24 * 60 // interval_minutes + 19 * 60 // interval_minutes
        ev_kw[start:start + 4 * 60 // interval_minutes] = 6.8

    baseline = np.clip(daily + evening + weekend + heat_load + noise, 0.1, None)
    raw_load = baseline + ev_kw

    return pd.DataFrame({
        "raw_load_kw": raw_load,
        "ev_power_kw": ev_kw,
        "baseline_load_kw": baseline,
        "wx_temperature_2m": weather,
        "import_price": 0.25 + 0.2 * ((hour >= 17) & (hour <= 21)),
        "feed_in_price": 0.05 - 0.1 * ((hour >= 11) & (hour <= 14)),
    }, index=idx)


def add_calendar_features(target_times: pd.DatetimeIndex, tz: str = "Australia/Adelaide") -> pd.DataFrame:
    local = target_times.tz_convert(tz)
    hour = local.hour + local.minute / 60
    dow = local.dayofweek
    month = local.month
    doy = local.dayofyear
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "doy_sin": np.sin(2 * np.pi * doy / 365),
        "doy_cos": np.cos(2 * np.pi * doy / 365),
        "is_weekend": (dow >= 5).astype(float),
    }, index=target_times)


def lookup_nearest(series: pd.Series, when: pd.Timestamp, tolerance_minutes: int = 3) -> float:
    out = series.reindex([when], method="nearest", tolerance=pd.Timedelta(minutes=tolerance_minutes))
    return float(out.iloc[0]) if len(out) and not pd.isna(out.iloc[0]) else np.nan


def build_rows(
    base: pd.DataFrame,
    origins: pd.DatetimeIndex,
    horizon_hours: int = 48,
    training_lead_interval_minutes: int = 60,
) -> pd.DataFrame:
    target = base["baseline_load_kw"]
    rows = []
    leads = range(training_lead_interval_minutes, horizon_hours * 60 + 1, training_lead_interval_minutes)

    for origin in origins:
        hist = target.loc[:origin]
        if len(hist) < 168 * 12:
            continue
        origin_features = {
            "origin_rolling_mean_1h_kw": hist.tail(12).mean(),
            "origin_rolling_mean_6h_kw": hist.tail(72).mean(),
            "origin_rolling_mean_24h_kw": hist.tail(288).mean(),
            "origin_rolling_mean_168h_kw": hist.tail(2016).mean(),
            "origin_rolling_std_24h_kw": hist.tail(288).std(),
            "origin_rolling_std_168h_kw": hist.tail(2016).std(),
            "origin_load_point_24h_before_origin_kw": lookup_nearest(target, origin - pd.Timedelta(hours=24)),
            "origin_load_point_168h_before_origin_kw": lookup_nearest(target, origin - pd.Timedelta(hours=168)),
            # Price default: origin current price only, not target realised price.
            "origin_import_price": lookup_nearest(base["import_price"], origin),
            "origin_feed_in_price": lookup_nearest(base["feed_in_price"], origin),
        }
        for lead in leads:
            target_time = origin + pd.Timedelta(minutes=lead)
            if target_time not in base.index:
                continue
            row = {
                "origin_time": origin,
                "target_time": target_time,
                "lead_minutes": lead,
                "lead_bucket": lead_bucket(lead),
                "target_kw": float(base.loc[target_time, "baseline_load_kw"]),
                # Synthetic weather proxy. Production should use historical forecast proxy.
                "wx_temperature_2m": float(base.loc[target_time, "wx_temperature_2m"]),
            }
            row.update(origin_features)
            for lag_h in (24, 48, 168):
                col = f"lag_{lag_h}h_target_kw"
                row[col] = lookup_nearest(target, target_time - pd.Timedelta(hours=lag_h)) if is_lag_valid(lead, lag_h) else np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    cal = add_calendar_features(pd.DatetimeIndex(df["target_time"]))
    cal.index = df.index
    return pd.concat([df, cal], axis=1)


def fit_model(rows: pd.DataFrame):
    exclude = {"origin_time", "target_time", "target_kw", "lead_bucket"}
    feature_cols = [c for c in rows.columns if c not in exclude]
    model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=120, random_state=42)
    model.fit(rows[feature_cols], rows["target_kw"])
    return model, feature_cols


def evaluate_by_bucket(model, feature_cols, rows: pd.DataFrame) -> dict:
    pred = model.predict(rows[feature_cols])
    out = {}
    rows = rows.copy()
    rows["pred"] = pred
    for bucket, part in rows.groupby("lead_bucket"):
        out[bucket] = round(float(mean_absolute_error(part["target_kw"], part["pred"])), 4)
    out["overall"] = round(float(mean_absolute_error(rows["target_kw"], rows["pred"])), 4)
    return out


def main():
    base = make_synthetic_base_frame()
    origins = pd.date_range(base.index[0] + pd.Timedelta(days=14), base.index[-1] - pd.Timedelta(hours=48), freq="60min", tz="UTC")
    rows = build_rows(base, origins, training_lead_interval_minutes=60)
    cutoff = rows["origin_time"].max() - pd.Timedelta(days=14)
    fit_rows = rows[rows["origin_time"] < cutoff]
    val_rows = rows[rows["origin_time"] >= cutoff]

    validation_model, feature_cols = fit_model(fit_rows)
    metrics = evaluate_by_bucket(validation_model, feature_cols, val_rows)
    print("validation_mae_by_bucket", metrics)

    final_model, final_features = fit_model(rows)
    print("final_refit_rows", len(rows), "features", len(final_features))


if __name__ == "__main__":
    main()
```

Acceptance for the skeleton:

```text
python synthetic_forecaster_reference.py
```

must print validation MAE by bucket and a final refit row count without exceptions.

---

## 20. Algorithm outline

1. Load configuration.
2. Validate required entity IDs.
3. Fetch Home Assistant history in chunks.
4. Resample all history to a UTC time grid.
5. Convert all power values to kW.
6. Create `raw_load_kw`.
7. Create `baseline_load_kw` by subtracting EV charging intervals.
8. Fetch historical forecast weather for the training period.
9. Fetch live future weather forecast.
10. Parse Amber price forecast attributes if available.
11. Select eligible forecast origins.
12. Build origin features once per origin.
13. Build lead rows in chunks.
14. Add target time calendar and holiday features.
15. Add only valid lag features.
16. Add origin rolling features.
17. Add target time forecast weather and price features.
18. Add optional contextual sensors as origin features only.
19. Drop rows with missing target.
20. Subsample by lead bucket if needed.
21. Run walk forward validation.
22. Refit final model on all eligible rows.
23. Generate live 48 hour forecast.
24. Publish HAFO compatible sensor.
25. Write forecast snapshot to `/share/powersync_ml_forecaster/forecast_snapshots.jsonl`.
26. Prune old snapshots.

---

## 21. Validation procedures

### 21.1 Sensor schema test

```python
attrs = state["attributes"]
assert isinstance(float(state["state"]), float)
assert attrs["status"] == "ok"
assert isinstance(attrs["forecast"], list)
assert {"time", "value"}.issubset(attrs["forecast"][0])
assert attrs["unit_of_measurement"] == "kW"
```

### 21.2 PowerSync handoff test

Compare the ML forecast with PowerSync's LP load forecast.

Pass condition where comparable arrays are available:

```text
correlation >= 0.90
median_absolute_ratio between 0.80 and 1.25
```

If no comparable arrays are available, inspect the PowerSync LP forecast chart and confirm it follows the ML sensor's shape and magnitude.

### 21.3 EV exclusion test

After at least one EV charging session exists in the training period:

```python
attrs = state["attributes"]
assert attrs.get("ev_exclusion_enabled") is True
assert attrs.get("ev_excluded_intervals", 0) > 0
assert attrs.get("ev_excluded_kwh", 0) > 0
```

Then check sampled training diagnostics or logs to confirm:

```text
raw_load_kw > baseline_load_kw during EV charging intervals
```

### 21.4 Model quality test

Pass condition:

```text
model_vs_best_persistence_delta_pct <= -5
```

If the result is between `0` and `-5`, mark the model as marginal and keep it in test mode unless the user explicitly accepts the EV artefact improvement.

If the result is positive, the model is worse than persistence. Do not hand it to PowerSync without tuning.

---

## 22. Operational guidance

### 22.1 Recommended initial settings

```yaml
minimum_training_days: 45
days_back: 90
training_origin_interval_minutes: 60
training_lead_interval_minutes: 15
max_training_rows: 250000
retrain_minutes: 360
update_minutes: 15
```

### 22.2 Low power hardware settings

```yaml
training_origin_interval_minutes: 60
training_lead_interval_minutes: 60
max_training_rows: 50000
retrain_minutes: 720
```

### 22.3 When to reduce complexity

Reduce complexity if:

```text
training takes longer than 5 minutes
AppDaemon memory grows above safe levels
Home Assistant recorder queries become slow
forecast refresh takes longer than the update interval
```

First reductions:

1. Increase `training_lead_interval_minutes` to 60.
2. Reduce `max_training_rows` to 50000.
3. Reduce `days_back` to 60.
4. Remove optional contextual sensors.
5. Disable feature importance if implemented.


### 22.3.1 Hard safety aborts

Abort the training cycle and publish `status: error` if any of these limits are exceeded:

```yaml
max_total_history_fetch_seconds: 600
max_single_history_chunk_seconds: 90
max_training_frame_memory_mb: 512
```

Error message must include a concrete remediation:

```text
Reduce days_back, increase training_lead_interval_minutes, reduce optional sensors, or move AppDaemon to stronger hardware.
```

Do not keep querying the recorder indefinitely.

---

## 23. Rollback plan

Rollback must be simple:

1. Disable the AppDaemon app block in `apps.yaml`.
2. Restart AppDaemon.
3. Reconfigure PowerSync back to HAFO or its internal persistence forecast.
4. Leave `/share/powersync_ml_forecaster/forecast_snapshots.jsonl` in place until troubleshooting is complete.

Do not delete historical snapshots until live error analysis is no longer required.

---

## 24. Acceptance checklist

```text
[ ] Entity discovery completed.
[ ] PowerSync mock sensor compatibility passed.
[ ] Output entity name confirmed.
[ ] AppDaemon app installed.
[ ] Data directory created under /share/powersync_ml_forecaster.
[ ] Dependencies installed.
[ ] First training run completed.
[ ] Validation metrics published by lead bucket.
[ ] Final model refit after validation.
[ ] persistence_24h omitted or null beyond 24h lead.
[ ] Optional sensors are origin features only.
[ ] Weather training mode explicitly set.
[ ] Historical Forecast API proxy used by default, or fallback caveat published.
[ ] Price training mode explicitly set and does not silently use target time realised prices.
[ ] Forecast output interval rule confirmed.
[ ] Forecast sensor status is ok.
[ ] Forecast attribute contains time/value entries.
[ ] Forecast snapshots are being written.
[ ] EV exclusion diagnostics show expected behaviour.
[ ] PowerSync LP forecast follows ML forecast with correlation >= 0.90 where measurable.
[ ] Model beats best valid persistence by at least 5 percent, passes EV artefact gate, or is kept in test mode.
[ ] Model cache startup behaviour tested.
[ ] History fetch and memory safety limits configured.
[ ] Rollback path has been confirmed.
```

---

## 25. Developer notes

### CONFIRMED requirements

- The model must not use target time values for ordinary Home Assistant sensors.
- `persistence_24h` is invalid beyond 24 hours of lead time.
- Forecast snapshots belong under `/share`, not the AppDaemon code directory.
- The deployed model must be refit on all eligible rows after validation.

### PROBABLE implementation risk

PowerSync may require a real entity registry entry rather than accepting a state written by AppDaemon. The compatibility test is therefore a blocking gate.

### POSSIBLE future improvement

A small Home Assistant custom integration may be cleaner than AppDaemon if long term maintenance matters, because it can create a registered sensor entity and package dependencies more predictably.

### v4.2 implementation stance

This specification is intended to be handed to a coding agent. Where the implementation may choose between correctness and convenience, choose the behaviour that preserves origin time information boundaries, even if it means disabling a feature and publishing a caveat.

