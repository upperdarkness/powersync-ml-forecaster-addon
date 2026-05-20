# PowerSync ML Forecaster Add-on Documentation

## Configuration fields

### `load_sensor`

The household load sensor in W or kW. This should be actual house load, not net grid import/export.

### `hafo_entity_id`

The forecast sensor entity to publish. If PowerSync follows HAFO naming conventions, use:

```text
sensor.hafo_<load_sensor_object_id>_forecast
```

Example:

```text
load_sensor: sensor.power_sync_home_load
hafo_entity_id: sensor.hafo_power_sync_home_load_forecast
```

### `ev_power_sensor`

EV charger power sensor in W or kW. When above `ev_exclusion_threshold_kw`, this power is subtracted from the load training target.

### `require_quality_gate`

When `true`, the model must beat persistence by the configured threshold before publishing as `status: ok`. For first install, use `false`; after validation, change to `true`.

## Forecast output

The published sensor has:

```yaml
state: first forecast value in kW
attributes:
  forecast:
    - time: ISO timestamp
      value: kW
  status: ok|training|error|quality_gate_failed
  validation_metrics: ...
  ev_excluded_kwh_last_training: ...
```

## Storage

Runtime data is stored in:

```text
/share/powersync_ml_forecaster/
```

## Troubleshooting

### Sensor does not appear

Check the add-on log. Usually this is a missing `load_sensor`, missing recorder history, or Python runtime error.

### Sensor appears but PowerSync ignores it

The entity ID likely does not match what PowerSync reads, or PowerSync requires a registered entity rather than a states API entity. Use the custom integration fallback contract in the docs folder.

### Forecast repeats EV charging

Check that `ev_power_sensor` is correct and reports W or kW. Confirm the published diagnostics show EV excluded intervals and kWh.

### Training is slow

Reduce:

```yaml
days_back: 30
training_lead_interval_minutes: 30
max_training_rows: 100000
```
