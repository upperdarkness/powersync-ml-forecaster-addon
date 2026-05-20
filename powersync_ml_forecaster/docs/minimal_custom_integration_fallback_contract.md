# Minimal Home Assistant custom integration fallback contract

Use this only if PowerSync refuses to consume an AppDaemon-created state entity.

The fallback integration should create a registry-backed sensor entity with this contract:

```text
entity_id: sensor.hafo_power_sync_home_load_forecast
platform: sensor
native_unit_of_measurement: kW
device_class: power
state_class: measurement
native_value: first forecast value in kW
```

Extra state attributes:

```yaml
forecast:
  - time: "2026-05-19T18:05:00+09:30"
    value: 1.234
  - time: "2026-05-19T18:10:00+09:30"
    value: 1.187
source_entity: sensor.power_sync_home_load
status: ok
source: powersync_ml_forecaster_v4_2
horizon_hours: 48
publish_interval_minutes: 5
unit_of_measurement: kW
```

The integration can read the latest forecast from:

```text
/share/powersync_ml_forecaster/latest_forecast.json
```

If implementing this fallback, also modify the AppDaemon app to write that file whenever it publishes the sensor.
