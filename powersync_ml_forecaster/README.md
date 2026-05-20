# PowerSync ML Forecaster

A Home Assistant add-on that trains a machine-learning baseline household load forecaster and publishes a HAFO-compatible forecast sensor for PowerSync.

## Required configuration

Set at least:

```yaml
load_sensor: sensor.your_household_load_sensor
hafo_entity_id: sensor.hafo_your_household_load_sensor_forecast
latitude: -35.07
longitude: 138.54
timezone: Australia/Adelaide
public_holiday_region: SA
```

Recommended:

```yaml
ev_power_sensor: sensor.your_ev_charger_power
import_price_sensor: sensor.your_amber_import_price
feed_in_price_sensor: sensor.your_amber_feed_in_price
```

## First start

The add-on starts training immediately. First training can take several minutes depending on your recorder database and hardware.

Watch the add-on log for:

```text
Starting training cycle
Validation complete
Published forecast
```

Then check Home Assistant Developer Tools → States for the configured `hafo_entity_id`.

## PowerSync handoff

PowerSync must be configured to read the same entity ID that this add-on publishes. If it does not, the ML forecast may appear in Home Assistant but PowerSync will continue using HAFO or internal persistence.
