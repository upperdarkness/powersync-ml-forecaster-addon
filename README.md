# PowerSync ML Forecaster Add-on Repository

This repository packages the PowerSync ML Load Forecaster v4.3.1 as a Home Assistant add-on.

It publishes a HAFO-compatible baseline household load forecast sensor for PowerSync, with EV charging excluded from the training target so one-off EV charging sessions are less likely to be repeated in the next day forecast.

## Near one-click install

Then use this Home Assistant repository link format:

```text
https://my.home-assistant.io/redirect/supervisor_addon/?addon=powersync_ml_forecaster&repository_url=https://github.com/upperdarkness/powersync-ml-forecaster-addon
```

Or add the repository manually:

1. Home Assistant → Settings → Add-ons → Add-on Store
2. Three dots → Repositories
3. Add the GitHub URL for this repository
4. Install **PowerSync ML Forecaster**
5. Configure entity IDs
6. Start the add-on

## What still needs configuration

This cannot be a literal one-click install because the add-on needs your local entity IDs:

- `load_sensor`: household load sensor in W or kW
- `hafo_entity_id`: forecast sensor name PowerSync expects
- `ev_power_sensor`: EV charger power sensor, optional but strongly recommended
- Amber import and feed-in price sensors, optional
- latitude and longitude
- timezone and public holiday region

## First-run recommendation

Start permissive, then tighten once it is stable:

```yaml
minimum_training_days: 21
days_back: 60
require_quality_gate: false
```

After it runs and PowerSync consumes the forecast:

```yaml
minimum_training_days: 45
days_back: 90
require_quality_gate: true
```

## Important PowerSync compatibility step

Before relying on the ML forecast, confirm that PowerSync reads the published entity.

The add-on publishes the sensor through the Home Assistant states API. If PowerSync requires entity registry metadata, use the custom integration fallback contract in `powersync_ml_forecaster/docs/minimal_custom_integration_fallback_contract.md`.

## Persistent files

The add-on stores runtime files in:

```text
/share/powersync_ml_forecaster/
```

This includes:

- `forecast_snapshots.jsonl`
- `model_cache.joblib`
- `latest_forecast.json`

## Included docs

- `powersync_ml_forecaster/DOCS.md`: add-on usage notes
- `powersync_ml_forecaster/docs/powersync_ml_forecaster_spec_v4_2.md`: full technical specification
- `powersync_ml_forecaster/docs/minimal_custom_integration_fallback_contract.md`: fallback entity contract
