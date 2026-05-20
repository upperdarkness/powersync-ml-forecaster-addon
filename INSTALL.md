# Install options

## Option A: GitHub hosted near one-click install

1. Create a new GitHub repository, for example:

```text
powersync-ml-forecaster-addon
```

2. Upload the contents of this folder to the root of that repository.

3. Commit and push the repository. The package is already configured for:

```text
https://github.com/upperdarkness/powersync-ml-forecaster-addon
```

4. In Home Assistant, add the repository:

```text
Settings → Add-ons → Add-on Store → ⋮ → Repositories
```

5. Paste the GitHub repository URL.

6. Install **PowerSync ML Forecaster**.

## Option B: Local development install

For testing before GitHub hosting, copy this repository into an add-on repository location accessible to Home Assistant, or use a private Git repository URL.

Home Assistant add-ons are normally installed from a repository URL, so the GitHub route is the cleanest way to get a near one-click install experience.

## Configuration checklist

Required:

```yaml
load_sensor: sensor.your_household_load_sensor
hafo_entity_id: sensor.hafo_your_household_load_sensor_forecast
latitude: your_latitude
longitude: your_longitude
timezone: Australia/Adelaide
public_holiday_region: SA
```

Recommended:

```yaml
ev_power_sensor: sensor.your_ev_charger_power
import_price_sensor: sensor.your_amber_import_price
feed_in_price_sensor: sensor.your_amber_feed_in_price
```

First run:

```yaml
minimum_training_days: 21
days_back: 60
require_quality_gate: false
```

After stable operation:

```yaml
minimum_training_days: 45
days_back: 90
require_quality_gate: true
```
