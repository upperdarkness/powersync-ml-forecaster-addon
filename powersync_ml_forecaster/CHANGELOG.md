# Changelog

## 4.2.1

- Align prepared training indexes to configured interval boundaries.
- Allow origin training to use the nearest base row within half an interval.
- Add diagnostics for empty origin training row failures.
- Add synthetic coverage for odd-minute history starts.

## 4.2.0

- Packaged v4.2 forecaster as a standalone Home Assistant add-on.
- Added Supervisor/Core API runner, removing the AppDaemon requirement.
- Added `/share/powersync_ml_forecaster` persistent storage.
- Added add-on configuration schema.
- Included v4.2 technical specification and fallback custom integration contract.
