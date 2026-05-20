# Changelog

## 4.3.1

- Add progress logging for larger statistics-backed training runs.

## 4.3.0

- Add Home Assistant recorder statistics fallback for older load history outside detailed recorder retention.
- Merge detailed state, short-term statistics, and long-term statistics into the baseline training frame with `history_source` diagnostics.
- Keep EV exclusion limited to detailed state history rows.

## 4.2.3

- Publish the HAFO forecast as an explicit post-training step with success/failure logs.
- Log Home Assistant state publish HTTP status and response body excerpts on failure.
- Use `powersync_ml_forecaster` as the published source attribute.
- Add synthetic coverage proving the HAFO state publish is called after training.

## 4.2.2

- Adjust validation split cutoff when the requested validation window lands beyond the latest usable training origin.
- Add synthetic coverage for late validation cutoff fallback.

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
