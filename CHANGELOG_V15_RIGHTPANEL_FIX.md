v15 Right Panel Integrated Fix (2026-01-04)
------------------------------------------
- Fix: KPI labels now read from digest_config_v15.json `title` (fallback: name/series) so the right panel never shows unnamed boxes.
- Add: KPI color (deterministic) + sparkline SVG generated from last N FRED observations (lookback in config).
- Add: delta and delta_pct computed from last two observations and displayed in UI.
- Keep: No API key required (uses fredgraph.csv endpoint).