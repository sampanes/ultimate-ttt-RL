# GUI backlog -- leftover ideas (parked 2026-06-24)

Low-priority front-end polish, deferred so we could pivot back to Phase-0 RL work.
None of these block training. Pick up when the GUI is back in focus.

## Wiring / data
- **Surface unused server fields.** `/api/status` already computes `total_games`
  and `best_elo` but nothing renders them -- add two stat cards (the row currently
  shows only Total/Active/Stagnant/Retired agents).
- **Smarter default tab.** On load, auto-select the tab that actually has data
  (Training if `metrics_log.jsonl` is non-empty and no league is running). Persist
  the chosen tab in `localStorage` so a refresh stays put.
- **`/api/metrics` on very long runs.** It currently returns the whole JSONL; the
  client caps the chart to 5000 points but the transfer grows unbounded. Add a
  server-side stride downsample (or a byte-range / incremental fetch) once logs get big.

## Aesthetic / polish
- **Refine the categorical `AGENT_COLORS`** for better series separation on the new
  cooler base (first two now track the accent; the rest are unchanged from the old set).
- **Optional dashboard polish** beyond the palette: spacing / typography pass if desired.

## Consolidation
- **Retire vs keep `live_metrics_plot.html`.** The standalone page is now duplicated
  by the dashboard's Training tab. Current call: keep it as the no-server fallback.
  Revisit (and maybe retire) once the tab is battle-tested on a real run.

## Phase-0-adjacent (higher value)
- **"Training health" card** tied to the Phase-0 exit check: last loss, recent slope,
  and a simple plateau / ^win-rate indicator vs a fixed anchor -- turns the loss tab from
  "a chart" into "did the plateau break?" at a glance.
