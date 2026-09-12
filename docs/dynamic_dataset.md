# Dynamic Traffic Dataset

## Purpose

The dynamic dataset is generated from a frozen static `Scenario` and produces
causal, time-indexed edge observations plus environment-side future truth. The
current backend is `synthetic_field_v1`; it is a deterministic pre-SUMO pilot,
not simulator ground truth.

## Data boundary

- `edge_truth.csv`: environment-side truth. It may contain future event effects.
- `observations.csv`: policy/forecast-visible measurements at each observation
  time. It does not contain event IDs, effect times, or future labels.
- `events.json`: environment-side event schedule and reveal/effect timeline.
- `episode_manifest.json`: provenance, backend, seeds, target kind, and causal
  visibility declaration.

A future label is considered available only at its declared availability time.
Later SUMO integration should replace synthetic-field truth with measured edge
states and realized traversal labels while retaining the same visibility
boundary.

## Topology families

The static generator now supports:

- `grid`: regular urban lattice;
- `irregular`: stratified, jittered street layout with non-grid connectivity;
- `radial`: spatially covered map with center-to-periphery spokes and
  circumferential redundancy;
- `hybrid`: regular urban core plus perturbed outer network and diagonal/local
  connections.

## Traffic regimes

- `normal`
- `morning_peak`
- `evening_peak`
- `corridor_surge`

Events can be `closure`, `scheduled_closure`, `incident`, or
`multi_disruption`. Event reveal time is independent of effect start so
scheduled/announced events can be represented without leaking future effects.

## Reproducibility

Use independent seeds for static scenario, traffic, incidents, windows,
optimization, and learning. The dynamic episode currently derives its event,
truth, and observation streams from the episode seed using deterministic offsets.

## Step 11 causal pilot

`src/data/causal_episodes.py` writes six split episodes:

- `observations.csv` — policy-visible 1-minute features, age, missingness
- `edge_truth.csv` — environment-only
- `trajectories.csv` — vehicle entry/exit times
- `issued_forecasts.jsonl` — forecasts stored without future labels
- `labels.jsonl` — matured `speed_proxy` and `realized_traversal` labels

A speed-proxy label is available only when that future minute was observed.
A traversal label is aligned to the entry-time bucket and available at exit.
Unused edges are missing, not 0.0.

```powershell
python -m src.data.generate_dataset causal-pilot fixtures/step3/base/scenario.json artifacts/step11_pilot
```

SUMO backend: `src/sim/sumo_causal.py` records TraCI speeds and entry/exit
times, then the same maturity code writes labels.

## Step 12 windows

```powershell
python -m src.data.generate_dataset forecast-windows artifacts/step11_pilot artifacts/step12_windows fixtures/step3/base/scenario.json
```

Inputs are `[B, 12, E, 6]` with 5/10/15-minute targets. The scaler is fit on
the training episodes only. Baselines: persistence and a temporal-only ridge
model. `inspect_example.json` shows issue time, visible window, and label times.
