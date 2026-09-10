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

## SUMO transition

The next simulator-backed stage should consume the same `Scenario` and episode
configuration, export a SUMO network, run TraCI, and write the same logical
observation/truth boundary. In final experiments, `target_kind` must distinguish
speed proxies from realized vehicle traversal times.
