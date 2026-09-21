# Delhi Corpus → PPO/DRL Integration Guide

## Purpose

This document describes the next project stage: connecting the final Delhi
corpus, the trained GNN–Transformer forecaster, PPO, QPSO, the safety layer,
and SUMO.

PPO is the project's DRL algorithm. There is no separate second algorithm
called “DRL” to implement: DRL is the broad learning approach and PPO is the
specific method used here.

## Refined changes in this branch

- `scripts/train_ppo_delhi.py` now opens the current Delhi corpus by default:
  `artifacts/delhi_2280/`.
- It looks first for `corpus_manifest_1615.json`, the current 1,615-episode
  index. It also accepts the old `corpus_manifest.json` if someone deliberately
  runs an older corpus. This fallback does not mix the two datasets; it simply
  keeps the launcher usable with either named index.
- Ten downloaded Delhi episodes were checked against the new manifest. Their
  `episode_manifest.json`, `events.json`, `observations.csv`, and
  `labels.jsonl` follow the expected structure.
- The existing PPO unit tests passed. This confirms the PPO learning pieces,
  action choices, and training foundation still work after the manifest change.
- The forecast-window reader now accepts both the old pilot index and the
  Delhi-1615 index. It resolves Delhi scenarios through `map_records`, rather
  than expecting the old `scenario_files` field.
- PPO now accepts one versioned signal containing 12 forecast values and 6
  uncertainty values. Existing 12-value providers remain supported and are
  treated as having unknown (zero) uncertainty.
- Small tests cover the Delhi manifest shape and prove that a controlled
  forecast and uncertainty signal reaches PPO unchanged.

## Final operating loop

```text
Scenario and initial route
        ↓
QPSO creates the initial feasible delivery plan
        ↓
SUMO advances vehicles and traffic
        ↓
Visible road observations enter the frozen GNN–Transformer
        ↓
The forecaster produces traffic predictions and uncertainty
        ↓
PPO chooses the replanning scope
        ↓
QPSO replans only that selected scope
        ↓
Safety validation accepts or rejects the candidate plan
        ↓
SUMO executes the approved plan and returns the realised reward
        ↓
PPO learns from that realised reward during training
```

## The Delhi corpus

The current corpus root is `artifacts/delhi_2280/`. Its index is
`corpus_manifest_1615.json`.

The name contains `2280` because that many episodes were requested. The
manifest records 1,615 episodes that were successfully generated.

Each episode contains:

| File | Plain meaning | Used by |
| --- | --- | --- |
| `episode_manifest.json` | The episode's ID card: map, split, duration, seed, and traffic regime. | Corpus loader, PPO runner, audit logs |
| `events.json` | Road incidents and when they become visible. | SUMO, PPO event state, scope selection |
| `observations.csv` | Traffic measurements known at each time step. | GNN–Transformer input; optional causal replay checks |
| `labels.jsonl` | What actually happened later on each road. | GNN–Transformer training and evaluation only |

The corpus manifest supplies one record per episode and one record per map.
Code must locate a scenario by following:

```text
episode.map_index → manifest.map_records[map_index].scenario_path
```

## Causality rule

At a decision time, PPO and the forecaster may use only information already
visible at that time.

`labels.jsonl` is not PPO input. It contains future outcomes, so supplying it
to a live PPO decision would let the policy see the answer before making its
choice. Labels are used to train and evaluate the forecaster. PPO receives a
forecast derived from current observations, not future labels.

## What already works

- QPSO creates initial and revised route plans.
- SUMO executes vehicle movement and incidents.
- The PPO actor, critic, rollout collection, update logic, scope actions, and
  safety override are implemented.
- `scripts/train_ppo_delhi.py` runs the PPO/SUMO/QPSO training loop.
- The launcher now prefers `corpus_manifest_1615.json` and defaults to
  `artifacts/delhi_2280/`.

## What is still required for the final learned loop

1. Add a shared Delhi-corpus loader that reads the 1,615-manifest structure.
2. Update the forecaster loader to convert manifest episode records into its
   train/validation view.
3. Add a forecast provider that loads frozen forecaster weights, scaler, and
   uncertainty calibration.
4. Build rolling 12-minute windows from live SUMO observations.
5. Turn the forecaster's road-level 5/10/15-minute predictions and uncertainty
   into the fixed features consumed by PPO.
6. Pass that provider to the PPO environment and record the exact forecaster
   version beside every PPO checkpoint.

## Current forecast placeholder

PPO always expects a fixed small summary of what traffic may do next. The
current launcher does not yet load a trained GNN–Transformer or a forecast
provider. To keep the rest of the PPO/SUMO/QPSO loop runnable, it fills the
12 forecast values and 6 uncertainty values with zeroes.

Those zeroes are a blank placeholder, not a real traffic prediction and not a
failure in `observations.csv`. The run metadata calls this a
`persistence-pilot`, but the code currently does not calculate a persistence
forecast either. A real forecast-aware run starts only after frozen forecaster
weights, a scaler, and uncertainty calibration are connected here.

The new corpus reader can prepare GNN--Transformer windows from real Delhi
episodes when the required map scenario files are present. A local ten-episode
set is suitable for checking this data path, but not for valid model training:
it has no separate validation split. The training command rejects that unsafe
setup instead of reporting an overfitted result as a real evaluation.

## Local SUMO smoke-test setup

SUMO 1.27.1 is installed locally for this branch's smoke tests. Before running
the PPO launcher from a terminal, point it to SUMO for that terminal session:

```bash
export SUMO_HOME="$(python3 -c 'import sumo; print(sumo.SUMO_HOME)')"
python3 scripts/train_ppo_delhi.py --smoke-test
```

This starts the simulator without opening its visual window. `sumo-gui` is
also installed and can be used later when a visual replay is needed.

## Local ten-episode smoke result

The local sample set has ten Connaught Place training episodes. It now has an
explicit `corpus_manifest_10_smoke.json`, separate from the real 1,615-episode
manifest, plus its matching scenario file. It is useful for format and
connection checks only.

The smoke check successfully built 90 causal forecast windows and ran a real
PPO/QPSO/SUMO episode. PPO collected five transitions, performed its update,
and SUMO accepted the route state. During this check, a live-replanning bug was
repaired: PPO now passes the current commitment snapshot and planning time to
SUMO when applying a QPSO candidate. SUMO therefore validates a mid-episode
candidate against work already completed, rather than incorrectly treating
every vehicle as if it were starting from the depot.

## Smoke-test strategy without GNN training

The first ten episodes have the same observation-file identities as the final
1,615-episode corpus. A small local smoke set can therefore check that:

1. the new manifest layout is read correctly;
2. episode scenario and event files resolve correctly;
3. SUMO starts and advances;
4. PPO chooses a permitted scope;
5. QPSO receives the selected scope;
6. the safety layer handles the candidate route;
7. PPO receives a realised reward and writes a checkpoint.

This smoke test can run before GNN training. Its output proves that the
decision and simulation loop works. It does not prove that routing decisions
are forecast-aware or that the PPO policy is good.

## Acceptance checks for the final integration

- A neural forecaster artifact is explicitly supplied to PPO.
- The PPO run manifest records forecaster weights, scaler, uncertainty file,
  corpus manifest, and seed.
- Future labels never enter a PPO observation.
- Replacing the forecast with a different valid forecast can change a PPO
  decision in a controlled test.
- The safety layer can still override an unsafe PPO request.
