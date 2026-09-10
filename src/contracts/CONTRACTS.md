
# SIH 26137 — Frozen Project Contracts

## Status

This document defines the version-1 interface contract shared by all
project modules.

Changes to field meaning, units, visibility, identifier semantics,
or serialization require an explicit schema-version change.

## Global Units

Distance: meters (m)
Time: seconds (s)
Speed: meters per second (m/s)
Demand: abstract load units unless a physical conversion is supplied

## Identity Rules

Every scenario, road edge, customer request, vehicle, episode,
event, decision, route version, forecast version, policy version,
and model pair must have a stable identifier.

A physical road edge is not a customer-to-customer routing arc.

## Visibility Rule

Policy-visible observation may contain only information that would
actually be available at the observation timestamp.

It must not contain:
- future incident times
- future incident outcomes
- unreleased orders
- hidden simulator seeds
- future labels
- future traffic truth

Environment truth is maintained separately.

## Road Representation

G_road:
physical road network consisting of junctions and directed road edges.

Requests:
delivery jobs mapped to road access points.

G_stops(t):
derived routing-cost graph containing depot/current vehicle
starts/pending stops and shortest-road-path costs at a specified time.

The optimizer decides stop order and vehicle allocation.
The path engine converts stop legs into physical road-edge sequences.
The simulator executes those sequences.
The forecaster predicts future edge conditions.

## Routing Rule

Every stop-to-stop leg must retain one internally consistent physical
road path. Its distance, travel time and congestion exposure must refer
to that same path.

## Dynamic Replanning

Executed prefixes are frozen.

Served jobs are frozen.

Service already in progress is frozen.

Onboard jobs remain with their current vehicle unless an explicit
transfer mechanism exists.

Every asynchronous candidate must carry a state version and be
revalidated against the current legal state before application.

## Forecast Contract

Forecast records include:
- issue timestamp
- edge IDs
- target timestamps/horizons
- prediction values
- validity masks
- target kind
- model version
- uncertainty/interval information where available

Speed-proxy labels and realized traversal-time labels are distinct.

## Scope Contract

The complete action space is:

KEEP
LOCAL
VEHICLE
REGIONAL
GLOBAL

Requested and executed actions are recorded separately.

Safety overrides are explicit and auditable.

## Model Release Contract

The forecasting model and PPO policy are released as a pair:

(forecast_version, policy_version)

The pair also records:

schema version
scaler version
action semantics version
reward version
QPSO version
SUMO version
training cutoff
parent pair
validation result
rollback target

## Benchmark Integrity

Original benchmark data and reference solutions are never modified.

Any road-network/traffic transformation receives a separate derived
scenario ID and provenance record.

A snapshot static optimum is never described as the optimum of the
full dynamic day.

A heuristic benchmark result is not described as certified optimal
unless the corresponding exact reference is actually certified.

## Addendum (V1.1) — Runtime Enforcement

The rules above are now backed by a runtime-validated contract layer
(`src/contracts/`, Pydantic v2, superseding the plain-dataclass V1
layer). Constructing an object that violates a rule in this document
raises `pydantic.ValidationError` immediately, rather than failing
silently or downstream. See `CHANGELOG.md` for the full diff from V1.

One clarification to the Visibility Rule, made precise during the V1.1
pass: the boundary that must never be crossed is on an event's
**reveal time**, not its **effect time**.

- `visible_events[i].revealed_at_s <= observation.observation_time_s`
  is enforced and must always hold. This is the actual leakage rule:
  an event the policy has not yet been told about must not appear in
  its observation.
- `visible_events[i].effect_start_s` may legitimately be **in the
  future** relative to `observation_time_s`. An announced-in-advance
  closure (e.g. a scheduled maintenance window revealed today, taking
  effect next week) is not a visibility violation — the policy is
  allowed to know about it before it starts. Only the fact of
  revelation is time-gated, not the fact of the event's own schedule.

Do not "fix" a future `effect_start_s` by clamping or hiding it; that
would remove legitimate, already-revealed information and produce a
weaker, less realistic policy input than the plan intends.