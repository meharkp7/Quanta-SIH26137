"""Run the exact production pilot pipeline (SUMO causal episodes -> the
same artifacts the GNN/transformer loader and PPO/DRL stack consume) on
real OSM-derived Delhi maps instead of the synthetic grid generator.

This is deliberately a thin adapter over ``src/data/training_pilot.py``,
not a parallel pipeline: every call below (``_generate_events``,
``run_sumo_causal_episode``, ``_finalize_existing_episode``,
``audit_episode``) is the same production function the synthetic pilot
uses. Only the map-generation step changes -- ``generate_osm_scenario``
takes the place of ``generate_dynamic_dataset``. Everything downstream
(SumoExporter, RouteBuilder, CausalSumoLogger, the episode manifest/label/
forecast writers, and therefore ``src.learning.loader.load_pilot_windows``
and the GNN/transformer/PPO stack that consumes it) already operates on
the generic ``Scenario`` contract and does not know or care whether the
network came from a synthetic grid or a real OSM extract.

Two real constraints this respects, both inherited from the production
pilot's contract (see ``training_pilot.py``'s ``map_fingerprint`` check):

1. Train/validation/test maps must be physically distinct road networks,
   not just different demand draws on the same network. Real OSM extracts
   satisfy this automatically as long as you export 4 different areas
   (this module will refuse to proceed otherwise, exactly like the
   synthetic pilot refuses to proceed on duplicate fingerprints).
2. ``run_sumo_causal_episode`` requires an explicit ``route_plan`` for any
   non-traffic-only episode ("the Step-3 fixture planner is never used
   here" -- see sumo_causal.py's own docstring). This module supplies one
   via the exact same ``InitialSolutionBuilder``/``RouteEvaluator`` you
   already validated the generated scenario against, rather than
   defaulting to traffic-only mode.

Nothing in this module has been executed end-to-end: this sandbox has
neither the SUMO ``netconvert``/``sumo`` binaries nor enough disk space
to install torch, so the SUMO run and the downstream GNN/PPO consumption
could not actually be exercised here. Every function call below is wired
against the real signatures in this repository (verified by reading
``sumo_causal.py``, ``causal_episodes.py``, ``dynamic_episodes.py``,
``route_plan.py``, and ``initial_solution.py`` directly), not guessed --
but you should treat the first real run on your machine as the actual
integration test, and send back the first traceback if one appears.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time

from src.contracts.scenario import Scenario
from src.data.causal_audit import audit_episode
from src.data.causal_episodes import _finalize_existing_episode, _pilot_coverage, _write_json
from src.data.dynamic_episodes import DynamicEpisodeConfig, _generate_events
from src.data.osm_demand_generator import OSMDemandConfig, generate_osm_scenario
from src.data.osm_ingestion import load_graphml
from src.data.training_pilot import map_fingerprint
from src.routing.initial_solution import InitialSolutionBuilder, InitialSolutionConfig
from src.routing.route_evaluator import RouteEvaluator
from src.sim.sumo_causal import run_sumo_causal_episode


class RealDelhiPilotError(ValueError):
    """Raised when a real-OSM pilot cannot be built safely."""


@dataclass(frozen=True, slots=True)
class RealDelhiMapSpec:
    """One physical map slot: a real OSM extract plus its demand policy."""

    graphml_path: str
    demand: OSMDemandConfig


def _solve_route_plan(scenario: Scenario):
    """Real VRP solve -- this is what feeds the SUMO route plan.

    Uses the exact ``InitialSolutionBuilder``/``RouteEvaluator`` pair you
    already validated this generator's scenarios against.
    """
    evaluator = RouteEvaluator(scenario)
    builder = InitialSolutionBuilder(scenario, evaluator, config=InitialSolutionConfig())
    result = builder.build()
    if not result.feasible or result.unassigned_customer_ids:
        raise RealDelhiPilotError(
            f"scenario {scenario.scenario_id!r} is not solvable end to end: "
            f"feasible={result.feasible}, "
            f"unassigned={result.unassigned_customer_ids}"
        )
    return result.route_plan


def generate_real_delhi_pilot(
    root: str | Path,
    map_specs: dict[int, RealDelhiMapSpec],
    *,
    duration_s: int = 2400,
    seed: int = 26137,
    traffic_only: bool = False,
) -> dict:
    """Real-OSM counterpart to ``training_pilot.generate_training_pilot``.

    ``map_specs`` must supply exactly the 4 map slots the production pilot
    manifest expects: ``{0: train_map_a, 1: train_map_b, 2: validation_map,
    3: test_map}``. Each must point at a *different* real OSM ``.graphml``
    extract (see the module docstring on why that's a hard requirement,
    not a style choice).

    ``traffic_only=False`` (the default here, unlike the pure-synthetic
    pilot) runs real solved vehicle routes inside SUMO, using the same
    ``InitialSolutionBuilder`` you already validated. Set ``True`` if you
    only want background-traffic episodes without vehicle routing.
    """
    required_slots = {0, 1, 2, 3}
    if set(map_specs) != required_slots:
        raise RealDelhiPilotError(
            f"map_specs must supply exactly slots {sorted(required_slots)} "
            f"(train x2, validation, test); got {sorted(map_specs)}"
        )

    root = Path(root).resolve()
    if (root / "corpus_manifest.json").exists():
        raise RealDelhiPilotError("Output already contains a pilot; choose a new directory")
    root.mkdir(parents=True, exist_ok=True)

    specs = [
        ("train", 0, "normal", None),
        ("train", 0, "morning_peak", "incident"),
        ("train", 1, "evening_peak", None),
        ("train", 1, "corridor_surge", "incident"),
        ("validation", 2, "morning_peak", "closure"),
        ("test", 3, "normal", "multi_disruption"),
    ]

    maps: dict[int, Scenario] = {}
    files: dict[str, str] = {}
    hashes: dict[str, str] = {}
    route_plans: dict[int, object] = {}

    for index, map_spec in sorted(map_specs.items()):
        network = load_graphml(map_spec.graphml_path)
        result = generate_osm_scenario(
            network,
            map_spec.demand,
            scenario_id=f"osm-delhi-map-{index}",
        )
        sc = result.scenario
        maps[index] = sc
        route_plans[index] = None if traffic_only else _solve_route_plan(sc)

        scenario_path = root / "maps" / f"map_{index}" / "scenario.json"
        scenario_path.parent.mkdir(parents=True, exist_ok=True)
        scenario_path.write_text(sc.model_dump_json(indent=2))

        files[sc.scenario_id] = str(scenario_path.relative_to(root)).replace("\\", "/")
        hashes[sc.scenario_id] = map_fingerprint(sc)

    if len(set(hashes.values())) != len(maps):
        raise RealDelhiPilotError(
            "Generated maps are not physically distinct -- at least two "
            "graphml_path entries in map_specs point at the same road "
            "network. Export 4 different OSM areas (e.g. different Delhi "
            "neighborhoods), not the same extract with different seeds."
        )

    manifest = {name: [] for name in ("train", "validation", "test")}
    for i, (split, *_rest) in enumerate(specs):
        manifest[split].append(f"ep-{i + 1:03d}")
    manifest.update(
        map_disjoint=True,
        scenario_files=files,
        map_fingerprints=hashes,
        rule="disjoint real OSM base maps and episodes assigned before windows",
        seed=seed,
        backend="sumo",
        source="openstreetmap",
    )
    _write_json(root / "corpus_manifest.json", manifest)

    results, audits = [], []
    started = time.perf_counter()
    for i, (split, index, regime, event_type) in enumerate(specs):
        eid = f"ep-{i + 1:03d}"
        sc = maps[index]
        config = DynamicEpisodeConfig(
            duration_s=duration_s,
            interval_s=60,
            warmup_s=300,
            regime=regime,
            event_type=event_type,
            event_count=1 if event_type else 0,
            event_duration_s=240,
            reveal_lead_s=60,
            observation_missing_fraction=0.03,
            seed=seed + 17 * i,
            backend="sumo",
        )
        folder = root / "episodes" / eid
        events = _generate_events(sc, config, eid)
        print(
            f"Running {eid}: {split}, {len(sc.nodes)} nodes, "
            f"{len(sc.edges)} roads, {len(sc.requests)} requests, {regime}",
            flush=True,
        )
        episode_start = time.perf_counter()
        run_sumo_causal_episode(
            scenario=sc,
            output_dir=folder,
            config=config,
            episode_id=eid,
            split=split,
            events=events,
            traffic_only=traffic_only,
            route_plan=route_plans[index],
        )
        result = _finalize_existing_episode(sc, folder, eid, split, config)
        result.wall_clock_s = time.perf_counter() - episode_start
        results.append(result)
        audit = audit_episode(folder)
        audits.append(audit)
        if not audit["ok"]:
            raise RuntimeError(audit)
        print(f"Finished {eid} in {result.wall_clock_s:.1f}s", flush=True)

    coverage = _pilot_coverage(results, time.perf_counter() - started)
    _write_json(root / "coverage.json", coverage)
    _write_json(
        root / "audit.json",
        dict(ok=all(a["ok"] for a in audits), episodes=audits),
    )
    return coverage