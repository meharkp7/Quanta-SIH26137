"""Generate six real SUMO episodes on disjoint training/validation/test maps."""
from pathlib import Path
import argparse
import hashlib
import json
import time
from src.data.dataset_generator import DynamicDatasetConfig, generate_dynamic_dataset
from src.contracts.scenario import Scenario
from src.data.dynamic_episodes import DynamicEpisodeConfig, _generate_events
from src.data.causal_episodes import _finalize_existing_episode, _pilot_coverage, _write_json
from src.data.causal_audit import audit_episode
from src.sim.sumo_causal import run_sumo_causal_episode

def map_fingerprint(scenario):
    # Physical coordinates/connectivity, independent of scenario names and splits.
    nodes = {n.node_id: (n.x_m,n.y_m) for n in scenario.nodes}
    roads = sorted((nodes[e.from_node],nodes[e.to_node],e.length_m,e.speed_limit_mps,e.lane_count)
                   for e in scenario.edges)
    return hashlib.sha256(json.dumps(roads).encode()).hexdigest()

def generate_training_pilot(root, *, duration_s=2400, seed=26137):
    root = Path(root).resolve()
    if (root / "split_manifest.json").exists():
        raise ValueError("Output already contains a pilot; choose a new directory")
    root.mkdir(parents=True, exist_ok=True)
    specs = [("train",0,"normal",None), ("train",0,"morning_peak","incident"),
             ("train",1,"evening_peak",None), ("train",1,"corridor_surge","incident"),
             ("validation",2,"morning_peak","closure"), ("test",3,"normal","multi_disruption")]
    maps, files, hashes = {}, {}, {}
    for split, index in (("train",0),("train",1),("validation",2),("test",3)):
        folder = root / "maps" / f"map_{index}"
        cols = 3 if index % 2 == 0 else 4
        path = generate_dynamic_dataset(folder, DynamicDatasetConfig(customer_count=5,
            road_junction_count=3*cols, grid_rows=3, grid_cols=cols, extent_m=900,
            topology_family="grid" if index % 2 == 0 else "irregular",
            jitter_fraction=0.1, one_way_fraction=0, seed=seed+index*101,
            dataset_split=split, max_customer_snap_m=500, minimum_road_segment_m=10))
        sc = Scenario.model_validate_json((folder / "scenario.json").read_text())
        maps[index] = sc
        files[sc.scenario_id] = str((folder / "scenario.json").relative_to(root)).replace("\\", "/")
        hashes[sc.scenario_id] = map_fingerprint(sc)
    if len(set(hashes.values())) != len(maps):
        raise ValueError("Generated maps are not distinct")
    manifest = {name: [] for name in ("train","validation","test")}
    for i,(split,*_) in enumerate(specs):
        manifest[split].append(f"ep-{i+1:03d}")
    manifest.update(map_disjoint=True, scenario_files=files, map_fingerprints=hashes,
        rule="disjoint base maps and episodes assigned before windows", seed=seed, backend="sumo")
    _write_json(root / "split_manifest.json", manifest)
    results, audits = [], []
    started = time.perf_counter()
    for i,(split,index,regime,event_type) in enumerate(specs):
        eid = f"ep-{i+1:03d}"
        sc = maps[index]
        config = DynamicEpisodeConfig(duration_s=duration_s, interval_s=60, warmup_s=300,
            regime=regime, event_type=event_type, event_count=1 if event_type else 0,
            event_duration_s=240, reveal_lead_s=60, observation_missing_fraction=0.03,
            seed=seed+17*i, backend="sumo")
        folder = root / "episodes" / eid
        events = _generate_events(sc,config,eid)
        print(f"Running {eid}: {split}, {len(sc.edges)} roads, {regime}", flush=True)
        episode_start = time.perf_counter()
        run_sumo_causal_episode(scenario=sc,output_dir=folder,config=config,
            episode_id=eid,split=split,events=events,traffic_only=True)
        result = _finalize_existing_episode(sc,folder,eid,split,config)
        result.wall_clock_s = time.perf_counter()-episode_start
        results.append(result)
        audit = audit_episode(folder)
        audits.append(audit)
        if not audit["ok"]:
            raise RuntimeError(audit)
        print(f"Finished {eid} in {result.wall_clock_s:.1f}s", flush=True)
    coverage = _pilot_coverage(results,time.perf_counter()-started)
    _write_json(root / "coverage.json",coverage)
    _write_json(root / "audit.json",dict(ok=all(a["ok"] for a in audits),episodes=audits))
    return coverage

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--duration",type=int,default=2400)
    args=parser.parse_args()
    print(json.dumps(generate_training_pilot(args.output,duration_s=args.duration),indent=2))

if __name__ == "__main__":
    main()
