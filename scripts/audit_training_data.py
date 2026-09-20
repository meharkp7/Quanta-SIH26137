"""Audit GNN training-data quality for corpus_delhi (read-only vs the corpus).

Reads corpus bytes but never modifies them. Produces:
  artifacts/data_audit/quality_report.json
  artifacts/data_audit/quality_report.md
  artifacts/data_audit/sampler_proposal.json

Sampling (stated in outputs): manifest-level metrics use ALL 250 episodes
(manifest JSON only, cheap). Episode-deep metrics stream a deterministic
stratified sample: 2 episodes per train map (6) + 3 validation + 3 test = 12.
Window/causal checks build windows for 1 episode per split (3 total).

Usage:
  python3 scripts/audit_training_data.py [--corpus corpus_delhi]
      [--out artifacts/data_audit] [--seed 0] [--gate]
--gate applies hard data-gate thresholds and exits non-zero on violation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Ensure repo root is importable when run as `python3 scripts/...` (script dir
# shadows cwd on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HORIZONS = (300, 600, 900)
KINDS = ("speed_proxy", "realized_traversal")

# Hard data-gate thresholds: violations FAIL loudly (non-zero exit).
GATE_THRESHOLDS = {
    "min_mean_speed_coverage": 0.01,   # per split, manifest label coverage
    "min_mean_trav_coverage": 0.005,   # per split, manifest label coverage
    "max_missing_fraction": 0.995,     # sampled observations
    "max_split_density_drift": 3.0,    # max(val,test mean coverage)/train
    "max_trav_p99_s": 600.0,           # traversal label sanity cap
    "max_speed_ratio_p999": 3.0,       # speed_ratio sanity cap
}


def load_manifest(corpus: Path) -> dict:
    return json.loads((corpus / "corpus_manifest.json").read_text())


def manifest_metrics(m: dict) -> dict:
    out: dict = {"per_split": {}, "regime_balance": {}, "event_balance": {},
                 "per_map": {}, "disjointness": {}}
    for split in ("train", "validation", "test"):
        eps = [e for e in m["episodes"] if e["split"] == split]
        sp = [e["label_coverage"]["speed_proxy"]["coverage"] for e in eps]
        tr = [e["label_coverage"]["realized_traversal"]["coverage"] for e in eps]
        out["per_split"][split] = {
            "episodes": len(eps),
            "speed_coverage_mean": float(np.mean(sp)),
            "speed_coverage_min": float(np.min(sp)),
            "speed_coverage_max": float(np.max(sp)),
            "trav_coverage_mean": float(np.mean(tr)),
            "trav_coverage_min": float(np.min(tr)),
            "trav_coverage_max": float(np.max(tr)),
            "speed_valid_total": int(sum(e["label_coverage"]["speed_proxy"]["valid"] for e in eps)),
            "trav_valid_total": int(sum(e["label_coverage"]["realized_traversal"]["valid"] for e in eps)),
        }
        out["regime_balance"][split] = dict(Counter(e["regime"] for e in eps))
        out["event_balance"][split] = dict(Counter(str(e["event_type"]) for e in eps))
    for idx, rec in m["map_records"].items():
        meps = [e for e in m["episodes"] if e["map_index"] == int(idx)]
        sp_valid = sum(e["label_coverage"]["speed_proxy"]["valid"] for e in meps)
        out["per_map"][rec["scenario_id"]] = {
            "split": rec["split"], "zone": rec["zone"], "edges": rec["edges"],
            "nodes": rec["nodes"], "episodes": len(meps),
            "speed_valid_per_edge_per_ep": float(sp_valid / max(len(meps), 1) / max(rec["edges"], 1)),
        }
    t, v, te = set(m["train"]), set(m["validation"]), set(m["test"])
    fps = m.get("map_fingerprints", {})
    scen_split: dict[str, set] = defaultdict(set)
    for e in m["episodes"]:
        scen_split[e["scenario_id"]].add(e["split"])
    out["disjointness"] = {
        "episode_overlap_train_val": len(t & v),
        "episode_overlap_train_test": len(t & te),
        "episode_overlap_val_test": len(v & te),
        "map_disjoint_flag": bool(m.get("map_disjoint")),
        "fingerprints_unique": len(set(fps.values())) == len(fps),
        "scenarios_single_split": all(len(s) == 1 for s in scen_split.values()),
        "scenario_to_split": {k: sorted(s) for k, s in scen_split.items()},
    }
    return out


def scenario_edge_info(corpus: Path, m: dict) -> dict[str, dict]:
    info: dict[str, dict] = {}
    for sid, rel in m["scenario_files"].items():
        raw = json.loads((corpus / rel).read_text())
        info[sid] = {e["edge_id"]: (float(e["speed_limit_mps"]), str(e.get("parent_road_id", "")))
                     for e in raw["edges"]}
    return info


def pick_sample(m: dict, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sample: list[dict] = []
    train_by_map: dict[int, list] = defaultdict(list)
    for e in m["episodes"]:
        if e["split"] == "train":
            train_by_map[e["map_index"]].append(e)
    for _, eps in sorted(train_by_map.items()):
        sample += rng.sample(sorted(eps, key=lambda x: x["episode_id"]), 2)
    for split, n in (("validation", 3), ("test", 3)):
        eps = sorted([e for e in m["episodes"] if e["split"] == split],
                     key=lambda x: x["episode_id"])
        sample += rng.sample(eps, n)
    return sample


def stream_labels(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def audit_episode(ep_dir: Path, meta: dict, edge_info: dict) -> dict:
    """Deep metrics for one episode: horizons x kinds, obs stats, skew, incidents."""
    horizon_kind = Counter()
    horizon_kind_valid = Counter()
    trav_vals: list[float] = []
    speed_ratio_vals: list[float] = []
    limits = {eid: lim for eid, (lim, _) in edge_info[meta["scenario_id"]].items()}
    parent_of = {eid: par for eid, (_, par) in edge_info[meta["scenario_id"]].items()}
    events = json.loads((ep_dir / "events.json").read_text())
    # affected (edge, bucket) pairs: buckets are minute-aligned target buckets
    affected: set[tuple[str, int]] = set()
    for ev in events:
        s, en = int(ev["effect_start_s"]), int(ev["effect_end_s"])
        roads = set(ev.get("affected_parent_road_ids", []))
        aff_edges = [eid for eid, p in parent_of.items() if p in roads]
        b = (s // 60) * 60
        while b < en:
            for eid in aff_edges:
                affected.add((eid, b))
            b += 60
    n_lab = 0
    valid_incident = 0
    valid_total = 0
    avail_viol = 0
    for d in stream_labels(ep_dir / "labels.jsonl"):
        n_lab += 1
        h = d["target_time_s"] - d["issue_time_s"]
        k = d["target_kind"]
        horizon_kind[(h, k)] += 1
        ok = (not d["missing"]) and (d["value"] is not None)
        if ok:
            horizon_kind_valid[(h, k)] += 1
            valid_total += 1
            tb = (d["target_time_s"] // 60) * 60
            if (d["edge_id"], tb) in affected:
                valid_incident += 1
            if k == "realized_traversal":
                trav_vals.append(float(d["value"]))
        if (not d["missing"]) and d["value"] is not None and d["available_at_s"] is not None:
            if float(d["available_at_s"]) < float(d["target_time_s"]):
                avail_viol += 1
    # observations
    n_obs = 0
    n_miss = 0
    ages: list[float] = []
    kc = 0
    spd_sum = 0.0
    with (ep_dir / "observations.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            n_obs += 1
            miss = int(r["missing"])
            n_miss += miss
            try:
                ages.append(float(r["observation_age_s"]))
            except ValueError:
                pass
            kc += int(r["known_closed"])
            if not miss and r["observed_speed_mps"] not in (None, ""):
                lim = limits.get(r["edge_id"], 0.0)
                if lim and lim > 0:
                    speed_ratio_vals.append(float(r["observed_speed_mps"]) / lim)
    ages_a = np.asarray(ages, dtype=np.float64)
    sr = np.asarray(speed_ratio_vals, dtype=np.float64)
    tv = np.asarray(trav_vals, dtype=np.float64)
    return {
        "episode_id": meta["episode_id"], "split": meta["split"],
        "n_labels": n_lab,
        "coverage_by_horizon_kind": {
            f"{h}s/{k}": {"total": horizon_kind[(h, k)],
                          "valid": horizon_kind_valid[(h, k)],
                          "coverage": horizon_kind_valid[(h, k)] / max(horizon_kind[(h, k)], 1)}
            for h in HORIZONS for k in KINDS},
        "missing_fraction": n_miss / max(n_obs, 1),
        "obs_age_p50": float(np.median(ages_a)) if len(ages_a) else float("nan"),
        "obs_age_p99": float(np.quantile(ages_a, 0.99)) if len(ages_a) else float("nan"),
        "obs_age_max": float(np.max(ages_a)) if len(ages_a) else float("nan"),
        "known_closed_fraction": kc / max(n_obs, 1),
        "speed_ratio_n": len(sr),
        "speed_ratio_p1": float(np.quantile(sr, 0.01)) if len(sr) else float("nan"),
        "speed_ratio_p50": float(np.median(sr)) if len(sr) else float("nan"),
        "speed_ratio_p99": float(np.quantile(sr, 0.99)) if len(sr) else float("nan"),
        "speed_ratio_p999": float(np.quantile(sr, 0.999)) if len(sr) else float("nan"),
        "speed_ratio_max": float(np.max(sr)) if len(sr) else float("nan"),
        "speed_ratio_frac_gt1": float(np.mean(sr > 1.0)) if len(sr) else float("nan"),
        "speed_ratio_frac_eq0": float(np.mean(sr == 0.0)) if len(sr) else float("nan"),
        "trav_n": len(tv),
        "trav_p50": float(np.median(tv)) if len(tv) else float("nan"),
        "trav_p99": float(np.quantile(tv, 0.99)) if len(tv) else float("nan"),
        "trav_max": float(np.max(tv)) if len(tv) else float("nan"),
        "incident_window_fraction_of_valid": valid_incident / max(valid_total, 1),
        "avail_before_target_violations": avail_viol,
    }


def window_checks(corpus: Path, m: dict, ep_ids: list[str]) -> dict:
    from src.contracts.scenario import Scenario
    from src.learning.windows import assert_window_causal, build_episode_windows
    scen_cache: dict[str, Scenario] = {}
    res: dict = {"episodes": {}, "assert_failures": 0, "windows_total": 0}
    for eid in ep_ids:
        meta = next(e for e in m["episodes"] if e["episode_id"] == eid)
        sid = meta["scenario_id"]
        if sid not in scen_cache:
            scen_cache[sid] = Scenario.model_validate_json(
                (corpus / m["scenario_files"][sid]).read_text())
        wins = build_episode_windows(corpus / "episodes" / eid, scen_cache[sid],
                                     split=meta["split"])
        fails = 0
        for w in wins:
            try:
                assert_window_causal(w)
            except AssertionError:
                fails += 1
        # window-level per-horizon coverage (mean over windows)
        cov = {}
        for i, h in enumerate((5, 10, 15)):
            sm = np.mean([w.speed_target_mask[:, i].mean() for w in wins]) if wins else 0.0
            tm = np.mean([w.traversal_target_mask[:, i].mean() for w in wins]) if wins else 0.0
            cov[f"{h}min"] = {"speed": float(sm), "traversal": float(tm)}
        res["episodes"][eid] = {"n_windows": len(wins), "assert_failures": fails,
                                "window_coverage": cov}
        res["assert_failures"] += fails
        res["windows_total"] += len(wins)
    return res


def scaler_analysis(corpus: Path, m: dict, train_sample: list[dict]) -> dict:
    from src.contracts.scenario import Scenario
    from src.learning.scaling import FeatureScaler
    from src.learning.schema import FEATURE_NAMES
    from src.learning.windows import build_episode_windows
    feats, masks = [], []
    for meta in train_sample:
        sid = meta["scenario_id"]
        sc = Scenario.model_validate_json((corpus / m["scenario_files"][sid]).read_text())
        for w in build_episode_windows(corpus / "episodes" / meta["episode_id"], sc, split="train"):
            feats.append(np.where(w.feature_mask, w.features, np.nan))
    stacked_list = []
    max_e = max(f.shape[1] for f in feats)
    for f in feats:
        if f.shape[1] < max_e:
            pad = np.full((f.shape[0], max_e - f.shape[1], f.shape[2]), np.nan,
                          dtype=np.float32)
            f = np.concatenate([f, pad], axis=1)
        stacked_list.append(f[None])
    stacked = np.concatenate(stacked_list, axis=0)  # [B,L,E,F], NaN-padded
    scaler = FeatureScaler.fit(stacked, FEATURE_NAMES)
    per_feature = {}
    flat = stacked.reshape(-1, stacked.shape[-1])
    for i, name in enumerate(FEATURE_NAMES):
        col = flat[:, i]
        fin = col[np.isfinite(col)]
        z = (fin - scaler.mean[i]) / scaler.scale[i] if scaler.scale[i] else fin * 0
        per_feature[name] = {
            "mean": scaler.mean[i], "std": scaler.scale[i],
            "finite_fraction": float(len(fin) / max(col.size, 1)),
            "skew": float((((fin - fin.mean()) ** 3).mean() / max(fin.std() ** 3, 1e-12)) if len(fin) > 2 else 0.0),
            "frac_abs_z_gt5": float(np.mean(np.abs(z) > 5)) if len(z) else 0.0,
            "p99": float(np.quantile(fin, 0.99)) if len(fin) else float("nan"),
            "p999": float(np.quantile(fin, 0.999)) if len(fin) else float("nan"),
        }
    return {"scaler_means": list(scaler.mean), "scaler_stds": list(scaler.scale),
            "clips_applied": False, "per_feature": per_feature,
            "windows_used": len(feats)}


def run_gate(report: dict) -> list[str]:
    T = GATE_THRESHOLDS
    fails: list[str] = []
    for split, ps in report["manifest"]["per_split"].items():
        if ps["speed_coverage_mean"] < T["min_mean_speed_coverage"]:
            fails.append(f"GATE FAIL: {split} mean speed coverage {ps['speed_coverage_mean']:.4f} < {T['min_mean_speed_coverage']}")
        if ps["trav_coverage_mean"] < T["min_mean_trav_coverage"]:
            fails.append(f"GATE FAIL: {split} mean traversal coverage {ps['trav_coverage_mean']:.4f} < {T['min_mean_trav_coverage']}")
    d = report["manifest"]["disjointness"]
    if d["episode_overlap_train_val"] or d["episode_overlap_train_test"] or d["episode_overlap_val_test"]:
        fails.append("GATE FAIL: episode leakage across splits")
    if not d["fingerprints_unique"] or not d["scenarios_single_split"]:
        fails.append("GATE FAIL: map/scenario leakage across splits")
    miss = np.mean([e["missing_fraction"] for e in report["episodes"]])
    if miss > T["max_missing_fraction"]:
        fails.append(f"GATE FAIL: sampled missing fraction {miss:.4f} > {T['max_missing_fraction']}")
    tr = report["manifest"]["per_split"]["train"]["speed_coverage_mean"]
    drift = max(report["manifest"]["per_split"][s]["speed_coverage_mean"] for s in ("validation", "test")) / max(tr, 1e-9)
    if drift > T["max_split_density_drift"]:
        fails.append(f"GATE FAIL: split density drift {drift:.2f}x > {T['max_split_density_drift']}")
    if report["windows"]["assert_failures"]:
        fails.append(f"GATE FAIL: {report['windows']['assert_failures']} causal-assert failures")
    if any(e["avail_before_target_violations"] for e in report["episodes"]):
        fails.append("GATE FAIL: label available_at_s precedes target_time")
    return fails


def build_sampler_proposal(report: dict) -> dict:
    agg = Counter()
    valid = Counter()
    for e in report["episodes"]:
        m = e["episode_id"]
        # re-derive from stored per-horizon coverage with totals
        for hk, c in e["coverage_by_horizon_kind"].items():
            agg[hk] += c["total"]
            valid[hk] += c["valid"]
    weights = {}
    inv = {hk: 1.0 / math.sqrt(max(valid[hk], 1)) for hk in agg}
    z = sum(inv.values())
    for hk in agg:
        weights[hk] = {"valid": valid[hk], "total": agg[hk],
                       "coverage": valid[hk] / max(agg[hk], 1),
                       "recommended_loss_weight": inv[hk] / z * len(agg)}
    inc = float(np.mean([e["incident_window_fraction_of_valid"] for e in report["episodes"]]))
    return {
        "status": "PROPOSAL ONLY - trainer untouched; adopt in next run after review",
        "measured_sample_episodes": len(report["episodes"]),
        "loss_weights_inverse_sqrt_coverage": weights,
        "incident_window_fraction_of_valid_sampled": inc,
        "incident_recommendation": (
            f"route ~{min(0.5, max(0.15, 4 * inc)):.2f} of each batch from "
            f"incident-affected (edge,bucket) pairs OR scale incident-window loss by "
            f"~{1.0 / max(inc, 1e-3):.1f}x capped at 8x; measured incident fraction={inc:.4f}"),
        "traversal_recommendation": ("log1p-transform or winsorize traversal targets "
            "at p99 before MSE (heavy right tail); see quality_report scaler section"),
    }


def write_markdown(report: dict, proposal: dict) -> str:
    L: list[str] = ["# Training-data quality report - corpus_delhi", ""]
    L.append(f"Sampling: manifest metrics = ALL {report['n_episodes_manifest']} episodes; "
             f"episode-deep metrics = stratified sample of {len(report['episodes'])} "
             f"episodes ({report['sample_ids']}); windows built for {report['window_ep_ids']}.")
    L.append("")
    L.append("## Headline coverage (manifest, all 250 episodes)")
    L.append("| split | eps | speed mean | speed min-max | trav mean | trav min-max |")
    L.append("|---|---|---|---|---|---|")
    for s, ps in report["manifest"]["per_split"].items():
        L.append(f"| {s} | {ps['episodes']} | {ps['speed_coverage_mean']:.4f} | "
                 f"{ps['speed_coverage_min']:.4f}-{ps['speed_coverage_max']:.4f} | "
                 f"{ps['trav_coverage_mean']:.4f} | {ps['trav_coverage_min']:.4f}-{ps['trav_coverage_max']:.4f} |")
    L.append("")
    L.append("## Sampled per-horizon x kind coverage (labels.jsonl stream)")
    L.append("| horizon/kind | valid | total | coverage |")
    L.append("|---|---|---|---|")
    agg = Counter()
    valid = Counter()
    for e in report["episodes"]:
        for hk, c in e["coverage_by_horizon_kind"].items():
            agg[hk] += c["total"]
            valid[hk] += c["valid"]
    for hk in sorted(agg):
        L.append(f"| {hk} | {valid[hk]} | {agg[hk]} | {valid[hk]/max(agg[hk],1):.4f} |")
    L.append("")
    L.append("## Ranked data-side limits (severity + evidence)")
    for i, lim in enumerate(report["ranked_limits"], 1):
        L.append(f"{i}. **[{lim['severity']}] {lim['title']}** - {lim['evidence']}")
    L.append("")
    L.append("## Leakage / causality")
    d = report["manifest"]["disjointness"]
    L.append(f"- episode overlap t/v/v/t: {d['episode_overlap_train_val']}/"
             f"{d['episode_overlap_train_test']}/{d['episode_overlap_val_test']}; "
             f"fingerprints unique: {d['fingerprints_unique']}; "
             f"scenarios single-split: {d['scenarios_single_split']}")
    L.append(f"- causal asserts over {report['windows']['windows_total']} windows: "
             f"{report['windows']['assert_failures']} failures; "
             f"avail<target violations: {sum(e['avail_before_target_violations'] for e in report['episodes'])}")
    L.append("")
    L.append("## Scaler / outliers")
    for name, pf in report["scaler"]["per_feature"].items():
        L.append(f"- {name}: mean={pf['mean']:.4g} std={pf['std']:.4g} skew={pf['skew']:.2f} "
                 f"|z|>5 frac={pf['frac_abs_z_gt5']:.4f} (no clipping in FeatureScaler)")
    L.append("")
    L.append("## Sampler proposal (not applied)")
    L.append(f"- incident fraction of valid labels: {proposal['incident_window_fraction_of_valid_sampled']:.4f}")
    L.append(f"- {proposal['incident_recommendation']}")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus_delhi")
    ap.add_argument("--out", default="artifacts/data_audit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    corpus = Path(args.corpus)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = load_manifest(corpus)
    mani = manifest_metrics(m)
    edge_info = scenario_edge_info(corpus, m)
    sample = pick_sample(m, args.seed)
    episodes = [audit_episode(corpus / "episodes" / e["episode_id"], e,
                              edge_info) for e in sample]
    train_sample = [e for e in sample if e["split"] == "train"][:3]
    win_eps = [next(e["episode_id"] for e in sample if e["split"] == s)
               for s in ("train", "validation", "test")]
    wins = window_checks(corpus, m, win_eps)
    scaler = scaler_analysis(corpus, m, train_sample)
    tr = mani["per_split"]["train"]["speed_coverage_mean"]
    drift = max(mani["per_split"][s]["speed_coverage_mean"] for s in ("validation", "test")) / max(tr, 1e-9)
    report = {
        "corpus": str(corpus), "seed": args.seed,
        "n_episodes_manifest": len(m["episodes"]),
        "sample_ids": [e["episode_id"] for e in sample],
        "window_ep_ids": win_eps,
        "manifest": mani,
        "episodes": episodes,
        "windows": wins,
        "scaler": scaler,
        "split_density_drift_vs_train": drift,
        "ranked_limits": [
            {"severity": "HIGH", "title": "Extreme supervision sparsity",
             "evidence": f"train speed coverage mean {tr:.4f}, traversal "
             f"{mani['per_split']['train']['trav_coverage_mean']:.4f}; sampled obs missing "
             f"{np.mean([e['missing_fraction'] for e in episodes]):.3f}; only 9 issue-times/episode"},
            {"severity": "HIGH", "title": "9-windows/episode thinness",
             "evidence": f"{sum(v['n_windows'] for v in wins['episodes'].values())} windows over "
             f"{len(win_eps)} sampled episodes; ~2250 windows corpus-wide dilute incident supervision"},
            {"severity": "MEDIUM-HIGH", "title": "Train/val/test density drift",
             "evidence": f"val/test mean speed coverage "
             f"{mani['per_split']['validation']['speed_coverage_mean']:.4f}/"
             f"{mani['per_split']['test']['speed_coverage_mean']:.4f} vs train {tr:.4f} ({drift:.2f}x)"},
            {"severity": "MEDIUM-HIGH", "title": "Traversal-target heavy tail + scale mismatch",
             "evidence": f"sampled trav p50 {np.median([e['trav_p50'] for e in episodes]):.1f}s, p99 "
             f"{np.median([e['trav_p99'] for e in episodes]):.1f}s, max "
             f"{max(e['trav_max'] for e in episodes):.1f}s vs speed_ratio~[0,1]"},
            {"severity": "MEDIUM", "title": "Incident supervision thin",
             "evidence": f"incident-window fraction of valid labels "
             f"{np.mean([e['incident_window_fraction_of_valid'] for e in episodes]):.4f} (sampled)"},
            {"severity": "MEDIUM", "title": "Per-map graph-size spread + single-map val/test",
             "evidence": f"edges {[mani['per_map'][k]['edges'] for k in sorted(mani['per_map'])]}; val/test rely on one map each"},
            {"severity": "LOW-MEDIUM", "title": "No scaler clipping on skewed features",
             "evidence": "FeatureScaler is pure z-score; halting/occupancy skew reported per-feature above"},
        ],
    }
    proposal = build_sampler_proposal(report)
    (out / "quality_report.json").write_text(json.dumps(report, indent=2))
    (out / "quality_report.md").write_text(write_markdown(report, proposal))
    (out / "sampler_proposal.json").write_text(json.dumps(proposal, indent=2))
    print(f"wrote {out}/quality_report.json+md sampler_proposal.json")
    print(f"train speed cov={tr:.4f} drift={drift:.2f}x "
          f"assert_fails={wins['assert_failures']} windows={wins['windows_total']}")
    if args.gate:
        fails = run_gate(report)
        for f in fails:
            print(f, file=sys.stderr)
        if fails:
            print(f"DATA GATE: {len(fails)} FAILURES", file=sys.stderr)
            return 2
        print("DATA GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

