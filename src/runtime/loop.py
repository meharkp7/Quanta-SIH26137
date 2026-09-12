"""Small closed routing loop used by the dispatcher UI.

Flow for one demo episode:

    load scenario
    -> persistence forecast (last free-flow speeds)
    -> rule scope (KEEP or GLOBAL if a road closed)
    -> QPSO
    -> independent validation
    -> optional SUMO execution

This is the Step 10 spine. Neural forecast and PPO are not required
for a legal, showable demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.platform.service import PlatformService, SolveOptions


@dataclass
class LoopState:
    scenario_id: str
    closed_edge_ids: tuple[str, ...]
    forecast_mode: str
    scope_action: str
    plan: dict[str, list[str]]
    evaluation: dict
    solve: dict
    notes: list[str] = field(default_factory=list)


class DemoLoop:
    def __init__(self, service: PlatformService | None = None) -> None:
        self.service = service or PlatformService()

    def run(
        self,
        scenario_id: str = "S3_BASE",
        *,
        closed_edge_ids: Sequence[str] = (),
        particles: int = 12,
        evaluations: int = 40,
        seed: int = 7,
        method: str = "qpso",
    ) -> LoopState:
        closed = tuple(closed_edge_ids)
        notes = [
            "Forecast = persistence of free-flow speeds (baseline, not a trained GNN).",
            "Scope = rule policy (GLOBAL if any road is closed, else KEEP).",
        ]
        action = "GLOBAL" if closed else "KEEP"
        if action == "KEEP" and not closed:
            notes.append("No incident visible. Rule policy keeps the current search as a full snapshot solve.")
        else:
            notes.append("Closed roads are visible. Rule policy forces GLOBAL replan of mutable jobs.")

        solve = self.service.solve(
            SolveOptions(
                method=method,
                particles=particles,
                evaluations=evaluations,
                seed=seed,
                closed_edge_ids=closed,
            ),
            scenario_id=scenario_id,
        )
        notes.append(
            f"{solve['method']} returned status={solve['status']} "
            f"in {solve['elapsed_s']:.2f}s."
        )
        if solve["evaluation"]["feasible"]:
            notes.append("Independent validator accepted the plan. It may be dispatched.")
        else:
            notes.append("Independent validator rejected the plan. It must not be dispatched.")

        return LoopState(
            scenario_id=scenario_id,
            closed_edge_ids=closed,
            forecast_mode="persistence",
            scope_action=action,
            plan=solve["plan"],
            evaluation=solve["evaluation"],
            solve=solve,
            notes=notes,
        )

    def as_dict(self, state: LoopState) -> dict:
        return {
            "scenario_id": state.scenario_id,
            "closed_edge_ids": list(state.closed_edge_ids),
            "forecast_mode": state.forecast_mode,
            "scope_action": state.scope_action,
            "plan": state.plan,
            "evaluation": state.evaluation,
            "solve": state.solve,
            "notes": state.notes,
        }
