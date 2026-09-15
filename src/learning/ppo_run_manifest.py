from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PPORunManifest:
    """Immutable provenance record for one PPO training run."""

    run_id: str

    seed: int

    representation_version: str
    forecaster_version: str
    reward_version: str
    simulator_version: str
    qpso_version: str

    curriculum_level: int

    total_env_steps: int
    rollout_steps: int

    device: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: str | Path) -> "PPORunManifest":
        data = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        return cls(**data)