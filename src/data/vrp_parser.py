"""CVRP parser for the Step 4 benchmark-to-road pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re


@dataclass(frozen=True)
class CustomerRecord:
    """One coordinate/demand record from the source CVRP instance."""

    customer_id: str
    x: float
    y: float
    demand: float


@dataclass(frozen=True)
class VrpInstance:
    """Immutable source information required by the Step 4 generator."""

    name: str
    capacity: float
    customers: tuple[CustomerRecord, ...]
    depot_id: str
    vehicle_count: int
    dimension: int
    distance_convention: str
    source_text: str
    source_checksum: str

    @property
    def depot(self) -> CustomerRecord:
        for record in self.customers:
            if record.customer_id == self.depot_id:
                return record
        raise ValueError(f"Depot {self.depot_id!r} has no coordinate record")


def parse_vrp(path: str | Path) -> VrpInstance:
    """Parse the CVRPLIB sections needed by Step 4."""

    source_path = Path(path)
    if source_path.suffix.lower() != ".vrp":
        raise ValueError(f"Expected a .vrp file, got {source_path.name!r}")

    source_text = source_path.read_text(encoding="utf-8")
    checksum = sha256(source_text.encode("utf-8")).hexdigest()

    fields: dict[str, str] = {}
    coordinates: dict[str, tuple[float, float]] = {}
    demands: dict[str, float] = {}
    depot_id: str | None = None
    section: str | None = None

    for raw_line in source_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.endswith("_SECTION"):
            section = upper
            continue
        if upper == "EOF":
            break

        if section is None and ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().upper()] = value.strip()
            continue

        parts = line.split()
        if section == "NODE_COORD_SECTION" and len(parts) >= 3:
            coordinates[parts[0]] = (float(parts[1]), float(parts[2]))
        elif section == "DEMAND_SECTION" and len(parts) >= 2:
            demands[parts[0]] = float(parts[1])
        elif section == "DEPOT_SECTION" and parts and parts[0] != "-1":
            depot_id = parts[0]

    dimension = _required_int(fields, "DIMENSION")
    capacity = _required_float(fields, "CAPACITY")
    if dimension != len(coordinates):
        raise ValueError(
            f"DIMENSION={dimension}, but {len(coordinates)} coordinates were parsed"
        )
    if depot_id is None or depot_id not in coordinates:
        raise ValueError("DEPOT_SECTION must contain a coordinate-defined depot")
    if not demands:
        raise ValueError("DEMAND_SECTION is required")
    if capacity <= 0:
        raise ValueError("CAPACITY must be positive")

    customers = tuple(
        CustomerRecord(
            customer_id=customer_id,
            x=coordinates[customer_id][0],
            y=coordinates[customer_id][1],
            demand=demands.get(customer_id, 0.0),
        )
        for customer_id in sorted(coordinates, key=_natural_id_key)
    )

    if any(record.demand < 0 for record in customers):
        raise ValueError("CVRP demand cannot be negative")
    if any(record.customer_id != depot_id and record.demand <= 0 for record in customers):
        raise ValueError("Every non-depot CVRP customer must have positive demand")
    if sum(record.demand for record in customers if record.customer_id != depot_id) <= 0:
        raise ValueError("CVRP instance contains no positive customer demand")

    return VrpInstance(
        name=fields.get("NAME", source_path.stem),
        capacity=capacity,
        customers=customers,
        depot_id=depot_id,
        vehicle_count=_vehicle_count(fields),
        dimension=dimension,
        distance_convention=_distance_convention(fields),
        source_text=source_text,
        source_checksum=checksum,
    )


def _required_int(fields: dict[str, str], key: str) -> int:
    try:
        return int(fields[key])
    except KeyError as exc:
        raise ValueError(f"Missing required CVRP field: {key}") from exc


def _required_float(fields: dict[str, str], key: str) -> float:
    try:
        return float(fields[key])
    except KeyError as exc:
        raise ValueError(f"Missing required CVRP field: {key}") from exc


def _vehicle_count(fields: dict[str, str]) -> int:
    for key in ("VEHICLES", "VEHICLE_COUNT", "K"):
        if key in fields:
            return max(1, int(fields[key]))

    comment = fields.get("COMMENT", "")
    match = re.search(r"(?:vehicles?|trucks?|k)\s*[:=]?\s*(\d+)", comment, re.I)
    return max(1, int(match.group(1))) if match else 2


def _distance_convention(fields: dict[str, str]) -> str:
    return fields.get("EDGE_WEIGHT_TYPE", fields.get("EDGE_WEIGHT_FORMAT", "UNSPECIFIED"))


def _natural_id_key(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"\d+", value)
    return (0, f"{int(value):012d}") if match else (1, value)
