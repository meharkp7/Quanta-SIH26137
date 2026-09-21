"""Turn SUMO floating-car output into map frames the dispatcher can play."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


FIXTURE_CLOSURES = (
    {"edge_id": "E23", "from_s": 50.0, "to_s": None},
)


def parse_fcd(path: Path, closures: tuple[dict, ...] | None = None) -> list[dict]:
    """Read SUMO FCD XML into per-second vehicle frames.

    Coordinates match the contract graph (meters), so the UI can project
    them with the same transform as nodes N0–N7.
    """
    tree = ET.parse(path)
    frames: list[dict] = []
    for timestep in tree.getroot().findall("timestep"):
        vehicles = []
        for vehicle in timestep.findall("vehicle"):
            lane = vehicle.get("lane") or ""
            edge_id = lane.rsplit("_", 1)[0] if lane else ""
            if edge_id.startswith(":"):
                edge_id = ""
            vehicle_id = vehicle.get("id") or ""
            vtype = vehicle.get("type") or ""
            speed = float(vehicle.get("speed") or 0.0)
            kind = (
                "delivery"
                if vehicle_id.startswith("V") or vtype.startswith("delivery")
                else "background"
            )
            vehicles.append(
                {
                    "id": vehicle_id,
                    "x": float(vehicle.get("x") or 0.0),
                    "y": float(vehicle.get("y") or 0.0),
                    "speed": speed,
                    "edge": edge_id,
                    "kind": kind,
                    "stopped": speed < 0.2,
                }
            )
        frames.append(
            {
                "t": float(timestep.get("time") or 0.0),
                "vehicles": vehicles,
                "closed": [],
            }
        )
    return apply_closures(frames, FIXTURE_CLOSURES if closures is None else closures)


def apply_closures(
    frames: list[dict],
    closures: tuple[dict, ...] = FIXTURE_CLOSURES,
) -> list[dict]:
    for frame in frames:
        closed = []
        for item in closures:
            start = float(item["from_s"])
            end = item.get("to_s")
            if frame["t"] >= start and (end is None or frame["t"] < float(end)):
                closed.append(item["edge_id"])
        frame["closed"] = closed
    return frames
