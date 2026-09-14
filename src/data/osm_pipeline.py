"""Explicit OSM-to-Scenario production pipeline.

Network acquisition remains local-file based. No implicit web download is
performed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.contracts.scenario import RoadNodeId, Scenario

from .osm_ingestion import OSMIngestionConfig, OSMNetwork, load_graphml, load_osm_xml
from .osm_scenario import OSMScenarioBuildResult, OSMScenarioBuilder, OSMSnapConfig


class OSMPipelineError(ValueError):
    """Raised when an OSM pipeline stage cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class OSMPipelineConfig:
    """Immutable, auditable configuration for one OSM conversion run."""

    ingestion: OSMIngestionConfig = OSMIngestionConfig()
    snapping: OSMSnapConfig = OSMSnapConfig()

    def __post_init__(self) -> None:
        if self.ingestion.provenance.strip() == "":
            raise OSMPipelineError("ingestion provenance must not be empty")


def load_network(
    path: str | Path,
    *,
    config: OSMPipelineConfig | None = None,
) -> OSMNetwork:
    """Load one local OSM XML or GraphML extract; never performs network I/O."""
    cfg = config or OSMPipelineConfig()
    path = Path(path)
    if not path.is_file():
        raise OSMPipelineError(f"OSM source file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".osm", ".xml"}:
        # The current ingestion boundary delegates to OSMnx. Whether a
        # particular PBF variant is accepted is therefore determined by
        # the installed OSMnx version; PBF acquisition is intentionally outside this XML loader.
        return load_osm_xml(str(path), config=cfg.ingestion)
    if suffix in {".graphml", ".xml.gz"}:
        return load_graphml(str(path), config=cfg.ingestion)
    raise OSMPipelineError(
        f"Unsupported OSM source format {suffix!r}; use .osm/.xml or .graphml"
    )


def build_scenario(
    scenario: Scenario,
    network: OSMNetwork,
    *,
    request_coordinates: Mapping[str, Any] | None = None,
    vehicle_start_nodes: Mapping[str, RoadNodeId] | None = None,
    vehicle_depot_nodes: Mapping[str, RoadNodeId] | None = None,
    config: OSMPipelineConfig | None = None,
) -> OSMScenarioBuildResult:
    """Attach an already-loaded OSM network to a Quanta Scenario."""
    cfg = config or OSMPipelineConfig()
    return OSMScenarioBuilder(
        network,
        snap_config=cfg.snapping,
    ).build(
        scenario,
        request_coordinates=request_coordinates,
        vehicle_start_nodes=vehicle_start_nodes,
        vehicle_depot_nodes=vehicle_depot_nodes,
    )
