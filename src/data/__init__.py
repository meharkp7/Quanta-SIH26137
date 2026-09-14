"""Data generation and dataset contracts."""

from .dynamic_episodes import DynamicEpisodeConfig, generate_episode
from .pilot_dataset import generate_pilot_dataset

__all__ = ["DynamicEpisodeConfig", "generate_episode", "generate_pilot_dataset"]

from .empirical_traffic import TrafficProfileConfig, TrafficProfile, build_profile, load_real_traffic, write_profile
from .real_conditioned_episode import generate_real_conditioned_episode

from .canonical_traffic import TrafficDataset, TrafficObservation, load_dataset, profile_dataset, validate_dataset, write_canonical
from .metrla_spatial import (
    SensorLocation,
    SensorDistance,
    SpatialGraphProfile,
    load_metrla_spatial,
    build_spatial_graph,
    validate_metrla_spatial,
    write_metrla_spatial,
)
# Optional real-road-network ingestion. Dependencies are imported lazily by the
# ingestion module, so importing src.data does not require OSMnx.
from .osm_ingestion import (
    OSMIngestionConfig,
    OSMIngestionError,
    OSMNetwork,
    graph_to_quanta,
    load_graphml,
    load_osm_xml,
)

__all__ += [
    "OSMIngestionConfig",
    "OSMIngestionError",
    "OSMNetwork",
    "graph_to_quanta",
    "load_graphml",
    "load_osm_xml",
]
