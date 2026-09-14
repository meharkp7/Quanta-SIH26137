"""Regression tests for the optional OSM ingestion public API."""

from __future__ import annotations

import importlib


def test_src_data_exposes_osm_api_without_eager_osmnx_import():
    data = importlib.import_module("src.data")
    assert hasattr(data, "OSMIngestionConfig")
    assert hasattr(data, "graph_to_quanta")
    module = importlib.import_module("src.data.osm_ingestion")
    assert "osmnx" not in module.__dict__
