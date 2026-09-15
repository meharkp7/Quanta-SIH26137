"""
Converts a Scenario contract into SUMO network input files (.nod.xml, .edg.xml)
and invokes netconvert to compile net.xml.  Also maintains the canonical
road_id <-> SUMO edge/lane mapping so all other sim modules can cross-reference
back to our contract IDs.
"""

from __future__ import annotations

import json
import os
import subprocess
import shutil
import textwrap
from pathlib import Path
from xml.etree import ElementTree as ET

from src.contracts.scenario import Scenario

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def _find_sumo_binary(name: str) -> Path | None:
    """Resolve a SUMO executable on Windows, macOS, or Linux.

    Resolution order:
      1. SUMO_HOME/bin/<name> (or <name>.exe)
      2. executable discovered on PATH

    Returning ``None`` lets callers raise a precise configuration error.
    """
    env_home = os.environ.get("SUMO_HOME")
    if env_home:
        home = Path(env_home)
        for candidate in (
            home / "bin" / name,
            home / "bin" / f"{name}.exe",
        ):
            if candidate.is_file():
                return candidate.resolve()

    executable = shutil.which(name)
    if executable:
        return Path(executable).resolve()

    if not name.endswith(".exe"):
        executable = shutil.which(f"{name}.exe")
        if executable:
            return Path(executable).resolve()

    return None


NETCONVERT_EXE = _find_sumo_binary("netconvert")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SumoExporter:
    """
    Exports a Scenario to SUMO network files and maintains the edge mapping.

    Usage::

        exporter = SumoExporter(scenario, output_dir=Path("artifacts/sumo/step6"))
        net_path = exporter.export()          # writes files, runs netconvert
        mapping = exporter.edge_mapping       # dict edge_id -> sumo_edge_id

    The SUMO edge ID convention used here is identical to the contract edge_id
    so that TraCI calls use the same identifiers everywhere.  The mapping dict
    is kept explicit to make any future renaming transparent.
    """

    def __init__(self, scenario: Scenario, output_dir: Path) -> None:
        self.scenario = scenario
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # edge_id (contract) -> SUMO edge id (string used in .edg.xml / TraCI)
        self.edge_mapping: dict[str, str] = {}
        # node_id (contract) -> SUMO junction id
        self.node_mapping: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def export(self) -> Path:
        """
        Write node/edge XML files and compile with netconvert.

        Returns the path to the compiled net.xml file.
        Raises RuntimeError if netconvert is not found or fails.
        """
        nod_path = self._write_nodes()
        edg_path = self._write_edges()
        con_path = self._write_connections()
        net_path = self.output_dir / "net.xml"

        self._run_netconvert(nod_path, edg_path, con_path, net_path)
        self._write_mapping_json()

        return net_path

    # ------------------------------------------------------------------
    # XML writers
    # ------------------------------------------------------------------

    def _write_nodes(self) -> Path:
        """Write .nod.xml: one SUMO junction per scenario node."""
        root = ET.Element("nodes")
        for node in self.scenario.nodes:
            sumo_id = node.node_id          # keep identical to contract ID
            self.node_mapping[node.node_id] = sumo_id
            ET.SubElement(
                root,
                "node",
                attrib={
                    "id": sumo_id,
                    "x": str(node.x_m),
                    "y": str(node.y_m),
                    "type": self._junction_type(node.signalized),
                },
            )
        path = self.output_dir / "nodes.nod.xml"
        _write_xml(root, path)
        return path

    def _write_edges(self) -> Path:
        """Write .edg.xml: one directed SUMO edge per scenario edge."""
        root = ET.Element("edges")
        for edge in self.scenario.edges:
            sumo_id = edge.edge_id          # identical to contract ID
            self.edge_mapping[edge.edge_id] = sumo_id
            ET.SubElement(
                root,
                "edge",
                attrib={
                    "id": sumo_id,
                    "from": self.node_mapping.get(edge.from_node, edge.from_node),
                    "to": self.node_mapping.get(edge.to_node, edge.to_node),
                    "numLanes": str(edge.lane_count),
                    "speed": f"{edge.speed_limit_mps:.4f}",
                    "length": f"{edge.length_m:.4f}",
                    "priority": str(self._road_priority(edge.road_class)),
                },
            )
        path = self.output_dir / "edges.edg.xml"
        _write_xml(root, path)
        return path

    def _write_connections(self) -> Path:
        """
        Write .con.xml: explicit turn connections at junctions.

        For each junction, find all incoming and outgoing edges.
        Every incoming->outgoing pair that is not a U-turn is permitted.
        This avoids SUMO's default heuristic from blocking valid turns in
        a small graph like the step-3 fixture.
        """
        # Build adjacency: to_node -> [edge], from_node -> [edge]
        incoming: dict[str, list] = {}
        outgoing: dict[str, list] = {}
        for edge in self.scenario.edges:
            incoming.setdefault(edge.to_node, []).append(edge)
            outgoing.setdefault(edge.from_node, []).append(edge)

        root = ET.Element("connections")
        all_junction_ids = {n.node_id for n in self.scenario.nodes}

        for junction_id in sorted(all_junction_ids):
            in_edges = incoming.get(junction_id, [])
            out_edges = outgoing.get(junction_id, [])
            for in_e in in_edges:
                # Disallow direct U-turn (from_node of in_e == to_node of out_e
                # means the out edge goes back where we came from) -- except
                # when that is the *only* physically possible continuation.
                # Real road networks contain genuine dead ends / cul-de-sacs
                # (a real vehicle that drives in must be able to turn around
                # and drive back out); forbidding the U-turn there would
                # strand every vehicle that enters via in_e with zero legal
                # moves, which a synthetic small fixture never exercises but
                # a real OSM extract does.
                allowed_out = [
                    out_e for out_e in out_edges if in_e.from_node != out_e.to_node
                ]
                if not allowed_out and out_edges:
                    allowed_out = out_edges
                for out_e in allowed_out:
                    ET.SubElement(
                        root,
                        "connection",
                        attrib={
                            "from": self.edge_mapping.get(in_e.edge_id, in_e.edge_id),
                            "to": self.edge_mapping.get(out_e.edge_id, out_e.edge_id),
                            "fromLane": "0",
                            "toLane": "0",
                        },
                    )

        path = self.output_dir / "connections.con.xml"
        _write_xml(root, path)
        return path

    # ------------------------------------------------------------------
    # netconvert invocation
    # ------------------------------------------------------------------

    def _run_netconvert(
        self,
        nod_path: Path,
        edg_path: Path,
        con_path: Path,
        net_path: Path,
    ) -> None:
        """
        Run netconvert to compile .nod.xml + .edg.xml + .con.xml -> net.xml.

        Raises RuntimeError on non-zero exit or missing executable.
        """
        if NETCONVERT_EXE is None:
            raise RuntimeError(
                "netconvert executable not found. Install SUMO, add its bin "
                "directory to PATH, or set SUMO_HOME to the SUMO root directory."
            )

        cmd = [
            str(NETCONVERT_EXE),
            "--no-internal-links", "true",
            "--node-files", str(nod_path),
            "--edge-files", str(edg_path),
            "--connection-files", str(con_path),
            "--output-file", str(net_path),
            # Keep coordinates as-is (synthetic Cartesian metres)
            "--offset.disable-normalization", "true",
            # Don't remove any edges or geometry — synthetic network is small
            "--edges.join", "false",
            # Log verbosely to file for debugging
            "--log", str(self.output_dir / "netconvert.log"),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"netconvert failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}\n"
                f"Check {self.output_dir / 'netconvert.log'} for details."
            )

    # ------------------------------------------------------------------
    # Mapping persistence
    # ------------------------------------------------------------------

    def _write_mapping_json(self) -> None:
        """Persist edge/node mappings alongside the net.xml for other modules."""
        mapping = {
            "edge_mapping": self.edge_mapping,
            "node_mapping": self.node_mapping,
            "lane_mapping": {e.edge_id: [f"{e.edge_id}_{i}" for i in range(e.lane_count)] for e in self.scenario.edges},
            "sumo_home": (
                str(Path(os.environ["SUMO_HOME"]).resolve())
                if os.environ.get("SUMO_HOME")
                else (
                    str(NETCONVERT_EXE.parent.parent)
                    if NETCONVERT_EXE is not None
                    else None
                )
            ),
            "net_file": "net.xml",
        }
        path = self.output_dir / "sumo_mapping.json"
        path.write_text(json.dumps(mapping, indent=2))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _junction_type(signalized: bool) -> str:
        return "traffic_light" if signalized else "priority"

    @staticmethod
    def _road_priority(road_class: str) -> int:
        return {"arterial": 3, "collector": 2, "local": 1}.get(road_class, 1)


# ---------------------------------------------------------------------------
# Convenience loader for already-exported mappings
# ---------------------------------------------------------------------------

def load_mapping(output_dir: Path) -> dict:
    """Load a previously written sumo_mapping.json."""
    path = Path(output_dir) / "sumo_mapping.json"
    if not path.exists():
        raise FileNotFoundError(f"sumo_mapping.json not found in {output_dir}")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_xml(root: ET.Element, path: Path) -> None:
    """Pretty-print an ElementTree to a file with XML declaration."""
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    with open(path, "wb") as fh:
        tree.write(fh, xml_declaration=True, encoding="utf-8")