"""Real traffic dataset adapters.

Adapters normalize source-specific files into the canonical traffic layer.
"""

from .dcrnn_hdf5 import DCRNNHDF5Config, load_dcrnn_hdf5, load_metr_la, load_pems_bay
from .pems_npz import PEMSNPZConfig, load_pems_npz, load_pems04, load_pems08

__all__ = [
    "DCRNNHDF5Config",
    "PEMSNPZConfig",
    "load_dcrnn_hdf5",
    "load_metr_la",
    "load_pems_bay",
    "load_pems_npz",
    "load_pems04",
    "load_pems08",
]
