from __future__ import annotations
import os
import json
from pathlib import Path
import pytest
# Torch and NumPy both use MKL on this Windows environment. Keep their thread
# pools deterministic when the optional Step 13 tests are collected alongside
# the Step 12 ridge baseline tests.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from src.contracts.scenario import Scenario

PROJECT_ROOT=Path(__file__).resolve().parents[1]
FIXTURE_ROOT=PROJECT_ROOT/'fixtures'/'step3'

def pytest_configure(config):
    # Isolate each run from Windows user-temp ACLs and concurrent runs.
    if config.option.basetemp is None:
        import tempfile
        root = PROJECT_ROOT / '.pytest_cache' / 'tmp'
        root.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = tempfile.mkdtemp(prefix='run-', dir=root)

def load_step3_case(case:str)->dict:
    return json.loads((FIXTURE_ROOT/'cases'/case/'scenario.json').read_text(encoding='utf-8'))

@pytest.fixture
def scenario()->Scenario:
    return Scenario.model_validate(load_step3_case('feasible_reference'))
