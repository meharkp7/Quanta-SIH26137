from __future__ import annotations
import json
from pathlib import Path
import pytest
from src.contracts.scenario import Scenario

PROJECT_ROOT=Path(__file__).resolve().parents[1]
FIXTURE_ROOT=PROJECT_ROOT/'fixtures'/'step3'

def load_step3_case(case:str)->dict:
    return json.loads((FIXTURE_ROOT/'cases'/case/'scenario.json').read_text(encoding='utf-8'))

@pytest.fixture
def scenario()->Scenario:
    return Scenario.model_validate(load_step3_case('feasible_reference'))
