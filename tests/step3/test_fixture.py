from __future__ import annotations
import json
from pathlib import Path
import pytest
from src.routing.validator import evaluate_scenario, shortest_directed_path

ROOT=Path(__file__).resolve().parents[2]; FIXTURE=ROOT/'fixtures'/'step3'
def load(case): return json.loads((FIXTURE/'cases'/case/'scenario.json').read_text())
REFERENCE={'V1':['J1','J2','J3'],'V2':['J4','J5']}

def test_reference_solution():
 r=evaluate_scenario(load('feasible_reference'),REFERENCE)
 assert r.feasible and r.capacity_feasible and r.connectivity_feasible and r.time_window_feasible and r.all_requests_served
 v1,v2=r.vehicles['V1'],r.vehicles['V2']
 assert (v1.load,v1.driving_time_s,v1.waiting_time_s,v1.service_time_s,v1.elapsed_time_s)==pytest.approx((7,70,10,35,115))
 assert v1.service_starts_s=={'J1':10.0,'J2':35.0,'J3':60.0}
 assert (v2.load,v2.driving_time_s,v2.waiting_time_s,v2.service_time_s,v2.elapsed_time_s)==pytest.approx((3,70,40,20,130))
 assert v2.service_starts_s=={'J4':80.0,'J5':100.0}

def test_early_arrival_waiting():
 r=evaluate_scenario(load('early_arrival_wait'),REFERENCE); assert r.feasible
 assert r.vehicles['V1'].waiting_time_s==pytest.approx(10); assert r.vehicles['V2'].waiting_time_s==pytest.approx(40)
 assert r.vehicles['V1'].service_starts_s['J2']==pytest.approx(35); assert r.vehicles['V1'].service_starts_s['J3']==pytest.approx(60); assert r.vehicles['V2'].service_starts_s['J4']==pytest.approx(80)

def test_excess_load_only_fails_capacity():
 r=evaluate_scenario(load('excess_load'),{'V1':['J1','J2','J3','J4'],'V2':['J5']})
 assert not r.feasible and not r.capacity_feasible and r.connectivity_feasible and r.time_window_feasible
 assert r.vehicles['V1'].load==pytest.approx(13)

def test_missed_window_fails_time_window():
 r=evaluate_scenario(load('missed_window'),REFERENCE); assert not r.feasible and r.capacity_feasible and r.connectivity_feasible and not r.time_window_feasible
 assert r.vehicles['V1'].service_starts_s['J3']==pytest.approx(75)

def test_reverse_traversal_is_not_inferred():
 assert shortest_directed_path(load('wrong_way'),'N7','N2') is None

def test_closure_selects_detour():
 base=load('feasible_reference'); closed=load('closure_with_detour')
 assert shortest_directed_path(base,'N1','N2')==(('E12',),pytest.approx(10))
 assert shortest_directed_path(closed,'N1','N2')==(('E16','E62'),pytest.approx(30))
 r=evaluate_scenario(closed,REFERENCE); assert r.feasible
 assert r.vehicles['V1'].service_starts_s['J2']==pytest.approx(50); assert r.vehicles['V1'].service_starts_s['J3']==pytest.approx(70)
 assert r.vehicles['V1'].legs[1].edge_ids==('E16','E62')

def test_v12_schema_fields_are_present():
 sc=load('feasible_reference')
 assert sc['schema_version']=='1.2'; assert 'configuration_version' in sc and 'config_version' not in sc
 assert set(sc['coordinate_transform'])=={'scale','translation_x_m','translation_y_m','description'}
 assert all('access_node_id' in r and 'earliest_service_start_s' in r for r in sc['requests'])
 assert all('start_node_id' in v and 'depot_node_id' in v and 'executed_prefix_edge_ids' in v for v in sc['fleet'])
 assert all('lane_count' in e and 'parent_road_id' in e and 'open_by_default' in e for e in sc['edges'])

def test_wrong_way_has_no_reverse_edge():
 sc=load('wrong_way'); assert not any(e['from_node']=='N7' and e['to_node']=='N2' for e in sc['edges'])

def test_closure_edge_is_closed_in_case():
 sc=load('closure_with_detour'); assert next(e for e in sc['edges'] if e['edge_id']=='E12')['open_by_default'] is False
