from copy import deepcopy

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import auth, server
from src.platform.service import PlatformService


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, 'config', lambda name, default='': default)
    return TestClient(server.app)


@pytest.fixture
def graph():
    return PlatformService().graph('S3_BASE')


def test_snapshot_routes_and_replay_use_catalog_inputs(client, graph):
    response = client.post('/api/workspace/solve', json={'graph': graph, 'method': 'qpso', 'particles': 4, 'evaluations': 12})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['evaluation']['feasible']
    assert result['evaluation']['distance_m'] == 1400
    assert result['trace']['best']
    replay = client.post('/api/workspace/replay', json={'graph': graph, 'plan': result['plan']})
    assert replay.status_code == 200, replay.text
    assert replay.json()['mode'] == 'kinematic_mock'
    assert replay.json()['episode']['delivered_count'] == len(graph['requests'])


def test_traffic_closures_and_changed_fleet_are_computed(client, graph):
    payload = {'graph': graph, 'source': 'N0', 'target': 'N1'}
    first = client.post('/api/workspace/path', json=payload).json()
    slow = client.post('/api/workspace/path', json={**payload, 'traffic_factor': .5}).json()
    assert slow['time_s'] == pytest.approx(first['time_s'] * 2)
    closed = client.post('/api/workspace/path', json={**payload, 'closed_edge_ids': [first['edge_ids'][0]]}).json()
    assert not closed['feasible'] or first['edge_ids'][0] not in closed['edge_ids']
    changed = deepcopy(graph)
    for vehicle in changed['fleet']:
        vehicle['capacity'] = .5
    infeasible = client.post('/api/workspace/solve', json={'graph': changed, 'method': 'constructive'}).json()
    assert not infeasible['evaluation']['feasible']
    original = client.post('/api/workspace/solve', json={'graph': graph, 'method': 'constructive'}).json()
    assert original['evaluation']['feasible']  # no cross-request cache contamination


def test_compare_and_invalid_snapshots(client, graph):
    comparison = client.post('/api/workspace/compare', json={'graph': graph, 'methods': ['qpso', 'pso', 'alns'], 'particles': 4, 'evaluations': 12})
    assert comparison.status_code == 200, comparison.text
    assert {row['method'] for row in comparison.json()['rows']} == {'QPSO', 'PSO', 'ALNS'}
    assert all(row['feasible'] for row in comparison.json()['rows'])
    for bad in ({'closed_edge_ids': ['missing']}, {'traffic_factor': 0}, {'particles': 400}):
        assert client.post('/api/workspace/solve', json={'graph': graph, **bad}).status_code == 422
    graph['requests'][0]['node'] = 'missing'
    assert client.post('/api/workspace/validate', json={'graph': graph}).status_code == 422


def test_company_api_access_is_fail_closed(client, graph, monkeypatch):
    monkeypatch.setattr(server, 'config', lambda name, default='': 'https://example.supabase.co' if name == 'SUPABASE_URL' else default)
    payload = {'graph': graph}
    assert client.post('/api/workspace/validate', json=payload).status_code == 401
    assert client.get('/api/scenarios').status_code == 200
    assert client.post('/api/workspace/validate', json=payload, headers={'X-Quanta-Demo': 'true'}).status_code == 200
    headers = {'Authorization': 'Bearer test', 'X-Company-Id': 'company'}
    monkeypatch.setattr(server, 'verify_company', lambda *_: 'viewer')
    assert client.post('/api/workspace/validate', json=payload, headers=headers).status_code == 403
    monkeypatch.setattr(server, 'verify_company', lambda *_: 'dispatcher')
    assert client.post('/api/workspace/validate', json=payload, headers=headers).status_code == 200
    assert client.post('/api/models/forecaster/train', json={}, headers=headers).status_code == 403
    monkeypatch.setattr(server, 'config', lambda name, default='': 'false' if name == 'QUANTA_ALLOW_DEMO' else 'https://example.supabase.co' if name == 'SUPABASE_URL' else default)
    assert client.post('/api/workspace/validate', json=payload, headers={'X-Quanta-Demo': 'true'}).status_code == 401


def test_membership_is_verified_for_the_authenticated_user(monkeypatch):
    paths = []
    def response(path, token):
        paths.append(path)
        assert token == 'test'
        return {'id': 'user-a'} if path.startswith('/auth/') else []
    monkeypatch.setattr(auth, 'supabase_json', response)
    with pytest.raises(HTTPException) as error:
        auth.verify_company('Bearer test', 'company-b')
    assert error.value.status_code == 403
    assert 'company_id=eq.company-b' in paths[1] and 'user_id=eq.user-a' in paths[1]


def test_spa_deep_link_and_missing_api(client):
    assert client.get('/app/results/example').status_code == 200
    assert client.get('/api/does-not-exist').status_code == 404
