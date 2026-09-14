from dataclasses import replace
import numpy as np
import pytest
from src.learning.windows import ForecastWindow, _fill_features
from src.learning.loader import WindowDataset, write_fixture_tensors
from src.learning.scaling import FeatureScaler
from src.learning.schema import FEATURE_NAMES

def window(n=2, scenario="map-a"):
    x=np.ones((12,n,6),dtype=np.float32)
    return ForecastWindow("ep-a","train",660,tuple(range(0,720,60)),(960,1260,1560),
        tuple(f"edge-{i}" for i in range(n)),x,np.ones_like(x,dtype=bool),
        np.ones((n,3)),np.ones((n,3),dtype=bool),np.ones((n,3))*10,
        np.ones((n,3),dtype=bool),np.broadcast_to(np.array([960,1260,1560])[None,:,None],(n,3,2)).copy(),
        scenario,scenario+":v1")

def test_missing_speed_preserves_known_closure_and_age():
    x=np.full((1,1,6),np.nan); mask=np.zeros(x.shape,dtype=bool)
    _fill_features(x,mask,0,0,{"missing":"1","known_closed":"1","observation_age_s":"120"},10)
    assert x[0,0,3:].tolist()==[2,1,1]
    assert mask[0,0].tolist()==[False,False,False,True,True,True]

def test_absent_row_does_not_invent_open_road():
    x=np.full((1,1,6),np.nan); mask=np.zeros(x.shape,dtype=bool)
    _fill_features(x,mask,0,0,None,10)
    assert not mask[0,0,5]
    assert x[0,0,4]==1

def test_graph_padding_and_round_trip_metadata(tmp_path):
    a,b=window(),window(3,"map-b")
    scaler=FeatureScaler.fit(a.features[None],FEATURE_NAMES)
    dataset=WindowDataset([a,b],scaler=scaler)
    batch=dataset.batch()
    assert batch.shape==(2,12,3,6)
    assert batch.edge_padding_mask.tolist()==[[False,False,True],[False,False,False]]
    assert not batch.feature_mask[0,:,2].any()
    assert not batch.traversal_target_mask[0,2].any()
    assert np.all(batch.features[0,:,2]==0)
    with np.load(write_fixture_tensors(tmp_path,dataset),allow_pickle=False) as saved:
        for field in ("traversal_targets","traversal_target_mask","label_available_at_s",
                      "edge_padding_mask","edge_ids_by_sample","history_times_s","target_times_s",
                      "scenario_ids","graph_versions","episode_ids","issue_times_s"):
            np.testing.assert_equal(saved[field],np.asarray(getattr(batch,field)))

def test_reordered_same_map_is_rejected():
    a=window()
    with pytest.raises(ValueError,match="stable"):
        WindowDataset([a,replace(a,edge_ids=tuple(reversed(a.edge_ids)))])

def test_label_without_maturity_is_rejected():
    a=window()
    a.label_available_at_s[0,0,0]=0
    with pytest.raises(AssertionError,match="maturity"):
        WindowDataset([a])
