import csv
from src.sim.causal_logger import CausalSumoLogger
from src.sim.traci_adaptor import SimStepOutput, EdgeStepData, VehicleStepData
from src.platform.catalog import load_scenario
from src.data.dynamic_episodes import DynamicEpisodeConfig

def test_completed_minute_never_contains_later_measurements(tmp_path):
    sc=load_scenario("S3_BASE"); edge=sc.edges[0]
    logger=CausalSumoLogger(sc)
    for t,speed,count in [(1,2,1),(60,4,1),(61,99,1)]:
        logger.on_step(SimStepOutput(t,edges=[EdgeStepData(edge.edge_id,speed,0.1, count,0)]))
    logger.write(tmp_path,episode_id="ep",config=DynamicEpisodeConfig())
    with (tmp_path/"observations.csv").open() as f:
        rows=list(csv.DictReader(f))
    assert {r["observation_time_s"] for r in rows}=={"60"}
    assert float(next(r for r in rows if r["edge_id"]==edge.edge_id)["observed_speed_mps"])==3
    assert next(r for r in rows if r["edge_id"]==sc.edges[1].edge_id)["missing"]=="1"

def test_arrival_records_final_edge_and_empty_lane_is_missing(tmp_path):
    sc=load_scenario("S3_BASE"); edge=sc.edges[0]
    logger=CausalSumoLogger(sc)
    logger.on_step(SimStepOutput(1,vehicles=[VehicleStepData("v",edge.edge_id,edge.edge_id+"_0",0,100,10,0,False)]))
    logger.on_step(SimStepOutput(60,arrived_vehicle_ids=["v"],edges=[EdgeStepData(edge.edge_id,10,0,0,0)]),known_closed_edges={edge.edge_id})
    assert logger.trajectories[0]["exit_time_s"]==60
    logger.write(tmp_path,episode_id="ep",config=DynamicEpisodeConfig())
    with (tmp_path/"observations.csv").open() as f:
        row=next(r for r in csv.DictReader(f) if r["edge_id"]==edge.edge_id)
    assert row["missing"]=="1" and row["known_closed"]=="1"
