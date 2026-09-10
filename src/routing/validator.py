"""Independent Step 3 V1.2 routing and feasibility evaluator."""
from __future__ import annotations
from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf
from typing import Any, Mapping, Sequence

EPS=1e-9
@dataclass(frozen=True)
class LegResult:
    from_node:str; to_node:str; edge_ids:tuple[str,...]; travel_time_s:float
@dataclass(frozen=True)
class VehicleEvaluation:
    vehicle_id:str; customer_order:tuple[str,...]; load:float; capacity:float
    capacity_feasible:bool; connectivity_feasible:bool; time_window_feasible:bool
    driving_time_s:float; waiting_time_s:float; service_time_s:float; elapsed_time_s:float
    service_starts_s:Mapping[str,float]; service_ends_s:Mapping[str,float]; legs:tuple[LegResult,...]
@dataclass(frozen=True)
class EvaluationResult:
    feasible:bool; capacity_feasible:bool; connectivity_feasible:bool; time_window_feasible:bool
    all_requests_served:bool; vehicles:Mapping[str,VehicleEvaluation]
    unserved_request_ids:tuple[str,...]=(); disconnected_legs:tuple[tuple[str,str,str,str],...]=()
    window_violations:tuple[tuple[str,str,float,float],...]=()

def _field(obj:Any,name:str,legacy:str|None=None):
    if isinstance(obj,Mapping):
        return obj[name]
    return getattr(obj,name)

def shortest_directed_path(scenario:Any, source:str, target:str, closed_edge_ids:Sequence[str]=()):
    if source==target: return (),0.0
    outgoing={}
    for e in _field(scenario,'edges'):
        outgoing.setdefault(_field(e,'from_node'),[]).append(e)
    closed=set(closed_edge_ids); dist={source:0.0}; prev={}; heap=[(0.0,source)]
    while heap:
        d,u=heappop(heap)
        if d>dist.get(u,inf)+EPS: continue
        if u==target: break
        for e in sorted(outgoing.get(u,()),key=lambda x:_field(x,'edge_id')):
            if not _field(e,'open_by_default') or _field(e,'edge_id') in closed: continue
            w=_field(e,'length_m')/_field(e,'speed_limit_mps'); v=_field(e,'to_node'); nd=d+w
            if nd<dist.get(v,inf)-EPS:
                dist[v]=nd; prev[v]=(u,_field(e,'edge_id')); heappush(heap,(nd,v))
    if target not in dist: return None
    ids=[]; u=target
    while u!=source:
        u,eid=prev[u]; ids.append(eid)
    ids.reverse(); return tuple(ids),dist[target]

def evaluate_scenario(scenario:Any, plan:Mapping[str,Sequence[str]], *, closed_edge_ids:Sequence[str]=()):
    jobs={_field(j,'request_id'):j for j in _field(scenario,'requests')}
    vehicles={_field(v,'vehicle_id'):v for v in _field(scenario,'fleet')}
    assigned=[]; results={}; disconnected=[]; violations=[]
    for vid,v in vehicles.items():
        order=tuple(plan.get(vid,()))
        if any(j not in jobs for j in order): raise ValueError(f'Unknown request IDs in {vid}: {[j for j in order if j not in jobs]}')
        if len(order)!=len(set(order)): raise ValueError(f'Duplicate request in {vid}: {order}')
        load=sum(_field(jobs[j],'demand') for j in order); cap=_field(v,'capacity'); capok=load<=cap+EPS
        current=_field(v,'depot_node_id'); t=0.; driving=waiting=service=0.; starts={}; ends={}; legs=[]; conn=True; win=True
        for jid in order:
            j=jobs[jid]; target=_field(j,'access_node_id'); path=shortest_directed_path(scenario,current,target,closed_edge_ids)
            if path is None:
                conn=False; disconnected.append((vid,current,target,jid)); continue
            eids,travel=path; legs.append(LegResult(current,target,eids,travel)); driving+=travel; arrival=t+travel
            start=max(arrival,_field(j,'earliest_service_start_s')); waiting+=max(0.,_field(j,'earliest_service_start_s')-arrival)
            if start>_field(j,'latest_service_start_s')+EPS: win=False; violations.append((vid,jid,start,_field(j,'latest_service_start_s')))
            end=start+_field(j,'service_duration_s'); service+=_field(j,'service_duration_s'); starts[jid]=start; ends[jid]=end; t=end; current=target
        ret=shortest_directed_path(scenario,current,_field(v,'depot_node_id'),closed_edge_ids)
        if ret is None: conn=False; disconnected.append((vid,current,_field(v,'depot_node_id'),'__return__'))
        else:
            eids,travel=ret; legs.append(LegResult(current,_field(v,'depot_node_id'),eids,travel)); driving+=travel; t+=travel
        results[vid]=VehicleEvaluation(vid,order,load,cap,capok,conn,win,driving,waiting,service,t,starts,ends,tuple(legs)); assigned.extend(order)
    required=set(jobs); aset=set(assigned); allserved=aset==required and len(assigned)==len(aset)
    capok=all(v.capacity_feasible for v in results.values()); connok=all(v.connectivity_feasible for v in results.values()); winok=all(v.time_window_feasible for v in results.values())
    return EvaluationResult(capok and connok and winok and allserved,capok,connok,winok,allserved,results,tuple(sorted(required-aset)),tuple(disconnected),tuple(violations))
