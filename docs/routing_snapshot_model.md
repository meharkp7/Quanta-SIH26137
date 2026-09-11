# Snapshot Routing Evaluation Model

## 1. Scope

This model defines the deterministic snapshot evaluation problem used by
`RouteEvaluator`.

It is a feasibility-and-objective model for a fixed routing snapshot. It is
not claimed to be an exact representation of the complete dynamic operating
day.

Dynamic traffic enters through the `CostView`, while vehicle state and
execution commitments enter through the commitment snapshot.

---

## 2. Sets

Let:

- \(V\) be the set of vehicles.
- \(R\) be the set of customer requests.
- \(N\) be the set of directed road-network nodes.
- \(E\) be the set of directed road edges.

For vehicle \(v \in V\):

- \(s_v\) is its current/start node.
- \(d_v\) is its depot/end node.
- \(Q_v\) is its capacity.
- \(q_v^0\) is its current committed load.
- \(t_v^0\) is its current evaluation time.

For request \(i \in R\):

- \(n_i\) is its road access node.
- \(q_i\) is its demand.
- \(r_i\) is its release time.
- \([a_i,b_i]\) is its service-start window.
- \(p_i\) is its service duration.

---

## 3. Logical route

Each vehicle receives an ordered customer sequence

\[
\pi_v=(i_1,i_2,\ldots,i_k).
\]

The evaluator maps every consecutive logical transition

\[
(s_v,i_1),\,
(i_1,i_2),\ldots,\,
(i_k,d_v)
\]

to a legal directed physical road path.

The evaluator does not optimize this sequence.

---

## 4. Dynamic physical path cost

For a directed edge \(e\) entered at time \(t\), the active
`CostView` provides:

\[
\tau_e(t) \ge 0
\]

for travel time and

\[
\ell_e \ge 0
\]

for distance.

For a physical path

\[
P=(e_1,\ldots,e_m)
\]

starting at \(t_0\), traversal time is propagated sequentially:

\[
t_{j+1}=t_j+\tau_{e_j}(t_j).
\]

Therefore,

\[
T(P,t_0)=t_m-t_0.
\]

Distance is

\[
D(P)=\sum_{j=1}^{m}\ell_{e_j}.
\]

Distance and travel time always belong to the same selected physical path.

---

## 5. Service timing

For customer \(i\), let \(A_i\) denote arrival time.

The earliest legal service start is

\[
S_i^{min}=\max(r_i,a_i).
\]

If waiting is permitted,

\[
S_i=\max(A_i,S_i^{min}).
\]

Waiting time is

\[
W_i=\max(0,S_i^{min}-A_i).
\]

Service completion is

\[
C_i=S_i+p_i.
\]

If waiting is disabled, an arrival before \(S_i^{min}\) is infeasible.

---

## 6. Time-window feasibility

A customer is time-window feasible iff

\[
a_i-\epsilon_t
\le S_i
\le b_i+\epsilon_t.
\]

Lateness is

\[
L_i=\max(0,S_i-b_i).
\]

Release feasibility is

\[
S_i\ge r_i-\epsilon_t.
\]

Here \(\epsilon_t\) is the configured time tolerance.

---

## 7. Capacity propagation

The vehicle begins with committed load

\[
q_v^0.
\]

For a sequence of newly serviced requests,

\[
q_{v,j}
=
q_v^0+\sum_{h=1}^{j}q_{i_h}.
\]

The maximum route load is

\[
Q_v^{max}=\max_j q_{v,j}.
\]

Capacity is feasible iff

\[
Q_v^{max}\le Q_v+\epsilon_q.
\]

The evaluator does not infer pickup/delivery semantics that are absent from
the scenario contract. The committed current load is therefore authoritative.

---

## 8. Commitment constraints

A commitment snapshot may specify:

- current vehicle node;
- current time;
- current load;
- onboard request IDs;
- frozen physical edge prefix;
- committed customer prefix.

If commitment enforcement is enabled, the candidate logical route must
preserve the committed customer prefix:

\[
\pi_v[1:k_c]=\pi_v^{commit}.
\]

If a candidate explicitly supplies physical edges, its frozen prefix must
match the committed frozen edge sequence.

A rolling route that begins after an already-executed prefix may omit those
historical edges because the commitment snapshot already represents the
post-prefix state.

---

## 9. Directed connectivity

Every logical transition must have a legal directed physical path.

For transition \(u\rightarrow w\),

\[
P(u,w)\neq\varnothing.
\]

A closed edge cannot be entered.

The final vehicle transition is

\[
n_{i_k}\rightarrow d_v
\]

when depot return is required.

---

## 10. Assignment validity

A complete plan must satisfy:

1. every assigned request exists;
2. no request is assigned more than once;
3. required requests are served when
   `require_all_requests_served=True`;
4. vehicle IDs are known.

Partial assignments may be evaluated when completeness is disabled.

---

## 11. Objective

The base objective is

\[
J =
w_D D
+w_T T
+w_W W
+w_P P
+\lambda_L L
+\lambda_Q C_Q
+\lambda_C C_C
+\lambda_D C_D
+\lambda_U C_U
+\lambda_V C_V
\]

where:

- \(D\) = total physical distance;
- \(T\) = total travel time;
- \(W\) = total waiting time;
- \(P\) = total service time;
- \(L\) = total lateness;
- \(C_Q\) = capacity violation penalty;
- \(C_C\) = connectivity violation penalty;
- \(C_D\) = duplicate-request penalty;
- \(C_U\) = unserved-request penalty;
- \(C_V\) = unknown-vehicle penalty.

The configured evaluator weights determine the scalar value.

---

## 12. Feasibility

A vehicle route is feasible iff all enabled constraints hold:

\[
F_v =
F_v^{connectivity}
\land
F_v^{capacity}
\land
F_v^{time-window}
\land
F_v^{release}
\land
F_v^{commitment}.
\]

A complete route plan is feasible iff every vehicle route is feasible and
the global assignment constraints hold.

The evaluator reports these components independently rather than collapsing
all failures into one Boolean.

---

## 13. Snapshot versus dynamic operation

This model evaluates one routing snapshot.

A dynamic operating episode may change:

- road costs;
- closures;
- vehicle positions;
- vehicle loads;
- visible requests;
- forecasts;
- commitments.

Such changes create a new evaluation state/version.

The evaluator therefore does not claim that solving one snapshot constitutes
an exact optimization of the entire future dynamic day.