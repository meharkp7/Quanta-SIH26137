# Step 3 — Canonical Five-Job / Two-Vehicle Fixture

Schema baseline: Contracts V1.2.

This fixture is a correctness/regression fixture, not performance evidence.

## Base network

Nodes:
N0 depot; N1..N5 customer access points; N6 detour junction; N7 one-way dead-end.

Directed edges:
E01 N0->N1 100 m / 10 s
E10 N1->N0 100 m / 10 s
E12 N1->N2 100 m / 10 s (closeable)
E23 N2->N3 100 m / 10 s
E34 N3->N4 100 m / 10 s
E45 N4->N5 100 m / 10 s
E51 N5->N1 100 m / 10 s
E16 N1->N6 150 m / 15 s
E62 N6->N2 150 m / 15 s
E27 N2->N7 80 m / 8 s

Reverse traversal is never implied. In particular N2->N1 and N7->N2 are absent.

## Requests

J1: N1, demand 2, service 10 s, window [0,30]
J2: N2, demand 2, service 10 s, window [35,60]
J3: N3, demand 3, service 15 s, window [60,90]
J4: N4, demand 2, service 10 s, window [80,120]
J5: N5, demand 1, service 10 s, window [100,140]

All requests are known/released at t=0.

## Fleet

V1: depot N0, capacity 7
V2: depot N0, capacity 6

## Reference solution

V1 serves J1,J2,J3. Load=7. Driving=70 s, waiting=10 s, service=35 s, elapsed=115 s.
Service starts: J1=10, J2=35, J3=60.

V2 serves J4,J5. Load=3. Driving=70 s, waiting=40 s, service=20 s, elapsed=130 s.
Service starts: J4=80, J5=100.

## Required derived cases

1. feasible_reference — PASS
2. early_arrival_wait — PASS; J2/J3/J4 wait 5/5/40 s
3. excess_load — FAIL; V1 load 9 > 7
4. missed_window — FAIL; E23=30 s, J3 latest=65 s, service starts at 75 s
5. wrong_way — FAIL; N7->N2 has no legal directed path
6. closure_with_detour — PASS; E12 closed, replace [E12] with [E16,E62], increasing N1->N2 travel from 10 s to 30 s

## Acceptance

The independent evaluator must reproduce these expected outcomes without special-casing scenario IDs.
