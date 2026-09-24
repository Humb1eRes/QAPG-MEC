# QAPG-R controller design

This is a new controller, separate from every previous manuscript controller,
simulator, and saved experiment. The comparison protocol controls the physical
environment and selects energy/backlog operating points outside this module.

## Objective fixed before trajectory experiments

Let q and h denote slot-start device and aggregate server queues in Mbit;
l, u, and d denote local execution, uploaded work, and old edge work executed.
Let Y be total newly uploaded work per server. With a known mean arrival a,
fixed energy coefficient v=V/10^12, and **w=M/N**, the controller minimizes

    J = 0.5 sum_n (q_n+a-l_n-u_n)^2
      + 0.5 w sum_m (h_m-d_m+Y_m)^2
      + v [ sum_n Kd l_n^3 + sum_n tau p_n + sum_m Ke d_m^3 ].

The first two terms are the decision-dependent expectation of a weighted
quadratic next-slot queue function; the arrival variance contributes an additive
constant. Current realized arrivals and future observations are not used.

The edge weight was changed from 1 to M/N **before any development, calibration,
or held-out trajectory run**. API smoke checks on isolated queue states had
already run, without performance comparisons. The design reason is that device
queues are per-origin whereas h aggregates approximately N/M origins: under
balanced per-origin queues, the sum of their squared queues equals h²/(N/M).
This is a fixed dimension-derived normalization, not a fitted value or a
hyperparameter selected to beat a comparator. It does not make the weighted
objective identical to the exact sum of per-origin squared edge queues when
origins are unbalanced. Protocol evaluations still measure actual unweighted
total backlog and energy identically for all algorithms.

Kd=kappa_device*cycles_per_bit^3*10^18/tau² and Ke uses the corresponding edge
constants. A chosen independent link satisfies p=(noise/g)[exp(k u)-1], where
k=ln(2)*10^6/(tau*bandwidth). Each device selects at most one server or null.
Resource bounds and l+u<=q are enforced jointly. Edge service obeys d<=h, so
new uploads cannot be executed in the same slot. CPU frequency never exceeds
the frequency needed for the chosen service; no idle processing capacity is
charged as executed work.

## Exact blocks and finite computational budget

Initialize all device associations to null, optimize local service, and optimize
edge service. Each cyclic sweep first optimizes all edge services for the current
uploads, then enumerates all server choices and null for each device. Given u,
the minimizing l is the positive root of 3vKd*l²+l=q+a-u clipped to both local
CPU capacity and q-u. The remaining one-dimensional convex problem in u is
solved by endpoint tests and at most 40 bisection steps (width tolerance 10^-10
Mbit). With C=h-d+Y_other, its derivative is

    w(C+u)+v*tau*(noise/g)*k*exp(k u)-3vKd*l²  if l=q-u is active;
    -(q+a)+l+u+w(C+u)+v*tau*(noise/g)*k*exp(k u) otherwise.

Edge service solves 3vKe*d²+w*d=w(h+Y), clipped to the old queue and CPU limit.
A device update requires an objective improvement greater than 10^-10 in scaled
units. Exact full objective values are recomputed after each accepted update
and edge update. Ties are deterministic; server enumeration begins at null.

The budget is 12 sweeps followed by a final exact edge update. A fresh full
unilateral-deviation scan reports the largest remaining device-block gain. A
convergence flag is true only if that gain is at most 10^-10. Objective decrease
does not establish global optimality, a global stochastic guarantee, or a Nash
equilibrium when the budget is exhausted. The arrival mean, edge normalization,
sweep budget, and tolerance are fixed; this module exposes no performance-tuned
adaptation coefficients.

## Implementation and checks

`qapg_revised_decision(Q,H,B,gains,cfg)` returns an `EEDODecision` in the common
simulator's original units. The C++17 source performs numerical optimization;
the Python wrapper validates inputs, compiles a source-hashed shared library,
and returns the per-origin accounting allocation. Compilation uses `-O3` without
`-ffast-math`. Diagnostic fields report objective decrease, convergence, sweep
count, accepted updates, and remaining deviation. They do not affect control.

`test_qapg_revised_api.py` checks feasibility, zero-work/channel/power/capacity
cases, immutable inputs, deterministic decisions, available-work CPU limits,
and the common physical ledger. Independent mathematical checks are provided
separately by the algorithm review. Any controller source change invalidates
source-hashed validation and requires revalidation before formal experiments.
