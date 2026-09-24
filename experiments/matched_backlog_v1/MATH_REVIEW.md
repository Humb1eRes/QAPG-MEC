# Revised QAPG objective and block solver: independent mathematical review

## Material Passport

- Date: 2026-09-22.
- Scope: the proposed queue-prediction objective, service-variable conversion, device/server updates, and the boundaries of claims supported by those updates.
- Status: **PASS** for the analytic derivation and independent implementation checks recorded in `QAPG_REVISED_MATH_VALIDATION.json`. This review does not infer empirical performance.
- Inputs: the existing common simulator's queue timing and hardware/energy model; the proposed objective supplied before new parameter-selection or held-out results were opened.
- Independence: this reviewer did not implement or modify the production controller. Numerical checks use direct objective evaluation and derivative-free scalar minimization rather than the controller's root formulas.

## 1. Units and feasible service variables

All lower-case workload variables below are in Mbit: device queues `q`, server queues `h`, local completions `l`, uploads `u`, and edge completions `d`. A device uploads to at most one server; let `Y_m` denote all uploads associated with server `m`.

The physical constraints are

```
0 <= l_n <= tau*fmax/(cycles_per_bit*1e6),
0 <= u_n <= tau*bandwidth*log2(1+pmax*g_nm/noise)/1e6,
l_n+u_n <= q_n,
0 <= d_m <= min(h_m, tau*Fmax/(edge_cycles_per_bit*1e6)).
```

The null association has `u_n = p_n = 0`. A zero-gain or zero-power link cannot upload. The `d <= h` constraint is essential: the existing simulator cannot process newly uploaded work in the same slot. Likewise, this slot's external arrivals cannot be served before they enter the device queue.

For an exactly provisioned service amount, `f = cycles_per_bit*1e6*l/tau`. The existing full-slot cubic energy becomes

```
E_device(l) = Kd*l^3, Kd = kappa_device*cycles_per_bit^3*1e18/tau^2,
E_edge(d)   = Ke*d^3, Ke = kappa_edge*edge_cycles_per_bit^3*1e18/tau^2.
```

For a positive channel, the least full-slot transmit power delivering `u` is

```
p(u) = (noise/g)*(exp(k*u)-1),
E_radio(u) = tau*p(u),
k = log(2)*1e6/(tau*bandwidth).
```

These are identities for the common physical model, not a different energy model. With Lyapunov workload terms converted from bit-squared to Mbit-squared, the physical energy weight is `v = V/1e12`.

## 2. Chosen normalization and exact conditional expected Lyapunov identity

At the decision instant, let the observed information include the current queues, current channel gains, configuration, and past history. Let `Abar_n` be the conditional mean of this slot's arrival in Mbit. In the declared independent Poisson arrival process,

```
Abar_n = beta*arrival_lambda*task_packet_bits/1e6.
```

Because service is capped by the available old work, the next queues have no active clipping:

```
q'_n = q_n - l_n - u_n + A_n,
h'_m = h_m - d_m + Y_m.
```

The revised objective chooses the hardware-derived constant `w = M/N > 0` and the weighted Lyapunov function `L_w(q,h) = 0.5*sum(q^2)+0.5*w*sum(h^2)`. This is a **chosen nominal-population normalization**. If a server hypothetically holds `k = N/M` equal origin queues, their squared sum is `H^2/k = w*H^2`. The actual ledger need not have exactly `k` contributing origins or equal queues; past reassociation can leave work from more origins at a server. Therefore `w*H^2` is not generally the sum of actual per-origin squares and is not an identity for total aggregate backlog. `w` was fixed from the device/server counts before the new trajectory experiments, not fitted from their outcomes.

For this explicitly weighted `L_w`,

```
E[L_w(q',h') + v*E_total | observed information]
 = J + 0.5*sum(Var(A_n | observed information)),

J = 0.5*sum((q+Abar-l-u)^2)
  + 0.5*w*sum((h-d+Y)^2)
  + v*[sum(Kd*l^3) + sum(E_radio(u)) + sum(Ke*d^3)].
```

The conditional expected drift adds the further action-independent term `-L_w(q,h)`. Arrival independence between devices is not needed for this sum-of-squares identity; only the correct conditional means/variances and their independence from the chosen actions are needed. For the current Poisson model the variance is `beta*arrival_lambda*(task_packet_bits/1e6)^2` per device.

The mean-arrival parameter is common configuration information. The revised controller can use that known mean, but must not receive this slot's realized end-of-slot arrivals or any future arrivals/channels. If arrivals later become state-dependent, the unconditional configured mean no longer automatically makes this identity exact.

This is a one-step expected objective. Minimizing it does not establish minimum long-run energy under a backlog bound, a stochastic stability theorem, or global optimality of a finite-sweep mixed association solver.

## 3. Exact conditional device subproblem

Fix all other devices and all edge-service amounts. For a candidate server `m`, let `C = h_m-d_m+Y_m_without_n` and `s = q_n+Abar_n`. Subtracting terms independent of this device leaves

```
F_m(l,u) = 0.5*(s-l-u)^2 + v*Kd*l^3 + v*E_radio_nm(u)
           + w*(C*u + 0.5*u^2).
```

The `w*(C*u + 0.5*u^2)` term follows exactly from expanding the server's weighted next-queue square. It accounts for other users' current uploads inside the candidate optimization. It is not an arbitrary additional congestion coefficient.

For fixed `u`, the unconstrained local optimum solves `3*v*Kd*l^2+l=s-u`. For positive `v*Kd` its stable nonnegative root is

```
l_root = 2*(s-u)/(1+sqrt(1+12*v*Kd*(s-u))).
l*(u) = min(Lmax, q-u, l_root).
```

For `v*Kd = 0`, the root is `s-u`. The feasible upload interval is `0 <= u <= min(q,Umax_nm)`.

For the reduced scalar objective `f_m(u) = F_m(l*(u),u)`, the derivative is

```
f'_m(u) = w*(C+u)+v*E'_radio(u)-3*v*Kd*l^2,
          if the available-task bound l=q-u is active;

f'_m(u) = -s+l+u+w*(C+u)+v*E'_radio(u),
          otherwise (interior l or fixed hardware-cap l).
```

The first expression accounts for `dl/du=-1` on the task bound. The second uses either the envelope condition or a constant local hardware cap. At a transition between a free local optimum and the task bound, the two expressions coincide by local stationarity. Endpoints and ties require deterministic feasible handling.

The workload part has Hessian `[[1,1],[1,1+w]]` (determinant `w > 0`), which is positive definite. The energy terms are convex for nonnegative weights. Consequently the device subproblem is strictly convex over its linear feasible set, and minimizing this scalar convex reduction gives its unique continuous optimum for each server. Enumerating all servers and the null action yields an exact conditional best response, apart from numerical tolerance.

## 4. Edge-service block

Fix all uploads and local actions. Each server independently minimizes

```
G_m(d) = 0.5*w*(h_m+Y_m-d)^2 + v*Ke*d^3,
0 <= d <= min(h_m,Dmax).
```

Its stationary equation is `3*v*Ke*d^2+w*d=w*(h_m+Y_m)`; dividing by `w` gives the same stable root formula with cubic coefficient `v*Ke/w`, followed by the old-task and hardware caps. The objective is strictly convex in `d`, so this is the exact conditional edge optimum.

Although `Y` raises the marginal value of freeing old server work, it does not increase executable work this slot: the upper bound remains `h`, not `h+Y`.

## 5. Potential identity and what convergence means

For any feasible unilateral device change, direct expansion gives

```
J(new)-J(old) = F_new_server(l_new,u_new)-F_old_server(l_old,u_old),
```

where both server residuals exclude the moving device, and the null action contributes its own local-only cost. All other server/device terms cancel. Edge updates have the analogous exact scalar difference. Defining each block's utility as minus its conditional contribution therefore makes `-J` an exact potential for these decision blocks.

Sequential strictly improving block updates cannot increase `J`; `J >= 0` is a lower bound. For a fixed association, the full continuous objective is convex. The union across discrete associations is not a single convex feasible set, so block improvements can stop at a local operating point.

There are continuously many service/power choices. Therefore the old finite-profile argument for fixed-candidate association cannot be reused to prove finite termination. A fixed 12-sweep cap guarantees bounded computational work only. The controller should expose the number of sweeps, whether its stopping criterion was met, and the largest remaining unilateral objective improvement. A positive residual at the cap must remain visible. Even convergence to a coordinatewise optimum does not establish the best joint association or the best long-run policy.

## 6. Interpretation for the matched-backlog comparison

The new objective removes avoidable excess CPU provisioning and jointly adjusts local computation, radio service, server assignment, and old-work edge service. These are substantive algorithm changes; the resulting controller must be labelled as a revision, not a same-code rerun of SR3.

All baselines must receive the same service-preserving CPU cap and retain their source-identifiable decision rules. Equal configured `V` alone does not imply equal backlog, hence the declared complete parameter grid, separate selection seeds, and untouched held-out seeds are needed for the requested energy comparison under the same backlog requirement. The mathematical checks support correctness of the proposed updates, not a predetermined performance ranking.

## 7. Independent numerical evidence

Run `test_qapg_revised_math.py` with NumPy 2.3.5. The recorded production hashes are Python `fabc01182fe8debae12ad46327f1811f7e6bc25334bec02d9f1d531f0894c420` and C++ `a299cf3daf46b7a4920b48a1466e20ba05371e76a0ebbdacabeb1ac55c40c9df`.

- Forty service/power samples reproduce the full-slot physical energy formulas.
- Twenty-eight one-device states with all server alternatives, zero old edge work, hardware/task limits, closed links and several physical energy weights agree with an independent nested derivative-free oracle. Maximum objective difference: `1.39e-17`.
- Twelve multi-user states reproduce the initialization and final global objective, every conditional edge optimum, physical bounds, origin accounting and every possible device/server deviation. Maximum final-objective difference: `2.23e-16`; maximum unilateral-residual difference: `2.50e-16`.
- A separate interacting-upload check includes the actual `N=100, M=10` population normalization. An intentionally imposed one-sweep budget leaves independently measured improvement `1.31e-3` and is correctly marked nonconverged; restoring the declared 12-sweep budget leaves `9.38e-11` and is correctly marked converged. Both outcomes are retained in the validation JSON. The production constant was not edited: the test changed its in-process value and restored it in a `finally` block.
- A complete four-outcome arrival distribution verifies the conditional expected Lyapunov identity including the variance constant without Monte Carlo approximation.
- One hundred twenty arbitrary feasible unilateral changes, including server switches and null actions, reproduce the exact weighted-potential difference within `4.03e-16`.
- Identical observed states with different configuration seeds produce identical decisions. The controller API receives current queues/current gains and the configured mean, not realized current or future arrivals.

The special one-user case is globally solved for validation because its empty old edge queues fix edge service at zero. This does not promote the general multi-user mixed association solver to a globally optimal algorithm.
