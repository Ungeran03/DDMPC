# Paper Revision Notes — MPC_Cabin_Climate_AEV (Model Revision July 2026)

**Audience:** Agent/author revising the manuscript `MPC_Cabin_Climate_AEV.pdf`.
**Status:** The model, the MPC formulation, and all three scenarios were fixed,
re-tuned, and re-run (July 2026). The headline savings changed from 51/65/70 %
to **38/56/51 %** — smaller, but now defensible. The paper text must be updated
throughout: numbers, the MPC formulation (Section 3.2), the energy model
(Section 2.4), and parts of the narrative.

---

## TL;DR — what changed and why

The previous results had three problems that a competent reviewer (or anyone
requesting the code) would have found:

1. **The MPC's horizon predictions were physically invalid.** The NLP declared
   every `Controlled` feature as free optimization variables over the horizon,
   including the PTC element temperature `T_ptc`, which had no predictor but
   fed the duct predictor with a 200 W/K coupling. The solver exploited this
   free variable: solutions showed the PTC element (a heater!) dropping to
   −19 °C, providing fictitious free cooling. IPOPT also failed to converge
   (`Maximum_Iterations_Exceeded`) on a substantial fraction of solves; the
   controller silently applied non-converged iterates.
2. **Blower fan power was not modeled.** The MPC ran the blower at 1.0
   permanently because it was free, while the PID's BlowerPI throttled it.
   A 300 W fan (≈150 Wh over 30 min at full speed) is on the order of the
   entire claimed MPC consumption — omitting it inflated savings by roughly
   15–25 percentage points.
3. **The paper's formulation (Eqs. 14–17) was not the implemented one.**
   Dead-band comfort, passenger-dependent weighting, the 800 ppm CO2 soft
   target, and the SOC weight factor 0.2 did not exist in the code.

All three are fixed. Savings are now 38–56 % with honest fan power, equal
comfort tracking (S1/S2), 100 % IPOPT convergence, and a code base whose
objective function matches what the paper will state.

---

## 1. Model and code fixes (code side, already done)

### 1.1 NLP structure (the critical fix)

- **Reduced MPC model** (`mpc_builder.py`): the NLP now contains only the
  predicted states (T_cabin, T_vent, CO2 + their change connections), the
  active MVs, and the forecast disturbances. `T_mass`, `P_hvac`, `T_ptc`
  and other dangling features are no longer free optimization variables.
- **PTC element removed from the cooling-mode duct predictor**
  (`s3_T_cabin_WB.py`): with τ = C_ptc/h_ptc = 7.5 s ≪ Δt = 60 s the element
  is quasi-stationary. The heating-mode predictor injects `u_ptc·Q_ptc_max`
  directly; a stable exact-discretization T_ptc predictor exists for
  diagnostics. (The old explicit-Euler T_ptc predictor had discrete
  eigenvalue −7 and made the NLP unsolvable — that is *why* it had been left
  out, which in turn created the free-variable hole.)
- **T_vent physical bounds [5, 65] °C** added as NLP constraint (matches the
  simulator's evaporator-icing/safety clip).
- **Smooth fresh-air fraction:** the predictor uses `f_fresh = 1 − u_recirc`
  with the box constraint `u_recirc ≤ 0.9` instead of the kinked
  `max(0.1, 1 − u_recirc)`. Equivalent within the admissible range, but the
  energy-optimal solution sits exactly at the kink and IPOPT stalled on it.
- **Result:** 100 % of solves converge (S1: 29/29, S2: 29/29, S3: 119/119),
  ~0.05 s per solve instead of ~1.1 s at the iteration limit.

### 1.2 Physics fixes

- **Blower fan power added** to the plant: `P_blower = 300 W · u_blower³`
  (fan affinity law). Counted in `hvac_power`, energy metrics, and CSV
  exports (`P_blower_W`, `P_hp_W` columns).
- **Blower gating of Q_hp added to the predictor** (was only in the
  simulator): `Q_hp = u_blower · u_hvac · Q_max · η_rad(v)`.
- **Heating-mode COP bug fixed in the predictor**: thermal output is no
  longer multiplied by COP (Q_hp_max_heat is already thermal power; COP only
  maps thermal → electrical). Affects future heating studies, not S1–S3.
- Consistency: `η_rad(v) = 0.5 + 0.5(1 − e^(−v/15))` everywhere (one stale
  variant with v_ref = 30 removed); `C_hvac` default raised to a numerically
  stable value; stale power telemetry reset between runs (removed a ~16 Wh
  first-step artifact from a previous run's last power sample).

### 1.3 Objective function (now identical in code and paper)

    J = Σ_k [ w_T · (T_cabin,k − 22 °C)²                (comfort tracking)
            + w_CO2 · (max(0, C_k − 800 ppm)/100)²      (air quality)
            + w_E,hvac · u_hvac,k²                      (HP energy)
            + w_E,blower · u_blower,k²                  (fan energy)
            + Σ_j w_Δu,j · Δu_j,k² ]                    (smoothness)

    s.t. 0 ≤ u_hvac ≤ 1, 0.1 ≤ u_blower ≤ 1, 0 ≤ u_recirc ≤ 0.9,
         C_CO2 ≤ 1200 ppm, 5 °C ≤ T_vent ≤ 65 °C

Weights per scenario:

| Weight | S1 | S2 | S3 comfort | S3 saving |
| --- | --- | --- | --- | --- |
| w_T | 5000 | 5000 | 1000 | 2 |
| w_CO2 | 30 | 50 | 30 | 10 |
| w_E,hvac | 100 | 100 | 50 | 400 |
| w_E,blower | 100 | 100 | 50 | 100 |
| w_Δu (hvac/blower/recirc) | 10/5/5 | 10/5/5 | 10/5/5 | 10/5/5 |

S3 mode switch: when min(SOC) over the 20-min horizon < 10 %, the controller
switches from the comfort weight set to the saving weight set (anticipation
through the horizon; switch fires at t ≈ 41 min, threshold crossing at 60 min).

---

## 2. New numbers (replace throughout the paper)

### Scenario 1: Pre-Conditioning (was 51 % → now **38 %**)

| Metric | PID | MPC | Change |
| --- | --- | --- | --- |
| T at boarding (t=8 min) | 22.4 °C | 22.2 °C | equal |
| Overshoot after boarding | 0.71 K | 0.24 K | **−66 %** |
| CO2 maximum | 1005 ppm | 815 ppm | **−19 %** |
| Energy total | 446 Wh | 275 Wh | **−38 %** |
| … of which heat pump | 367 Wh | 186 Wh | −49 % |
| … of which blower fan | 79 Wh | 89 Wh | +13 % |
| Energy pre-conditioning (t<8 min) | 103 Wh | 70 Wh | −32 % |

**Key new talking point:** the MPC deliberately spends *more* fan energy than
the PID (89 vs 79 Wh) because high airflow lets the compressor run at low
modulation where its COP is best — and still cuts heat-pump energy by 49 %.
This is genuine multi-actuator optimization, not an artifact of free
ventilation. The CO2 improvement is also much stronger now (−19 % peak,
soft target 800 ppm actively controlled).

### Scenario 2: Highway Anticipation (was 65 % → now **56 %**)

| Metric | PID | MPC | Change |
| --- | --- | --- | --- |
| Energy total | 572 Wh | 255 Wh | **−56 %** |
| Energy highway phase (t=5–25 min) | 407 Wh | 177 Wh | −57 % |
| Energy heat pump | 519 Wh | 184 Wh | −65 % |
| Energy blower fan | 53 Wh | 71 Wh | +33 % |
| T mean | 22.5 °C | 22.5 °C | equal |
| Mean u_hvac (highway) | 1.0 | ≈ 0.4 | −60 % |
| η_rad city / highway | 0.64 / 0.91 | 0.64 / 0.91 | identical |

Narrative unchanged from the last revision (PLR partial-load operation
enabled by multi-MV coordination; η_rad as the multiplier that makes the
low-u_hvac regime sustainable on the highway; PID trapped at u_hvac = 1 by
integral wind-up + BlowerPI deadband).

### Scenario 3: SOC Relaxation (was 70 % → now **51 %**)

| Metric | PID | MPC | Change |
| --- | --- | --- | --- |
| Energy total | 1866 Wh | 916 Wh | **−51 %** |
| Energy after threshold (t>60 min) | 970 Wh | 437 Wh | **−55 %** |
| Energy heat pump | 1703 Wh | 670 Wh | −61 % |
| Energy blower fan | 163 Wh | 246 Wh | +51 % |
| Mean u_hvac | 0.98 | 0.44 | −55 % |
| T mean | 22.3 °C | 23.8 °C | MPC allows drift |
| T final | 22.3 °C | 25.1 °C | graceful degradation |
| Comfort violation (>±2 K) | 0 K·min | 66 K·min | accepted trade-off |

Mode-switch timeline (unchanged): comfort mode until t = 41 min (MPC sees
SOC < 10 % inside its 20-min horizon), then saving mode; threshold actually
crossed at t = 60 min; cabin drifts from ~22 °C to 25.1 °C by t = 120 min.

### Summary table (Table 7)

| Scenario | Forecast used | PID [Wh] | MPC [Wh] | Savings |
| --- | --- | --- | --- | --- |
| S1: Pre-Conditioning | n_pass | 446 | 275 | **38 %** |
| S2: Highway Anticipation | v | 572 | 255 | **56 %** |
| S3: SOC Relaxation | SOC (mode switch) | 1866 | 916 | **51 %** |

---

## 3. Required paper text changes

### Abstract & Conclusions
- Replace 51/65/70 % with 38/56/51 %.
- The sentence "without compromising passenger comfort" still holds for S1/S2
  (equal tracking) and S3 explicitly trades comfort — keep the framing.

### Section 2 (System Model)
- **Add blower fan power** to the HVAC component description (2.4):
  `P_blower = P_blower,max · u_blower³` with `P_blower,max = 300 W`
  (fan affinity law). Add `P_blower,max` to Table 1.
- **Add T_vent limits** [5, 65] °C (evaporator icing / safety) to the model
  description and Table 1 — they are now also MPC constraints.
- Eq. (16) energy approximation: extend with the blower term or state that
  the implementation penalizes u_hvac² and u_blower² separately (see 3.2.1
  below — recommend the latter, it is what the code does).

### Section 3.1 (Predictor Bias Correction) — extend by one paragraph
Keep the UA augmentation text, and add the quasi-stationary PTC argument:
"Because the PTC element time constant (7.5 s) is far below the 60 s sample
time, the element is treated as quasi-stationary in the predictor; in
cooling operation the PTC branch is omitted entirely. The duct predictor is
additionally bounded to the physical vent-temperature range [5, 65] °C."

### Section 3.2 (MPC Formulation) — **rewrite Eqs. (14), (15), (17)**
The implemented objective is quadratic tracking with energy terms, NOT the
dead-band formulation currently printed:
- Replace Eq. (14)/(15) with the objective in Section 1.3 of these notes
  (tracking of 22 °C; one-sided CO2 penalty above 800 ppm; u_hvac² and
  u_blower² energy terms; Δu smoothness terms).
- **Drop the passenger-dependent weighting factor (1 + α_pass·n)** — it is
  not implemented and contributes nothing to the three scenarios (occupancy
  is constant in S2/S3). Alternatively implement it later; do not claim it.
- Replace Eq. (17): the SOC mechanism is a **weight-set switch**, not a
  scalar factor 0.2. When min(SOC) over the horizon drops below 10 %, the
  controller swaps the comfort weight set for the saving weight set
  (w_T: 1000 → 2, w_E,hvac: 50 → 400). The anticipation property (acting
  ~20 min early) is preserved and visible at t = 41 min.
- Update Table 2 with the per-scenario weights from Section 1.3.
- Constraints (18)/(19): add `u_recirc ≤ 0.9` (equivalent to the 10 %
  minimum fresh-air guarantee, keeps the NLP smooth) and the T_vent bounds.

### Section 4 (Results)
- Replace all numbers per Section 2 of these notes; regenerate Figures 4–7
  from the new CSVs (`s*_pid.csv`, `s*_mpc.csv`, includes new columns
  `P_hp_W`, `P_blower_W` for an energy-breakdown plot).
- **New recommended plot/argument:** stacked energy breakdown (HP vs fan)
  per scenario — visually proves the savings are not bought with free
  ventilation (MPC fan energy is *higher* than PID's in all scenarios).
- S1: add the CO2 result (−19 % peak) — stronger than before.
- S3: comfort violation is now 66 K·min and T_final 25.1 °C; the
  "time in comfort band" metric drops to 39 % *by design* in saving mode —
  present it as the quantified price of range preservation.

### Section 5 (Discussion)
- 5.1 "Why the savings are this large": tone down. New honest decomposition:
  (a) partial-load COP exploitation enabled by high-blower/low-compressor
  coordination (dominant, S1/S2), (b) recirculation management against the
  CO2 soft target, (c) η_rad timing as a multiplier (S2), (d) conscious
  comfort relaxation (S3 only). State explicitly that fan power is modeled
  and that the MPC spends more fan energy than the PID.
- 5.3 Limitations: you can now *remove* the implicit free-blower issue and
  instead state: compressor electrical power is modeled as proportional to
  modulation (u·Q_max·η_rad/COP) independent of evaporator airflow — a
  starved evaporator therefore wastes energy rather than tripping the
  compressor; a production system would cycle or derate instead, which
  narrows the PID gap. Keep the other limitations (lumped model, simple PID
  baseline, hidden T_mass, binary SOC switch → smooth weighting curve).

---

## 4. Sanity checklist before resubmission

- [x] All IPOPT solves converge (S1 29/29, S2 29/29, S3 119/119).
- [x] Horizon trajectories physically plausible (no PTC below ambient, no
      duct temperatures outside [5, 65] °C).
- [x] Energy metrics include fan power for BOTH controllers.
- [x] JSON/CSV/PNG regenerated from the fixed code (July 2026).
- [ ] Regenerate paper figures from the new CSVs.
- [ ] Cross-check every number in the manuscript against
      `s*_results.json` (single source of truth).
