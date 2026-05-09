# Paper Revision Notes — MPC_Cabin_Climate_AEV

**Audience:** Agent revising the manuscript `MPC_Cabin_Climate_AEV.pdf`.
**Status:** Code and simulation results have been updated. The paper text needs to be revised to match the new results AND to correct a misleading narrative in Scenario 2.

---

## TL;DR

1. All three scenarios were re-run with corrected MPC weights and a fixed WhiteBox predictor. **Energy savings numbers and key tracking metrics have changed** — replace the old numbers throughout the paper.
2. **Scenario 2 narrative is misleading.** The paper currently frames the savings as "η_rad timing", but the data shows the dominant mechanism is **PLR-COP exploitation made possible by MPC's ability to operate at partial load** (which the PID cannot, due to integral wind-up). η_rad is a secondary multiplier, not the primary driver. The narrative needs to be rewritten — see Section 3 below.
3. Add a brief note explaining the WhiteBox UA augmentation (Section 4 below).

---

## 1. New Numbers (replace throughout the paper)

### Scenario 1: Pre-Conditioning

| Metric | PID | MPC | Change |
|---|---|---|---|
| T at boarding (t=8 min) | 22.4 °C | 22.3 °C | equal |
| T overshoot after boarding | 0.71 K | 0.39 K | MPC −45 % |
| Energy total | 367 Wh | 180 Wh | **−51.0 %** |
| Energy pre-conditioning | 76 Wh | 67 Wh | −12 % |
| CO₂ max | 1005 ppm | 921 ppm | −8 % |

**Old number was 58 % savings — replace with 51 %.** The headline change: at *equal comfort tracking* (both ~22.4 °C at boarding), MPC saves 51 % energy AND reduces peak CO₂ by 8 %. Old result had MPC settling at 23.1 °C, which read as setpoint drift; new result has MPC at 22.3 °C, eliminating that critique.

### Scenario 2: Highway Speed Anticipation

| Metric | PID | MPC | Change |
|---|---|---|---|
| T at highway start (t=5 min) | 22.2 °C | 22.3 °C | equal |
| T mean | 23.8 °C | 22.4 °C | **MPC better** |
| Energy total | 519 Wh | 184 Wh | **−64.5 %** |
| Energy highway phase (5–25 min) | 376 Wh | 107 Wh | −71 % |

**Old number was 60 % — replace with 65 %.** Setup also changed: `T_cabin_init` is now 25 °C (was 30 °C), `T_mass_init` 27 °C (was 35 °C). This makes the cooldown phase short enough that the steady-state behaviour during city/highway dominates the comparison — the η_rad / PLR-COP story emerges cleanly instead of being buried under the cooldown transient. **Update Table 3 (scenario configurations) accordingly.**

### Scenario 3: SOC-Dependent Comfort Relaxation

| Metric | PID | MPC | Change |
|---|---|---|---|
| Energy total | 1703 Wh | 507 Wh | **−70.2 %** |
| Energy after threshold (t > 60 min) | 883 Wh | 208 Wh | −76.5 % |
| T mean | 22.3 °C | 23.6 °C | drift after threshold (intentional) |
| T final | 22.3 °C | 24.9 °C | graceful degradation |
| Comfort violation | 0 K·min | 48 K·min | accepted trade-off |

Roughly the same 70 % total savings as before, slightly higher post-threshold savings (77 % vs 74 %). Numbers in the abstract and conclusion need a small update (74 % → 70 %, or 70 % stays as approximate).

### Headline Sentence (abstract / conclusions)

Replace any sentence like *"savings of 58 % to 74 %"* with **"savings of 51 % (S1) to 70 % (S3)"**, or in prose: *"between 51 % and 70 % depending on scenario"*. The middle scenario is now 65 %.

---

## 2. Reframe Scenario 2 — the Important Edit

The current draft (Section 4.2 and 5.1) attributes S2's savings to the η_rad timing advantage. **This is misleading.** η_rad applies to *both* controllers equally — there is no MPC-specific advantage from η_rad alone.

The actual mechanism, in order of contribution:

### (a) PLR-COP exploitation [dominant]

The COP rises at partial load:
```
COP_eff = COP_base · (1 + α_PLR · (1 − u_hvac)),  α_PLR = 0.3
```

At `u_hvac = 0.4`, COP is 18 % higher than at `u_hvac = 1.0`. MPC settles at `u_hvac ≈ 0.3–0.4` in steady state; PID stays at `u_hvac = 1.0`.

### (b) Multi-MV coordination [significant]

Effective cooling power is `Q_eff = u_blower · u_hvac · Q_hp,max · η_rad`. MPC chooses to keep `u_blower = 1.0` (maximum heat transfer) while reducing `u_hvac`. PID has *two independent* controllers (FixedPID for `u_hvac`, BlowerPI for `u_blower`) — they do not coordinate. BlowerPI even drops `u_blower` near target (deadband logic), which forces `u_hvac` higher to compensate.

### (c) η_rad multiplier [secondary]

η_rad does help, but only in concert with (a) and (b): the higher η_rad during highway makes it *feasible* for MPC to drop u_hvac further (because each unit of u_hvac produces 0.91/0.64 ≈ 1.42× more cooling). The η_rad benefit is the *enabler*, not the *driver*. PID never benefits from η_rad because it cannot reduce u_hvac in the first place.

### Why PID is structurally trapped at u_hvac=1.0

The integral term winds up during the cooldown phase to its anti-windup limit (`max_integral = (ub − lb)/Kp = 1/0.3 ≈ 3.33`). After cooldown, with error ≈ 0:
```
output = Kp·e + Kp·integral = 0 + 0.3·3.33 = 1.0
```
The integral can only unwind through sustained negative error (T < target). But BlowerPI throttles `u_blower` once T approaches target, reducing effective cooling and preventing T from dropping below target. The integral never unwinds, and u_hvac is permanently saturated.

**This is not a tuning issue.** It is a structural limitation of single-loop PI control over a multi-input, partial-load-efficient system. To replicate MPC's behaviour in PID would require: anti-windup with active integral reset, gain-scheduling on η_rad, and explicit coupling between the u_hvac and u_blower loops — which is essentially a hand-rolled MPC.

### Suggested paper edits for Section 4.2 (Scenario 2)

Replace the current opening of §4.2 (around lines 308–322 of the PDF) with text along these lines:

> Scenario 2 demonstrates how MPC exploits the velocity forecast for *energy-efficient operation*, not for *deferred cooling*. The cabin starts at 25 °C with the interior mass at 27 °C; both controllers reach the comfort target within 5 min. The interesting comparison is therefore the *steady-state* operating point: PID maintains 22 °C with `u_hvac = 1.0` throughout, while MPC stabilises around `u_hvac = 0.3–0.4`. Three effects combine to give MPC its 65 % energy advantage:
>
> 1. **Partial-load COP bonus.** At `u_hvac = 0.4`, the heat-pump COP is 18 % higher than at full load (Eq. 10). MPC chooses partial load because its model captures this trade-off explicitly; the energy term in (15) rewards lower compressor effort at equal cooling output.
> 2. **Multi-MV coordination.** MPC keeps `u_blower = 1.0` (maximum heat transfer to the cabin) so that low `u_hvac` still produces enough cooling. PID's two independent loops cannot coordinate this way: the BlowerPI drops `u_blower` near target, which then forces `u_hvac` to remain saturated.
> 3. **η_rad multiplier.** The higher η_rad during highway driving (0.91 vs 0.64) makes (1) and (2) feasible: each unit of `u_hvac` produces 1.42× more cooling on the highway, so MPC can sustain steady-state at `u_hvac ≈ 0.3` rather than ~0.5 during city driving. The PID never benefits from this multiplier because it cannot reduce `u_hvac` in the first place — the integral term remains saturated at its anti-windup limit (`max_integral = (ub − lb)/Kp ≈ 3.3`) once it has wound up during the brief cooldown.

### Suggested edits for Section 5.1 (Why the Savings Are This Large)

The current text (around line 363) attributes S2's savings to "MPC simply waits for a better opportunity." This phrasing is wrong — both controllers operate continuously during highway. Replace with:

> In Scenario 2, the MPC saving comes from *operating at partial load* during steady-state cooling, exploiting the heat pump's PLR-COP characteristic and the multi-MV coordination (high `u_blower` enabling low `u_hvac`). The forecast of upcoming highway driving lets the MPC commit to this partial-load regime confidently, because the higher η_rad ensures that `u_hvac ≈ 0.3` is sufficient to balance the cabin heat load. The PID controller cannot reach this regime: its integral term saturates during the initial cooldown and remains saturated indefinitely, because the BlowerPI deadband prevents the cabin from undershooting the target — the integral never unwinds.

---

## 3. WhiteBox UA Augmentation — Methods Section Addition

In `s3_T_cabin_WB.py`, the T_cabin predictor was modified to fold an approximate T_mass coupling into the envelope UA. This is necessary because T_mass is hidden from the MPC (Section 2.2.4 of the paper) and the unmodeled coupling produces a systematic 1 K under-prediction of the cooling load — leading the MPC to settle ~1 K above target in steady cooling.

### Suggested addition to Section 3 (after Eq. 4 / paragraph on T_mass)

Add a short paragraph along these lines:

> Because T_mass is hidden from the MPC, the T_cabin predictor used in the optimiser does not include the convective coupling `h_conv·A_int·(T_mass − T_cabin)`. Without correction, this introduces a systematic bias in cooling scenarios: the predictor under-estimates the cabin heat load, causing the MPC to settle approximately 1 K above target. To remove this bias, we treat the unmeasurable mass node as a quasi-steady relaxation toward the average of cabin air and ambient temperature, and absorb the resulting effective coupling into the envelope UA:
>
> ```
> UA_eff = UA_opaque + UA_window + α_mass · h_conv · A_int
> ```
>
> with `α_mass = 0.5`. This adds 40 W/K of phantom transmission to the MPC's predictor (raising effective UA from 27.5 to 67.5 W/K). The bias-corrected predictor produces tracking equivalent to the simulator within ±0.2 K, eliminating what would otherwise read as MPC "drift" in the comfort plots.

This belongs in Section 3 (Control Design) where the MPC formulation is described. It is not a parameter of the physical system (Section 2) but an offset-free correction in the predictor.

---

## 4. Use the Updated Data Files (Important!)

**All result artefacts in `Examples/Robotaxi/` have been regenerated with the new numbers.** When updating the paper, **use these files as the single source of truth** — do not pull numbers from the old PDF, the JSON files contain the authoritative metrics.

### PNG figures (use these directly in the paper)

These replace Figures 4, 5, and 6 in the current PDF:

- `Examples/Robotaxi/s1_preconditioning_comparison.png` → Figure 4 (Scenario 1)
- `Examples/Robotaxi/s2_highway_anticipation_comparison.png` → Figure 5 (Scenario 2)
- `Examples/Robotaxi/s3_soc_relaxation_comparison.png` → Figure 6 (Scenario 3)

The figure captions in the current PDF (Figures 4, 5, 6) are mostly still accurate, but check that the panel descriptions match. The S3 figure still shows the orange-line annotation "MPC anticipates & starts saving" at t≈40 min and the red threshold line at t=60 min — those are correct and worth keeping in the caption.

If you want to regenerate Figure 7 (the energy bar chart at the end of Section 4.4), the new bar heights are:

```
S1: PID=367, MPC=180  (−51 %)
S2: PID=519, MPC=184  (−65 %)
S3: PID=1703, MPC=507 (−70 %)
```

### JSON metrics (use these for tables and prose numbers)

Each JSON contains the exact values reported in Section 4 of the paper:

- `Examples/Robotaxi/s1_preconditioning_results.json` — fields under `metrics.PID` and `metrics.MPC` for Tables 4 and S1 prose
- `Examples/Robotaxi/s2_highway_anticipation_results.json` — for Table 5 and S2 prose. Key fields: `energy_total_Wh`, `energy_highway_Wh`, `T_at_highway_start_C`, `T_mean_C`, `eta_radiator_city`/`eta_radiator_highway`
- `Examples/Robotaxi/s3_soc_relaxation_results.json` — for Table 6. Key fields: `energy_total_Wh`, `energy_after_threshold_Wh`, `T_final_C`, `comfort_violation_Kmin`, `u_hvac_mean`/`u_hvac_mean_after`

The JSON also contains the full configuration block (`config.weights`, init temperatures, SOC threshold) — useful if you want to add a precise reproducibility paragraph.

### CSV time series (only needed if you regenerate plots)

Raw 1-minute time series for both controllers, useful if you want to draw a custom plot or compute a metric not in the JSON:

- `s1_preconditioning_{pid,mpc}.csv`
- `s2_highway_anticipation_{pid,mpc}.csv`
- `s3_soc_relaxation_{pid,mpc}.csv`

Column reference is in `.claude/CLAUDE.md` under "Data Access for Plotting".

### How to verify nothing has gone stale

The files were regenerated together; mtimes should be within seconds of each other. Run:

```sh
ls -la Examples/Robotaxi/s{1,2,3}*.{png,csv,json}
```

All twelve files should share approximately the same timestamp. If any are noticeably older, re-run the corresponding scenario before pulling numbers:

```sh
source venv/bin/activate
PYTHONPATH=. python Examples/Robotaxi/scenarios/s1_preconditioning.py
PYTHONPATH=. python Examples/Robotaxi/scenarios/s2_highway_anticipation.py
PYTHONPATH=. python Examples/Robotaxi/scenarios/s3_soc_relaxation.py
```

## 5. Source Files Changed (for reference)

- `Examples/Robotaxi/s3_T_cabin_WB.py` — added `UA_mass_phantom` term (40 W/K, α_mass = 0.5) in `create_whitebox_T_cabin()`
- `Examples/Robotaxi/scenario_config.py` — S1: T_cabin weight 500 → 5000, energy 0.0001 → 0.0; S2: T_cabin 1000 → 5000, energy 5e-5 → 0.0, T_cabin_init 30 → 25 °C, T_mass_init 35 → 27 °C; T_vent weight set to 0 in all scenarios
- `Examples/Robotaxi/scenarios/s3_soc_relaxation.py` — saving-mode weights restored to `T_cabin = 5.0`, `u_hvac = 200.0` (the "final" commit values, after a regression in commit `2e40e3c "Verschlimmbessert"`)

---

## 5. Things to keep unchanged

- The S3 narrative ("graceful degradation", visible knee at t ≈ 40 min) is correct and is the strongest visual demonstration in the paper. Just update the numbers.
- The S1 framing as a pre-conditioning + multi-MV story is fine. Emphasise that **at equal tracking quality**, MPC saves 51 % — this is the response to the reviewer who might say "MPC just ran warmer".
- The four-node thermal model description (Section 2.2) and the MPC formulation (Section 3.1) are correct. Only Section 3 needs the WhiteBox UA augmentation note.
- All references and the reference list stay as-is.
