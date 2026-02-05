# DDMPC Project - Claude Context

## Project Overview

**DDMPC** (Data-Driven Model Predictive Control) is a framework that combines machine learning models with Model Predictive Control for building energy systems. The framework supports:

- **Process Models**: ANN, GPR, Linear Regression, WhiteBox (physics-based)
- **Controllers**: PID, Model Predictive Control (MPC)
- **Systems**: FMU, BopTest, Custom simulators

## Robotaxi Cabin Climate Control Extension

A new example demonstrates **MPC for autonomous robotaxi cabin climate control**. The key advantage of a robotaxi over a conventional vehicle is **full forecast availability**: the robotaxi knows its upcoming bookings (passengers), route (velocity, heading), and battery state (SOC). MPC can exploit this knowledge; a PID controller cannot.

### Why MPC for Robotaxis?

1. **Predictive passenger preparation**: MPC pre-cools/pre-heats before passengers board (known from booking schedule), while PID only reacts after temperature deviates.
2. **Active CO2 control**: MPC simultaneously optimizes `u_hvac`, `u_blower`, and `u_recirc` to balance energy vs air quality. PID only controls `u_hvac` + `u_blower`; `u_recirc` stays fixed at 0.5.
3. **Energy efficiency via PLR-COP**: MPC can deliberately operate at partial load (better COP) over longer periods rather than short full-load bursts.
4. **Thermal mass as storage**: MPC knows `T_mass` dynamics and can plan around thermal inertia instead of fighting it.
5. **SOC-aware operation**: When battery is low, MPC relaxes comfort to preserve range (stage parameter weighting). PID has no concept of energy cost.

### File Structure

```
Examples/Robotaxi/
├── configuration.py      # Features, variables, system definition
├── cabin_simulator.py    # Two-node thermal cabin model (simulation environment)
├── controllers.py        # FixedPID, BlowerPI controllers
├── s2_generate_data.py   # PID baseline test (scenario-adaptive parameters)
├── s3_T_cabin_WB.py      # 3 WhiteBox predictors for MPC
├── s4_mpc.py             # MPC vs PID comparison
└── *.png                 # Generated plots
```

## MPC Control Architecture

### Manipulated Variables (MV)

Variables the MPC directly optimizes at each time step:

| MV | Range | Description |
|----|-------|-------------|
| `u_hvac` | [0, 1] | Heat pump compressor modulation |
| `u_ptc` | [0, 1] | PTC heater modulation (independent, only available when T_amb < -5C) |
| `u_blower` | [0.1, 1] | Blower fan speed — scales HVAC heat transfer and fresh air flow |
| `u_recirc` | [0, 1] | Recirculation (0=fresh air, 1=full recirculation) |

### Measured Disturbances (MD)

Known over the forecast horizon (robotaxi has route + booking data). MDs appear directly in the state equations (ODEs):

| MD | Source | Role in Physics (ODE term) |
|----|--------|----------------------------|
| `T_ambient` | Weather data along route | Q_transmission, Q_fresh, COP calculation |
| `solar_radiation` | Weather + time of day | Q_solar (split 30% air / 70% mass) |
| `heading` | Route planning | Solar gain angle factor |
| `n_passengers` | Booking schedule (0-4) | Q_passengers (90W per person), CO2 generation |
| `v_vehicle` | Route/traffic prediction | Radiator/condenser efficiency eta_radiator(v) |

Note: SOC is **not** an MD. It does not appear in any state equation. The causality is reversed: u_hvac and v_vehicle determine energy consumption, which reduces SOC — but SOC does not feed back into the physics.

### Stage Parameters

Values that **vary per time step in the MPC horizon** and **modify the cost function or constraints**, but are neither MV nor MD. They parametrize the optimization problem itself, not the plant model.

**SOC as Stage Parameter**: SOC influences the **comfort weight** in the MPC objective:

```
J = Sum_k [ w_comfort(soc_k) * (T_cabin_k - T_target)^2 + w_energy * u_hvac_k^2 ]
```

When SOC drops below threshold, `w_comfort` decreases, allowing the MPC to tolerate larger temperature deviations to save energy (range > comfort).

| Stage Parameter | Source | Role in Optimization |
|-----------------|--------|----------------------|
| `soc` | BMS forecast along route | Modifies comfort weight w_comfort(soc) |

Proposed SOC weighting:
```
w_comfort(soc) = w_base               if soc >= 0.2
w_comfort(soc) = w_base * soc / 0.2   if soc <  0.2
```
At SOC=10%: half comfort weight. At SOC=5%: quarter. Temperature drifts toward ambient to preserve range.

**Key distinction:**
- **MD** affects the **physics** (state equations): e.g. T_amb -> Q_transmission in dT_air/dt
- **Stage Parameter** affects the **cost function / constraints**: e.g. SOC -> w_comfort in J

## Physical Model

### Two-Node Thermal Model with CO2 Tracking

**Air Node (T_cabin):**
```
C_air * dT_air/dt = u_blower*(Q_hp_raw + Q_ptc_raw) + f_solar_air*Q_solar + Q_passengers
                  + Q_transmission + h_conv*A_int*(T_mass - T_air) + Q_fresh
where Q_fresh uses m_dot = m_dot_max * u_blower * (1 - u_recirc)
```

**Mass Node (T_mass) — interior surfaces (dashboard, seats, trim):**
```
C_mass * dT_mass/dt = f_solar_mass*Q_solar + h_conv*A_int*(T_air - T_mass)
```

**CO2 Balance:**
```
V_cabin * dC_CO2/dt = n*R_CO2 - m_dot_fresh*(C_CO2 - C_ambient)/rho_air
where m_dot_fresh = m_dot_max * u_blower * (1 - u_recirc)
```

### Heat Flow Components

| Heat Flow | Formula |
|-----------|---------|
| Q_transmission | (UA_opaque + UA_window) * (T_amb - T_cabin) |
| Q_solar | A_window * tau * I_solar * f(heading), split 30% air / 70% mass |
| Q_passengers | n * 90 W (0-4 passengers from booking schedule) |
| Q_fresh | m_dot_max * u_blower * (1-u_recirc) * c_p * (T_amb - T_cabin) |
| Q_conv | h_conv * A_int * (T_mass - T_cabin) |
| Q_hp | u_blower * u_hvac * Q_max * eta_radiator(v), COP with PLR correction |
| Q_ptc | u_blower * u_ptc * Q_ptc_max (only when T_amb < -5C) |

### HVAC System Components

1. **Heat Pump (Reversible)** with PLR-dependent COP
   - COP_eff = COP_base * (1 + alpha_plr * (1 - u_hvac))
   - Cooling: COP = 1.5-4.0 (x PLR correction)
   - Heating: COP = 1.2-4.0 (x PLR correction)
   - Max Cooling: 5,000 W, Max Heating: 4,000 W

2. **Radiator/Condenser** efficiency depends on vehicle speed (airflow)
   - eta_radiator(v) = 0.5 + 0.5 * (1 - exp(-v / 15))
   - v=0 (standstill): eta=0.50 (fan only)
   - v=15 m/s (~54 km/h): eta=0.82
   - v=30 m/s (~108 km/h): eta=0.93
   - This is the **front heat exchanger** of the heat pump (condenser when cooling, evaporator when heating)

3. **PTC Heater** (for very cold conditions)
   - COP = 1.0, Max: 6,000 W, Activation: T_amb < -5C
   - Used alongside heat pump in `heating_ptc` mode

4. **Blower Fan** (u_blower control)
   - Scales all HVAC heat transfer to cabin: Q_eff = u_blower * Q_raw
   - Also scales fresh air flow: m_dot_fresh = m_dot_max * u_blower * (1 - u_recirc)
   - Without blower, no warm/cold air enters the cabin
   - Range [0.1, 1.0]: minimum 0.1 for ventilation

5. **Fresh Air / Recirculation** (u_recirc control)
   - u_recirc = 0: full fresh air (thermal load, but good air quality)
   - u_recirc = 1: full recirculation (no thermal load, CO2 builds up)
   - Coupling: m_dot_fresh = m_dot_max * u_blower * (1 - u_recirc)

6. **CO2 Tracking**
   - Constraint: C_CO2 <= 1200 ppm (Pettenkofer limit)
   - Coupled with u_recirc: MPC balances energy vs air quality
   - PID cannot control this (u_recirc fixed at 0.5)

### Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| C_cabin | 50,000 J/K | Air thermal capacity |
| C_mass | 120,000 J/K | Interior mass capacity (dashboard, seats, trim) |
| UA_opaque | 15.0 W/K | Opaque envelope (roof+walls+floor) |
| UA_window | 12.5 W/K | Windows |
| A_window | 2.5 m2 | Window area |
| tau_window | 0.6 | Window transmittance |
| h_conv | 10.0 W/(m2K) | Interior convective HTC |
| A_int | 8.0 m2 | Interior surface area |
| alpha_plr | 0.3 | COP partial load factor |
| m_dot_blower_max | 0.08 kg/s | Max blower mass flow (scaled by u_blower) |
| V_cabin | 3.0 m3 | Cabin volume |
| R_CO2 | 5e-6 m3/s | CO2 generation per person |
| C_CO2_ambient | 420 ppm | Ambient CO2 |
| rho_air | 1.2 kg/m3 | Air density |
| Q_hp_max_cool | 5,000 W | Max HP cooling |
| Q_hp_max_heat | 4,000 W | Max HP heating |
| Q_ptc_max | 6,000 W | Max PTC power |
| T_ptc_threshold | -5C (268.15 K) | PTC activation threshold |
| T_target | 22C (295.15 K) | Comfort target |

## Scenarios

| Scenario | T_ambient | T_init (cabin/mass) | Description |
|----------|-----------|---------------------|-------------|
| `summer_city` | 30-35C | 35C / 45C | Hot, city traffic, parked in sun |
| `winter_highway` | -10 to -5C | ~-10C / ~-10C | Cold, highway, parked in cold |
| `winter_city` | -5 to -2C | ~-5C / ~-5C | Cold, city traffic, borderline PTC |
| `mild_mixed` | 20-24C | ambient / ambient | Mild, mixed traffic |

### Initial Conditions (Realistic Parked Vehicle)

- **Summer:** Cabin soaked at 35C, dashboard/mass at 45C from solar absorption (parked in sun)
- **Winter:** Cabin and mass at ambient temperature (parked in cold)
- **Mild:** At ambient temperature

### Passenger Model

Deterministic booking schedule with stops every 10 minutes. Passenger count varies 0-4, same schedule for all scenarios. The MPC has access to this schedule via forecast (the robotaxi knows upcoming bookings).

Schedule (30-entry cycle): `[1, 3, 2, 0, 4, 2, 1, 3, 0, 2, 4, 1, 3, 2, 0, 1, 4, 3, 2, 1, 0, 3, 2, 4, 1, 0, 3, 4, 2, 1]`

## PID Baseline (FixedPID + BlowerPI, 0-4 passengers)

The PID baseline uses two controllers:
1. **FixedPID**: Controls `u_hvac` based on T_cabin error (scenario-specific gains)
2. **BlowerPI**: Controls `u_blower` — proportional response to abs(T_cabin error) with exponential smoothing and minimum ventilation floor

PTC auto-mirrors u_hvac in the simulator when no controller sets it explicitly.

### BlowerPI Design

Proportional + exponential smoothing (no integral → no windup issues):
```
target = Kp * |error| + min_vent        # proportional + floor
output = alpha * target + (1-alpha) * output_prev   # smoothing
alpha  = step_size / tau
```

- **Transient** (large error): blower ramps to 1.0
- **Steady state** (small error): blower settles at ~min_vent (0.5)
- **Passenger disturbance**: error rises → blower ramps up → more fresh air

Parameters: `Kp=1.0, tau=300s, min_vent=0.5`

### Results

| Scenario | Kp_hvac | Ti | reverse_act | T_mean | T_range | CO2 max | Energy |
|----------|---------|------|-------------|--------|---------|---------|--------|
| Summer City | 0.30 | 100 | True | 22.2C | [21.5, 35.0]C | 1532 ppm | 1,674 Wh |
| Winter Highway | 0.04 | 100 | False | 21.8C | [-10.0, 26.0]C | 1514 ppm | 6,592 Wh |

Note: CO2 exceeds 1200 ppm with 4 passengers. This is a fundamental PID limitation — the BlowerPI is driven by temperature error, not CO2. With u_recirc fixed at 0.5 and 4 passengers, u_blower ≈ 0.77 would be needed for CO2 < 1200 ppm. MPC resolves this through coordinated u_blower/u_recirc optimization with a hard CO2 constraint.

### PID Tuning Notes

- Winter requires much lower Kp (0.04 vs 0.3) due to blower coupling + PLR-COP amplification
- Blower coupling: Q_eff = u_blower * Q_raw. During warmup u_blower→1.0; at steady state ~0.5
- BlowerPI min_vent=0.5 ensures minimum ventilation; tau=300s smooths transitions
- Summer cooldown from 35C to 22C takes ~20 min; mass (45C) takes ~100 min
- Winter warmup from -10C to 22C takes ~25 min; overshoot to ~26C before settling

### PID Limitations (MPC advantages)

- PID controls `u_hvac` + `u_blower`; `u_ptc` mirrors u_hvac automatically, `u_recirc` stays fixed at 0.5
- No independent PTC control (MPC can prefer heat pump over PTC for better COP)
- No anticipation of passenger changes (reactive only)
- **CO2 violation with 4 passengers**: BlowerPI driven by temperature, not air quality → CO2 peaks ~1500 ppm. MPC keeps CO2 < 1200 ppm via coordinated u_blower + u_recirc
- No SOC awareness (same energy usage regardless of battery state)
- Cannot exploit thermal mass dynamics

## WhiteBox Predictors (3 models for MPC)

| Predictor | Output | Inputs | ODE |
|-----------|--------|--------|-----|
| T_cabin | dT_air | T_cabin, T_mass, T_amb, u_hvac, u_ptc, u_blower, u_recirc, solar, n_pass, v, heading | Air energy balance |
| T_mass | dT_mass | T_cabin, T_mass, solar, heading | Mass energy balance |
| CO2 | dC_CO2 | C_CO2, n_pass, u_blower, u_recirc | CO2 mass balance |

The WhiteBox models use CasADi symbolic expressions, matching the physics in `cabin_simulator.py`. The `hvac_mode` parameter selects cooling or heating formulation for the T_cabin predictor (COP affects Q_hp differently in each mode).

## Development Environment

### Virtual Environment

Location: `/Users/unger/Developer/code/DDMPC/venv`

### Running Examples

```bash
cd /Users/unger/Developer/code/DDMPC

# Activate environment
source venv/bin/activate

# Run PID baseline (default: summer_city)
PYTHONPATH=. python Examples/Robotaxi/s2_generate_data.py

# Run MPC vs PID comparison
PYTHONPATH=. python Examples/Robotaxi/s4_mpc.py
```

To switch scenarios, change `scenario = 'winter_highway'` in s2_generate_data.py or s4_mpc.py. The PID parameters are automatically selected based on scenario.

### Key Dependencies

- casadi (optimization/MPC solver)
- numpy, pandas (data handling)
- matplotlib (plotting)
- keras, tensorflow (ANN models - optional)
- scikit-learn (GPR models - optional)
- fmpy (FMU simulation)

## Code Conventions

### PID Controller

**Important:** The standard DDMPC `PID` has an anti-windup bug (see Known Issues). Always use `FixedPID` from `Examples/Robotaxi/controllers.py` for stable operation.

```python
from controllers import FixedPID, BlowerPI

# For cooling scenarios (reverse_act=True)
pid = FixedPID(
    y=T_cabin, u=u_hvac, step_size=one_minute,
    Kp=0.3, Ti=100.0, reverse_act=True,
)

# For heating scenarios (reverse_act=False)
pid = FixedPID(
    y=T_cabin, u=u_hvac, step_size=one_minute,
    Kp=0.04, Ti=100.0, reverse_act=False,
)

# Blower PI (same for all scenarios)
blower = BlowerPI(
    y=T_cabin, u=u_blower, step_size=one_minute,
    Kp=0.2, Ti=150.0,
)

# Run with both controllers
system.run(duration=..., controllers=(pid, blower))
```

### MPC Controller (NLP Pattern)

MPC uses the NLP-based pattern (matching ASHRAE example):

```python
mpc = ModelPredictive(
    step_size=one_minute,
    nlp=NLP(
        model=model, N=20,
        objectives=[
            Objective(feature=T_cabin, cost=Quadratic(weight=100)),
            Objective(feature=T_mass, cost=Quadratic(weight=1)),
            Objective(feature=C_CO2, cost=Quadratic(weight=50)),
            Objective(feature=u_hvac_change, cost=Quadratic(weight=10)),
            Objective(feature=u_ptc_change, cost=Quadratic(weight=10)),
            Objective(feature=u_blower_change, cost=Quadratic(weight=5)),
            Objective(feature=u_recirc_change, cost=Quadratic(weight=5)),
        ],
        constraints=[
            Constraint(feature=u_hvac, lb=0, ub=1),
            Constraint(feature=u_ptc, lb=0, ub=1),
            Constraint(feature=u_blower, lb=0.1, ub=1),
            Constraint(feature=u_recirc, lb=0, ub=1),
            Constraint(feature=C_CO2, lb=0, ub=1200),
        ],
    ),
    forecast_callback=system.get_forecast,
)
# Must build before running:
mpc.nlp.build(solver_options=opts, predictors=[wb_T_cabin, wb_T_mass, wb_CO2])
```

### WhiteBox Model

WhiteBox models use CasADi symbolic expressions for physics-based prediction:

```python
WhiteBox(
    inputs=[source1, source2, ...],
    output=output_change_connection,
    output_expression=casadi_expression,
    step_size=one_minute,
)
```

### Feature Types

| Type | Purpose |
|------|---------|
| `Controlled` | Variables to be controlled (with target/bounds) |
| `Control` | Manipulated variables (MV, with lb/ub) |
| `Disturbance` | External inputs (MD) with forecasts |
| `Connection` | Derived/calculated variables |
| `Tracking` | Monitoring only |

## Known Issues & Fixes

### PID Anti-Windup Bug (CRITICAL)

The standard DDMPC PID controller (`ddmpc/controller/conventional.py`) has a bug in its anti-windup logic:

```python
# Bug in ddmpc/controller/conventional.py line 136:
self.i = output / self.Kp - e
```

When `output=0` (saturated) and `e` is negative (T > target for heating), this sets the integral to a **large positive value**, causing the controller to keep heating even when temperature is way above target. This leads to severe oscillations (15-35C swings).

**Solution:** Use `FixedPID` from `Examples/Robotaxi/controllers.py` which has corrected anti-windup:
- Only integrates when not saturated in the wrong direction
- Clamps integral to `[-max_integral, max_integral]` where `max_integral = (ub - lb) / Kp`
- Verified stable across summer (cooling) and winter (heating+PTC) scenarios

### Locale Error on macOS

The plotting module had a Windows-specific locale setting. Fixed in `ddmpc/utils/plotting.py`:

```python
# Cross-platform locale setting
for loc in ['de_DE.UTF-8', 'de_DE', 'deu_deu', '']:
    try:
        locale.setlocale(locale.LC_ALL, loc)
        break
    except locale.Error:
        continue
```

## References

- RWTH Aachen Dissertation: Poovendran, 2024. DOI: 10.18154/RWTH-2025-05081
- Liu & Zhang, 2021: EV Battery Thermal and Cabin Climate MPC (velocity-dependent h_ex, passenger heat model)
- Pulvirenti et al., 2025: Dynamic HVAC Energy Consumption (thermal inertia, air infiltration, ISO 8996)
- Schutzeich et al., 2024: Predictive Cabin Conditioning BEV (CO2 tracking, recirculation, EQT)
- Nam & Ahn, 2025: ANN-based Battery Thermal Management (infinity-horizon MPC)

## Next Steps / TODO

- [ ] Run MPC vs PID comparison (s4_mpc.py)
- [x] Tune PID gains for winter scenario with blower coupling (Kp=0.04, Ti=100)
- [x] Add u_blower as MV with BlowerPI controller and blower coupling physics
- [ ] Implement SOC-dependent comfort weighting as stage parameter in MPC
- [ ] Optional: EQT comfort metric (Schutzeich)
- [ ] Validate with real vehicle data (future)
