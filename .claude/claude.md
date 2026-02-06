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
├── cabin_simulator.py    # Four-node thermal cabin model (T_ptc→T_vent→T_cabin→T_mass)
├── controllers.py        # FixedPID, BlowerPI, PTCRelay controllers
├── s2_generate_data.py   # PID baseline test (scenario-adaptive parameters)
├── s3_T_cabin_WB.py      # 4 WhiteBox predictors for MPC (T_ptc, T_vent, T_cabin, CO2)
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

Implemented SOC weighting (single threshold):
```
w_comfort(soc) = w_base        if soc >= 0.1  (full comfort)
w_comfort(soc) = 0.2 * w_base  if soc <  0.1  (energy saving mode)
```
At SOC<10%: comfort weight drops to 20% of base. MPC allows temperature to drift toward ambient to preserve range. The MPC anticipates this ~20 min ahead using its forecast horizon.

**Key distinction:**
- **MD** affects the **physics** (state equations): e.g. T_amb -> Q_transmission in dT_air/dt
- **Stage Parameter** affects the **cost function / constraints**: e.g. SOC -> w_comfort in J

## Physical Model

### Four-Node Thermal Model with CO2 Tracking

**T_ptc (PTC Element):** C_ptc=1500 J/K, h_ptc=200 W/K, τ=7.5s
```
C_ptc * dT_ptc/dt = u_ptc * Q_ptc_max - h_ptc * (T_ptc - T_vent)
```

**T_vent (HVAC Duct):** C_hvac=5000 J/K — receives HP + PTC heat
```
C_hvac * dT_vent/dt = Q_hp + h_ptc*(T_ptc - T_vent) - m_dot*c_p*(T_vent - T_inlet)
where T_inlet = fresh_frac * T_amb + (1 - fresh_frac) * T_cabin
```

**T_cabin (Cabin Air):** C_cabin=50000 J/K — receives heat via air flow from T_vent
```
C_cabin * dT_cabin/dt = m_dot*c_p*(T_vent - T_cabin) + f_solar_air*Q_solar + Q_passengers
                      + Q_transmission + h_conv*A_int*(T_mass - T_cabin)
```

**T_mass (Interior Surfaces):** C_mass=120000 J/K — hidden from MPC (unmeasurable)
```
C_mass * dT_mass/dt = f_solar_mass*Q_solar + h_conv*A_int*(T_cabin - T_mass)
```

**CO2 Balance:**
```
V_cabin * dC_CO2/dt = n*R_CO2 - m_dot_fresh*(C_CO2 - C_ambient)/rho_air
where fresh_frac = max(min_fresh_frac, 1 - u_recirc)   # min 10% fresh air
      m_dot_fresh = m_dot_max * u_blower * fresh_frac
```

Note: Simulator uses sub-stepping (60 x 1s) for T_ptc/T_vent dynamics (fast time constants).

### Heat Flow Components

| Heat Flow | Formula |
|-----------|---------|
| Q_transmission | (UA_opaque + UA_window) * (T_amb - T_cabin) |
| Q_solar | A_window * tau * I_solar * f(heading), split 30% air / 70% mass |
| Q_passengers | n * 90 W (0-4 passengers from booking schedule) |
| Q_fresh | m_dot_max * u_blower * fresh_frac * c_p * (T_amb - T_cabin), fresh_frac = max(0.1, 1-u_recirc) |
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
   - u_recirc = 1: full recirculation (minimal thermal load, CO2 builds up)
   - Minimum fresh air guarantee: `fresh_frac = max(min_fresh_frac, 1 - u_recirc)` with `min_fresh_frac=0.1`
   - Even at full recirculation, 10% fresh air leaks through (models imperfect seal)
   - Coupling: m_dot_fresh = m_dot_max * u_blower * fresh_frac

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
| min_fresh_frac | 0.1 | Min fresh air fraction even at full recirculation |
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

### Stochastic Drive Cycle

Pre-generated at `setup()` time using a state machine approach. Velocity and passenger profiles are indexed by simulation time.

**Segment Types:**
| Segment | Probability | Speed Range | Description |
|---------|-------------|-------------|-------------|
| STOP | 25% | 0 km/h | Traffic lights, pickup/dropoff |
| SLOW_URBAN | 35% | 10-30 km/h | Dense city driving |
| FAST_URBAN | 30% | 30-50 km/h | Main roads |
| SUBURBAN | 5% | 50-70 km/h | Rare faster roads |
| DECELERATE | 5% | variable | Approaching stops/turns |

**Key Features:**
- Passengers (0-4) can ONLY board/alight when vehicle is completely stopped (v=0)
- Forced stops every 8-12 minutes for passenger changes
- Smooth acceleration (~1 m/s²) and deceleration (~1.5 m/s²)
- `seed` parameter for reproducibility (e.g., seed=42 for summer, seed=123 for winter)
- MPC has access to velocity + passenger forecasts (the robotaxi knows its route + bookings)

## PID Baseline (FixedPID + BlowerPI + PTCRelay, 0-4 passengers)

The PID baseline uses three controllers:
1. **FixedPID**: Controls `u_hvac` based on T_cabin error (scenario-specific gains)
2. **BlowerPI**: Controls `u_blower` — PI with deadband, integral unwinds near target
3. **PTCRelay**: Controls `u_ptc` — relay with hysteresis based on internal PTC element temperature

### BlowerPI Design (PI with Product Error)

Uses product error to keep blower higher in extreme conditions even when T_cabin is near target:
```
e_cabin = abs(T_cabin - T_target)      # cabin error [K]
e_env = abs(T_amb - T_target)          # environmental extremity [K]
e_product = e_cabin * e_env / scale    # dimensionless after scaling

P = Kp * e_product
e_I = e_product - deadband             # negative near target → integral unwinds
integral += (1/Ti) * e_I * dt          # can decrease!
integral = clip(integral, 0, max_int)  # never negative
output = max(u_min, clip(P + Kp*integral, lb, ub))
```

Physics motivation: Heat load scales with |T_amb - T_target| (transmission, fresh air), so required blower scales with environmental extremity.

- **Extreme conditions** (summer/winter): high e_product → high blower
- **Mild conditions**: low e_product → lower blower OK
- **Near target**: small T_cabin error reduces blower appropriately

Parameters: `Kp=0.5, Ti=150s, deadband=0.3, scale=10, u_min=0.15`

### PTCRelay Design (Relay with Hysteresis)

Reads T_ptc from the simulator (4-node model) and applies relay logic based on the PTC element temperature.

Logic:
1. If T_amb >= T_setpoint: PTC OFF (no heating needed)
2. If T_cabin >= T_setpoint - T_margin: PTC OFF (let HP fine-tune)
3. Otherwise: relay on T_ptc from simulator:
   - T_ptc > 65°C (338.15K): relay OFF
   - T_ptc < 50°C (323.15K): relay ON
   - Between: hysteresis (maintain state)

Parameters: `T_on=50°C, T_off=65°C, T_margin=1K`
- Thresholds tuned for 4-node model with C_ptc=1500 J/K, h_ptc=200 W/K
- T_ptc_ss ≈ T_vent + Q_ptc/h_ptc = T_vent + 30K when ON
- **MPC**: does NOT use PTCRelay — controls u_ptc continuously [0,1]

### Results

| Scenario | Kp_hvac | Ti | reverse_act | T_mean | T_range | CO2 max | Energy |
|----------|---------|------|-------------|--------|---------|---------|--------|
| Summer City | 0.30 | 100 | True | 22.4C | [20.9, 35.0]C | 2381 ppm | 3,051 Wh |
| Winter Highway | 0.04 | 100 | False | 22.2C | [-10.0, 28.4]C | 4451 ppm | 6,317 Wh |

**CO2**: High values are expected. BlowerPI is driven by temperature error, not CO2. When u_blower drops at steady state, fresh air decreases. This is a fundamental PID limitation — MPC resolves it through coordinated u_blower/u_recirc optimization with a hard CO2 constraint.

**Energy**: Higher than with fixed u_blower=0.5 because the blower-HP coupling means Q_eff = u_blower * Q_raw. When u_blower drops at steady state, the HP must run harder (higher u_hvac) to maintain temperature, but only a fraction reaches the cabin.

### PID Tuning Notes

- Winter requires much lower Kp (0.04 vs 0.3) due to blower coupling + PLR-COP amplification
- Blower coupling: Q_eff = u_blower * Q_raw. During warmup u_blower→1.0; at steady state ~0.3
- Winter warmup overshoot (~28°C) is from HP integral wind-up during fast warmup. PTC stops at T_cabin=21°C (margin=1K) but HP integral takes ~20 min to unwind
- PTC relay cycling visible in winter for first ~2 hours (while T_amb < PTC threshold)
- Summer cooldown from 35C to 22C takes ~20 min; mass (45C) takes ~100 min

### PID Limitations (MPC advantages)

- PID controls `u_hvac` + `u_blower` + `u_ptc` (relay); `u_recirc` stays fixed at 0.5
- PTC is on/off relay only — MPC can use continuous u_ptc and prefer heat pump over PTC for better COP
- No anticipation of passenger changes (reactive only)
- **CO2 uncontrolled**: BlowerPI driven by temperature, not air quality → CO2 peaks 2000-4000+ ppm. MPC keeps CO2 < 1200 ppm via coordinated u_blower + u_recirc
- No SOC awareness (same energy usage regardless of battery state)
- Cannot exploit thermal mass dynamics

## WhiteBox Predictors (4 models for MPC)

| Predictor | Output | Inputs | ODE |
|-----------|--------|--------|-----|
| T_ptc | dT_ptc | T_ptc, T_vent, u_ptc | PTC element energy balance |
| T_vent | dT_vent | T_vent, T_ptc, T_cabin, T_amb, u_hvac, u_blower, u_recirc, v | HVAC duct energy balance |
| T_cabin | dT_cabin | T_cabin, T_vent, T_amb, u_blower, u_recirc, solar, n_pass, heading | Cabin air energy balance |
| CO2 | dC_CO2 | C_CO2, n_pass, u_blower, u_recirc | CO2 mass balance |

Note: **T_mass is hidden from MPC** (unmeasurable in real vehicle). The MPC only knows T_ptc, T_vent, T_cabin, and CO2.

The WhiteBox models use CasADi symbolic expressions, matching the physics in `cabin_simulator.py`. The `hvac_mode` parameter selects cooling or heating formulation for the T_vent predictor (HP heat flow direction).

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
from controllers import FixedPID, BlowerPI, PTCRelay

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

# Blower PI with deadband (same for all scenarios)
blower = BlowerPI(
    y=T_cabin, u=u_blower, step_size=one_minute,
    Kp=0.5, Ti=150.0, deadband=0.3,
)

# PTC relay (only active in winter when T_amb < -5°C)
ptc = PTCRelay(
    y=T_cabin, u=u_ptc, T_amb_feature=T_ambient,
    step_size=one_minute,
)

# Run with all three controllers
system.run(duration=..., controllers=(pid, blower, ptc))
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

## Paper Scenarios (Robotaxi Application)

Three scenarios demonstrate MPC advantages over PID, ordered from simple to complex:

### Scenario 1: Pre-Conditioning Before Boarding ⭐ (Robotaxi Killer Feature)

| Aspect | Details |
|--------|---------|
| **Forecast used** | n_passengers (booking system) |
| **MPC action** | Pre-cools/heats cabin + pre-ventilates (CO2↓) before passengers board |
| **PID behavior** | Reacts only after boarding → T overshoot + CO2 spike |
| **Key physics** | Q_passengers = n × 90W, CO2 generation = n × R_CO2 |
| **Metric** | T_cabin at boarding, CO2 at boarding, comfort during ride |
| **MVs needed** | u_hvac, u_blower, u_recirc |
| **MDs needed** | n_passengers |

### Scenario 2: Highway Speed Anticipation ⭐⭐

| Aspect | Details |
|--------|---------|
| **Forecast used** | v_vehicle (route planning) |
| **MPC action** | Delays compressor until vehicle accelerates → higher eta_radiator |
| **PID behavior** | Ignores v_vehicle, runs compressor immediately at lower efficiency |
| **Key physics** | eta_radiator(v) = 0.5 + 0.5×(1 - exp(-v/15)) |
| **Metric** | Energy savings [Wh] |
| **MVs needed** | u_hvac |
| **MDs needed** | v_vehicle |

### Scenario 3: SOC-Dependent Comfort Relaxation ⭐⭐⭐

| Aspect | Details |
|--------|---------|
| **Situation** | SOC drops from 20% → 0% over 2 hours |
| **MPC action** | Anticipates SOC drop (~20 min ahead) and switches to low comfort mode BEFORE threshold |
| **PID behavior** | No SOC awareness → keeps cooling at full blast until battery dies |
| **Key physics** | w_comfort = w_base if soc ≥ 0.1, w_comfort = 0.2×w_base if soc < 0.1 |
| **Metric** | Energy savings, T_final (allows drift to save energy) |
| **MVs needed** | u_hvac, u_blower, u_recirc (cooling scenario) |
| **Stage param** | soc (threshold at 10%) |

### Scenario Roadmap

| # | Scenario | Complexity | Implementation Status |
|---|----------|------------|----------------------|
| 1 | Pre-Conditioning | ⭐ | [x] DONE |
| 2 | Highway Anticipation | ⭐⭐ | [x] DONE |
| 3 | SOC Relaxation | ⭐⭐⭐ | [x] DONE |

### Scenario 1 Results: Pre-Conditioning

**Setup:**
- T_cabin_init: 28°C, T_mass_init: 30°C (moderately warm cabin)
- T_ambient: 32°C, Solar: 800 W/m²
- Passengers board at t=8 min (3 passengers), alight at t=25 min
- Duration: 30 min

**Results:**

| Metric | PID | MPC | Improvement |
|--------|-----|-----|-------------|
| T at boarding (t=8min) | 22.4°C | 23.1°C | - |
| Energy total | 367 Wh | 154 Wh | **-57.9%** |
| Energy pre-conditioning | 76 Wh | 54 Wh | -29% |
| CO2 max | 1005 ppm | 1000 ppm | similar |

**Key Insight:** MPC uses gradual u_hvac (~0.4-0.8) instead of PID's immediate max (1.0), achieving same comfort at 58% less energy. MPC exploits PLR-COP efficiency (partial load = higher COP).

**Output Files:**
- `Examples/Robotaxi/s1_preconditioning_comparison.png` - Comparison plot
- `Examples/Robotaxi/s1_preconditioning_results.json` - Metrics + config
- `Examples/Robotaxi/s1_preconditioning_pid.csv` - PID raw data
- `Examples/Robotaxi/s1_preconditioning_mpc.csv` - MPC raw data

**Script:** `Examples/Robotaxi/scenarios/s1_preconditioning.py`

### Scenario 2 Results: Highway Speed Anticipation

**Setup:**
- T_cabin_init: 30°C, T_mass_init: 35°C (warm cabin, just got in)
- T_ambient: 33°C, Solar: 600 W/m²
- 2 passengers throughout
- Velocity: city (5 m/s, t=0-3min) → highway (25 m/s, t=5-25min)
- Duration: 30 min

**Results:**

| Metric | PID | MPC | Improvement |
|--------|-----|-----|-------------|
| Energy total | 519 Wh | 210 Wh | **-59.5%** |
| Energy city (t=0-3min) | 27 Wh | 29 Wh | ~same |
| Energy highway (t=5-25min) | 376 Wh | 132 Wh | **-65%** |
| T steady state (t>15min) | 22.2°C | 23.5°C | Both in band |
| eta_radiator city | 0.64 | 0.64 | same |
| eta_radiator highway | 0.91 | 0.91 | same |

**Key Insight:** MPC saves 60% energy by exploiting better radiator efficiency on highway. At city speed (5 m/s), eta=0.64; at highway (25 m/s), eta=0.91. MPC settles at 23.5°C (upper comfort band) instead of 22°C, trading 1.3K comfort for significant energy savings. Both stay within comfort band [20-24°C].

**Output Files:**
- `Examples/Robotaxi/s2_highway_anticipation_comparison.png` - Comparison plot
- `Examples/Robotaxi/s2_highway_anticipation_results.json` - Metrics + config
- `Examples/Robotaxi/s2_highway_anticipation_pid.csv` - PID raw data
- `Examples/Robotaxi/s2_highway_anticipation_mpc.csv` - MPC raw data

**Script:** `Examples/Robotaxi/scenarios/s2_highway_anticipation.py`

### Scenario 3 Results: SOC-Dependent Comfort Relaxation

**Setup:**
- T_cabin_init: 22°C (already at target), T_mass_init: 24°C
- T_ambient: 32°C, Solar: 600 W/m²
- 2 passengers throughout
- SOC profile: 20% → 0% linear over 2 hours
- SOC threshold: 10% (comfort weight drops to 20% of base)
- Duration: 2 hours

**Results:**

| Metric | PID | MPC | Improvement |
|--------|-----|-----|-------------|
| Energy total | 1693 Wh | 412 Wh | **-75.7%** |
| Energy after threshold | 883 Wh | 205 Wh | **-76.8%** |
| T_mean | 22.3°C | 22.7°C | MPC allows drift |
| T_final | 22.3°C | 22.8°C | Both in comfort band |
| u_hvac mean | 0.97 | 0.28 | **-71%** |
| Comfort violations | 0 K·min | 0 K·min | Both acceptable |

**Key Insight:** MPC switched to low comfort mode at **t=41 min** when it saw SOC would drop below 10% within its 20-min horizon (current SOC=13.2%, min in horizon=9.8%). This is **19 minutes BEFORE** the threshold is actually reached. PID has no SOC awareness and keeps u_hvac at 1.0 the entire time, wasting 76% more energy.

**Anticipation Timeline:**
```
t=0min:   SOC=20%   MPC: normal operation
t=41min:  SOC=13%   MPC: sees threshold in horizon → switches to low comfort
t=60min:  SOC=10%   Threshold reached (MPC already saving for 19 min!)
t=120min: SOC=0%    End of scenario
```

**Output Files:**
- `Examples/Robotaxi/s3_soc_relaxation_comparison.png` - Comparison plot
- `Examples/Robotaxi/s3_soc_relaxation_results.json` - Metrics + config
- `Examples/Robotaxi/s3_soc_relaxation_pid.csv` - PID raw data
- `Examples/Robotaxi/s3_soc_relaxation_mpc.csv` - MPC raw data

**Script:** `Examples/Robotaxi/scenarios/s3_soc_relaxation.py`

---

## MPC Parameter Summary

### Manipulated Variables (MV)

| Variable | Symbol | Range | Unit | Description |
|----------|--------|-------|------|-------------|
| HP Compressor | `u_hvac` | [0, 1] | - | Heat pump modulation |
| PTC Heater | `u_ptc` | [0, 1] | - | PTC modulation (only active when T_amb < -5°C) |
| Blower Fan | `u_blower` | [0.1, 1] | - | Air mass flow scaling |
| Recirculation | `u_recirc` | [0, 1] | - | 0=fresh air, 1=full recirculation |

### Measured Disturbances (MD)

| Variable | Symbol | Typical Range | Unit | Source |
|----------|--------|---------------|------|--------|
| Ambient Temp | `T_amb` | 263–308 | K | Weather + route |
| Solar Radiation | `I_solar` | 0–1000 | W/m² | Weather + time |
| Heading | `heading` | 0–360 | ° | Route planning |
| Passengers | `n_pass` | 0–4 | - | Booking system |
| Vehicle Speed | `v_vehicle` | 0–30 | m/s | Route + traffic |

### States

| State | Symbol | Init Summer | Init Winter | Unit | MPC visible? |
|-------|--------|-------------|-------------|------|--------------|
| PTC Element | `T_ptc` | 308 | 263 | K | ✓ |
| HVAC Duct | `T_vent` | 308 | 263 | K | ✓ |
| Cabin Air | `T_cabin` | 308 | 263 | K | ✓ |
| Interior Mass | `T_mass` | 318 | 263 | K | ✗ (hidden!) |
| CO2 | `C_CO2` | 420 | 420 | ppm | ✓ |

### Stage Parameters

| Parameter | Symbol | Range | Unit | Effect on Optimization |
|-----------|--------|-------|------|------------------------|
| State of Charge | `soc` | [0, 1] | - | w_comfort = w_base if soc≥0.1, else 0.2×w_base |

### Constraints

| Constraint | Type | Value | Description |
|------------|------|-------|-------------|
| CO2 | Hard | ≤ 1200 ppm | Pettenkofer air quality limit |
| T_cabin | Hard | [18, 26]°C | Safety limits (wider than comfort band) |
| T_cabin | Soft | [20, 24]°C | Comfort band (no penalty inside) |
| u_blower | Box | ≥ 0.1 | Minimum ventilation |

### MPC Objective Formulation

The MPC uses a multi-objective cost function with the following structure:

```
J = Σ_k [
    # Temperature: Band violation penalty [20, 24]°C
    w_T × (1 + α_pass × n_pass) × (max(0, T - T_ub)² + max(0, T_lb - T)²)

    # CO2: Soft target at 800 ppm (no penalty below)
    + w_CO2 × max(0, CO2 - 800)²

    # Energy: Approximated electrical power
    + w_E × (u_hvac × Q_hp_max/COP_nom + u_ptc × Q_ptc_max)²

    # Smoothness: Control change penalties
    + w_du_hvac × Δu_hvac²
    + w_du_ptc × Δu_ptc²
    + w_du_blower × Δu_blower²
    + w_du_recirc × Δu_recirc²
]
```

**Key Design Decisions:**

1. **Temperature Band [20, 24]°C**: No penalty inside the band → MPC can optimize for energy. Quadratic penalty outside → fast correction.

2. **Passenger-Dependent Comfort Weight**: More passengers = higher comfort priority.
   ```
   w_comfort_eff = w_comfort_base × (1 + 0.5 × n_passengers)
   ```
   | n_pass | w_eff |
   |--------|-------|
   | 0 | 1.0 × w_base (empty taxi, energy priority) |
   | 2 | 2.0 × w_base |
   | 4 | 3.0 × w_base (full taxi, comfort priority) |

3. **CO2 Two-Stage**: Soft target 800 ppm + hard limit 1200 ppm. Below 800: no penalty → save ventilation energy.

4. **Energy Approximation**: Uses `P ≈ u × Q_max / COP_nom` which automatically prefers heat pump (COP~2.5) over PTC (COP=1.0).

**Default Weights:**

| Weight | Value | Description |
|--------|-------|-------------|
| w_T | 100 | Temperature band violation |
| α_pass | 0.5 | Passenger scaling factor |
| w_CO2 | 50 | CO2 above 800 ppm |
| w_E | 0.001 | Energy (scaled for Watts) |
| w_du_hvac | 10 | HVAC smoothness |
| w_du_ptc | 10 | PTC smoothness |
| w_du_blower | 5 | Blower smoothness |
| w_du_recirc | 5 | Recirc smoothness |

### Key Physical Couplings

```
Q_hp_eff = u_blower × u_hvac × Q_hp_max × eta_radiator(v)
           ─────────   ─────────────────────────────────
           Blower gate    HP output with radiator efficiency

eta_radiator(v) = 0.5 + 0.5 × (1 - exp(-v/15))
                  ───   ─────────────────────
                  Fan     Ram-air bonus

fresh_frac = max(0.1, 1 - u_recirc)
             ────────
             Min 10% fresh air

m_dot_fresh = m_dot_max × u_blower × fresh_frac
              ──────────────────────────────────
              CO2 dilution + thermal load from outside
```

### MPC Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Step size | 60 s | Sample time |
| Horizon N | 20 | 20-minute prediction |
| Solver | IPOPT | Nonlinear programming |

---

## References

- RWTH Aachen Dissertation: Poovendran, 2024. DOI: 10.18154/RWTH-2025-05081
- Liu & Zhang, 2021: EV Battery Thermal and Cabin Climate MPC (velocity-dependent h_ex, passenger heat model)
- Pulvirenti et al., 2025: Dynamic HVAC Energy Consumption (thermal inertia, air infiltration, ISO 8996)
- Schutzeich et al., 2024: Predictive Cabin Conditioning BEV (CO2 tracking, recirculation, EQT)
- Nam & Ahn, 2025: ANN-based Battery Thermal Management (infinity-horizon MPC)

## Next Steps / TODO

- [x] Scenario 1: Pre-Conditioning (MPC 58% energy savings vs PID)
- [x] Scenario 2: Highway Speed Anticipation (MPC 60% energy savings vs PID)
- [x] Scenario 3: SOC-Dependent Comfort Relaxation (MPC 76% energy savings vs PID)
- [x] Tune PID gains for winter scenario with blower coupling (Kp=0.04, Ti=100)
- [x] Add u_blower as MV with BlowerPI controller and blower coupling physics
- [x] Redesign BlowerPI as PI with product error (keeps blower high in extreme conditions)
- [x] Add PTCRelay controller with hysteresis on T_ptc from simulator
- [x] Add min_fresh_frac to physics (10% fresh air even at full recirculation)
- [x] Implement stochastic drive cycle (passengers only board at stops, seed for reproducibility)
- [x] Extend to 4-node thermal model (T_ptc → T_vent → T_cabin → T_mass)
- [x] Implement SOC-dependent comfort weighting (anticipation-based, threshold at 10%)
- [ ] Optional: EQT comfort metric (Schutzeich)
- [ ] Validate with real vehicle data (future)
