"""
Step 3: WhiteBox Models for Robotaxi Cabin (4-Node Model)

Four physics-based predictors for the MPC controller:
1. T_ptc (PTC element) - PTC heater element temperature
2. T_vent (HVAC duct) - duct air temperature after HP and PTC
3. T_cabin (cabin air) - cabin air temperature
4. C_CO2 - CO2 concentration

The MPC does NOT model T_mass (interior surfaces) because this state
is unmeasurable in a real vehicle. The thermal mass acts as an
unmodeled disturbance that the MPC handles through feedback.

4-node thermal model:
    T_ptc -> T_vent -> T_cabin -> (T_mass hidden)
"""

from configuration import *
from ddmpc.modeling.process_models.physics_based.white_box import WhiteBox
import casadi as ca


# =============================================================================
# MODEL PARAMETERS (must match cabin_simulator.py)
# =============================================================================

# Cabin air node
C_cabin = 50000.0           # Air thermal capacity [J/K]

# HVAC duct node
C_hvac = 5000.0             # HVAC duct thermal capacity [J/K]

# PTC element node
C_ptc = 1500.0              # PTC element thermal capacity [J/K]
h_ptc = 200.0               # PTC-to-vent heat transfer coeff [W/K]
Q_ptc_max = 6000.0          # Max PTC electrical power [W]

# Transmission (cabin envelope)
UA_opaque = 15.0            # Opaque envelope [W/K]
UA_window = 12.5            # Windows [W/K]
A_window = 2.5              # Window area [m²]
tau_window = 0.6            # Window transmittance [-]

# Solar distribution (cabin air only, T_mass hidden from MPC)
f_solar_air = 0.3           # Solar fraction to air (rest goes to hidden mass)

# Heat Pump
Q_hp_max_cool = 5000.0      # Max HP cooling [W]
Q_hp_max_heat = 4000.0      # Max HP heating [W]
alpha_plr = 0.3             # COP partial load factor [-]

# Fresh air / recirculation
m_dot_blower_max = 0.08     # Max blower flow [kg/s]
c_p_air = 1005.0            # Specific heat [J/(kg*K)]
min_fresh_frac = 0.1        # Min fresh air even at full recirc [-]

# CO2
V_cabin = 3.0               # Cabin volume [m³]
R_CO2_ppm = 5.0             # = 5e-6 * 1e6, CO2 rate [ppm*m³/s per person]
C_ambient = 420.0           # Ambient CO2 [ppm]
rho_air = 1.2               # Air density [kg/m³]

# Step size
dt = one_minute             # 60 seconds


# =============================================================================
# WHITEBOX 1: T_ptc (PTC Element Node)
# =============================================================================

def create_whitebox_T_ptc() -> WhiteBox:
    """
    WhiteBox predictor for PTC element temperature change.

    Energy balance (PTC element):
        C_ptc * dT_ptc/dt = u_ptc * Q_ptc_max - h_ptc * (T_ptc - T_vent)

    NOTE: The element time constant is tau = C_ptc/h_ptc = 7.5 s, far below
    the 60 s MPC sample time. An explicit Euler step at dt=60 s has the
    discrete eigenvalue 1 - h_ptc*dt/C_ptc = -7 and diverges violently
    (this made the NLP unsolvable when this predictor was included).
    We therefore use the EXACT discretization of the linear ODE, which is
    unconditionally stable:

        T_ptc[k+1] = T_vent + (T_ptc - T_vent)*exp(-dt/tau)
                     + (u_ptc*Q_ptc_max/h_ptc)*(1 - exp(-dt/tau))

    In practice exp(-60/7.5) ~ 3e-4, i.e. the element is quasi-stationary at
    the control time scale. The cooling-mode T_vent predictor therefore does
    not use T_ptc at all, and the heating-mode predictor injects the PTC
    power directly (quasi-steady). This predictor is kept for completeness
    and diagnostics only.
    """

    T_p = T_ptc_elem.source[0]
    T_v = T_vent.source[0]
    u_p = u_ptc.source[0]

    import numpy as np
    decay = float(np.exp(-h_ptc * dt / C_ptc))  # exp(-dt/tau), ~3e-4

    T_ptc_next = T_v + (T_p - T_v) * decay + (u_p * Q_ptc_max / h_ptc) * (1 - decay)
    dT_ptc = T_ptc_next - T_p

    wb = WhiteBox(
        inputs=[
            T_ptc_elem.source,
            T_vent.source,
            u_ptc.source,
        ],
        output=T_ptc_elem_change,
        output_expression=dT_ptc,
        step_size=dt,
    )
    return wb


# =============================================================================
# WHITEBOX 2: T_vent (HVAC Duct Node)
# =============================================================================

def create_whitebox_T_vent(hvac_mode: str = 'cooling') -> WhiteBox:
    """
    WhiteBox predictor for HVAC duct air temperature change.

    Energy balance (HVAC duct):
        C_hvac * dT_vent/dt = Q_hp + Q_ptc - m_dot*c_p*(T_vent - T_inlet)

    where T_inlet = fresh_frac * T_amb + (1 - fresh_frac) * T_cabin

    Design decisions (match cabin_simulator.py physics):
    - Q_hp is gated by u_blower, exactly like the simulator: without air
      flow across the heat exchanger no heat is transferred to the duct air.
    - The PTC element state T_ptc is NOT an input. Its time constant
      (tau = C_ptc/h_ptc = 7.5 s) is far below the 60 s sample time, so the
      element is quasi-stationary: all PTC electrical power reaches the duct
      within one step (Q_ptc = u_ptc * Q_ptc_max). Including T_ptc as a
      predicted state made the NLP numerically unstable (explicit Euler
      eigenvalue -7); including it WITHOUT a predictor left it as a free
      optimization variable that the solver exploited as an unphysical
      200 W/K heat source/sink.
    - Heating mode: Q_hp_max_heat is max THERMAL power. The COP only maps
      thermal to electrical power (energy cost), it must NOT amplify the
      thermal output (this bug was fixed in the simulator earlier and is
      now also fixed here).
    """

    T_v = T_vent.source[0]
    T_cab = T_cabin.source[0]
    T_amb = T_ambient.source[0]
    u = u_hvac.source[0]
    u_bl = u_blower.source[0]
    u_rec = u_recirc.source[0]
    v = v_vehicle.source[0]

    # Fresh air fraction. The simulator floors this at min_fresh_frac via
    # max(0.1, 1-u_recirc); the MPC instead enforces u_recirc <= 0.9 as a box
    # constraint, which is equivalent within the admissible range but keeps
    # the expression SMOOTH. (The former ca.fmax() kink at u_recirc = 0.9 is
    # exactly where the energy-optimal solution sits, and IPOPT stalled on
    # the non-differentiable point.)
    fresh_frac = 1 - u_rec

    # Inlet temperature (mixed air entering HVAC)
    T_inlet = fresh_frac * T_amb + (1 - fresh_frac) * T_cab

    # Mass flow through duct
    m_dot = m_dot_blower_max * u_bl

    # Radiator efficiency (velocity-dependent)
    v_ref = 15.0
    eta_radiator = 0.5 + 0.5 * (1 - ca.exp(-v / v_ref))

    # Heat pump thermal power (blower-gated, like the simulator)
    inputs = [
        T_vent.source,
        T_cabin.source,
        T_ambient.source,
        u_hvac.source,
        u_blower.source,
        u_recirc.source,
        v_vehicle.source,
    ]

    if hvac_mode == 'cooling':
        # Cooling: HP removes heat from duct (negative Q). PTC inactive.
        Q_hp = -u * u_bl * Q_hp_max_cool * eta_radiator
        Q_ptc = 0.0
    else:
        # Heating: Q_hp_max_heat is THERMAL power (no COP amplification!)
        Q_hp = u * u_bl * Q_hp_max_heat * eta_radiator
        # PTC quasi-steady: element passes its full power to the duct
        u_p = u_ptc.source[0]
        Q_ptc = u_p * Q_ptc_max
        inputs.append(u_ptc.source)

    # Heat transported to cabin
    Q_to_cabin = m_dot * c_p_air * (T_v - T_inlet)

    # Temperature change
    dT_vent = (dt / C_hvac) * (Q_hp + Q_ptc - Q_to_cabin)

    wb = WhiteBox(
        inputs=inputs,
        output=T_vent_change,
        output_expression=dT_vent,
        step_size=dt,
    )
    return wb


# =============================================================================
# WHITEBOX 3: T_cabin (Cabin Air Node)
# =============================================================================

def create_whitebox_T_cabin() -> WhiteBox:
    """
    WhiteBox predictor for cabin air temperature change.

    Energy balance (MPC does NOT know T_mass directly):
        C_cabin * dT/dt = m_dot*c_p*(T_vent - T_cabin)
                        + f_solar_air * Q_solar
                        + Q_passengers
                        + Q_transmission_eff

    The hidden thermal mass T_mass would normally contribute
    h_conv*A_int*(T_mass - T_cabin) ≈ 80 W/K * (T_mass - T_cabin).
    Since T_mass is unmeasurable, we approximate its coupling using a
    quasi-steady assumption: T_mass relaxes toward the average of
    T_cabin and T_ambient on a slow timescale, giving an effective
    coupling that scales with (T_amb - T_cabin). We fold this into
    an augmented envelope UA: UA_eff = UA_real + alpha_mass * h_conv*A_int.

    With alpha_mass=0.5 (half-weight to mass coupling), this adds
    40 W/K of "phantom transmission" to the predictor — eliminating
    the systematic cooldown bias that otherwise leaves T_cabin ~1K
    above target in steady cooling.
    """

    T_cab = T_cabin.source[0]
    T_v = T_vent.source[0]
    T_amb = T_ambient.source[0]
    u_bl = u_blower.source[0]
    I_solar = solar_radiation.source[0]
    n_pass = n_passengers.source[0]
    head = heading.source[0]

    # Effective UA approximating the hidden T_mass coupling
    h_conv = 10.0
    A_int = 8.0
    alpha_mass = 0.5  # T_mass treated as half-weighted lag of (T_cabin, T_amb)
    UA_mass_phantom = alpha_mass * h_conv * A_int  # 40 W/K
    UA_eff = UA_opaque + UA_window + UA_mass_phantom

    # Mass flow from HVAC duct
    m_dot = m_dot_blower_max * u_bl

    # Heat from HVAC duct
    Q_from_hvac = m_dot * c_p_air * (T_v - T_cab)

    # Effective transmission (envelope + phantom mass coupling)
    Q_transmission = UA_eff * (T_amb - T_cab)

    # Solar gain (air portion only)
    solar_factor = 0.3 + 0.2 * ca.fabs(ca.sin(head))
    Q_solar_total = A_window * tau_window * I_solar * solar_factor
    Q_solar_air = f_solar_air * Q_solar_total

    # Passenger heat load
    Q_passengers = n_pass * 90.0

    Q_total = Q_from_hvac + Q_transmission + Q_solar_air + Q_passengers

    dT = (dt / C_cabin) * Q_total

    wb = WhiteBox(
        inputs=[
            T_cabin.source,
            T_vent.source,
            T_ambient.source,
            u_blower.source,
            solar_radiation.source,
            n_passengers.source,
            heading.source,
        ],
        output=T_cabin_change,
        output_expression=dT,
        step_size=dt,
    )
    return wb


# =============================================================================
# WHITEBOX 4: CO2 Concentration
# =============================================================================

def create_whitebox_CO2() -> WhiteBox:
    """
    WhiteBox predictor for CO2 concentration change.

    Balance:
        V * dC/dt = n * R_CO2 - m_dot_fresh/rho * (C - C_ambient)
    """

    C_co2 = C_CO2.source[0]
    n_pass = n_passengers.source[0]
    u_bl = u_blower.source[0]
    u_rec = u_recirc.source[0]

    # Fresh air volumetric flow (scaled by blower). Smooth expression;
    # the min-fresh-air guarantee is enforced via u_recirc <= 0.9 in the MPC
    # (see create_whitebox_T_vent for rationale).
    fresh_frac = 1 - u_rec
    m_dot_fresh = m_dot_blower_max * u_bl * fresh_frac
    Q_vol_fresh = m_dot_fresh / rho_air

    # CO2 generation [ppm/s]
    CO2_generation = n_pass * R_CO2_ppm / V_cabin

    # CO2 removal by ventilation [ppm/s]
    CO2_ventilation = (Q_vol_fresh / V_cabin) * (C_co2 - C_ambient)

    # CO2 change per time step
    dC = dt * (CO2_generation - CO2_ventilation)

    wb = WhiteBox(
        inputs=[
            C_CO2.source,
            n_passengers.source,
            u_blower.source,
            u_recirc.source,
        ],
        output=C_CO2_change,
        output_expression=dC,
        step_size=dt,
    )
    return wb


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    print("=" * 60)
    print("Robotaxi Cabin - WhiteBox Models (4-Node)")
    print("=" * 60)
    print("\nModel structure: T_ptc -> T_vent -> T_cabin")
    print("Note: T_mass is hidden from MPC (unmeasurable)")

    # Create all four predictors
    wb_T_ptc = create_whitebox_T_ptc()
    wb_T_vent = create_whitebox_T_vent(hvac_mode='cooling')
    wb_T_cabin = create_whitebox_T_cabin()
    wb_CO2 = create_whitebox_CO2()

    print("\n1. T_ptc WhiteBox (PTC element):")
    print(f"   Inputs: {[str(inp) for inp in wb_T_ptc.inputs]}")
    print(f"   Output: {wb_T_ptc.output}")

    print("\n2. T_vent WhiteBox (HVAC duct):")
    print(f"   Inputs: {[str(inp) for inp in wb_T_vent.inputs]}")
    print(f"   Output: {wb_T_vent.output}")

    print("\n3. T_cabin WhiteBox (cabin air):")
    print(f"   Inputs: {[str(inp) for inp in wb_T_cabin.inputs]}")
    print(f"   Output: {wb_T_cabin.output}")

    print("\n4. CO2 WhiteBox:")
    print(f"   Inputs: {[str(inp) for inp in wb_CO2.inputs]}")
    print(f"   Output: {wb_CO2.output}")

    print("\nAll models ready for MPC.")
