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

    The PTC heats up when u_ptc > 0 and transfers heat to T_vent via
    convection coefficient h_ptc.
    """

    T_p = T_ptc_elem.source[0]
    T_v = T_vent.source[0]
    u_p = u_ptc.source[0]

    # PTC electrical power
    P_ptc = u_p * Q_ptc_max

    # Heat transfer to vent air
    Q_ptc_to_vent = h_ptc * (T_p - T_v)

    # Temperature change
    dT_ptc = (dt / C_ptc) * (P_ptc - Q_ptc_to_vent)

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
        C_hvac * dT_vent/dt = Q_hp + h_ptc*(T_ptc - T_vent) - m_dot*c_p*(T_vent - T_inlet)

    where T_inlet = fresh_frac * T_amb + (1 - fresh_frac) * T_cabin

    Heat sources: HP and PTC element
    Heat sink: air flow to cabin
    """

    T_v = T_vent.source[0]
    T_p = T_ptc_elem.source[0]
    T_cab = T_cabin.source[0]
    T_amb = T_ambient.source[0]
    u = u_hvac.source[0]
    u_bl = u_blower.source[0]
    u_rec = u_recirc.source[0]
    v = v_vehicle.source[0]

    # Fresh air fraction with minimum guarantee
    fresh_frac = ca.fmax(min_fresh_frac, 1 - u_rec)

    # Inlet temperature (mixed air entering HVAC)
    T_inlet = fresh_frac * T_amb + (1 - fresh_frac) * T_cab

    # Mass flow through duct
    m_dot = m_dot_blower_max * u_bl

    # Radiator efficiency (velocity-dependent)
    v_ref = 15.0
    eta_radiator = 0.5 + 0.5 * (1 - ca.exp(-v / v_ref))

    # Heat pump thermal power
    if hvac_mode == 'cooling':
        # Cooling: HP removes heat from duct (negative Q)
        Q_hp = -u * Q_hp_max_cool * eta_radiator
    else:
        # Heating: HP adds heat to duct (positive Q with COP)
        COP_base = 3.0
        COP_eff = COP_base * (1 + alpha_plr * (1 - u))
        Q_hp = u * Q_hp_max_heat * eta_radiator * COP_eff

    # Heat from PTC element
    Q_from_ptc = h_ptc * (T_p - T_v)

    # Heat transported to cabin
    Q_to_cabin = m_dot * c_p_air * (T_v - T_inlet)

    # Temperature change
    dT_vent = (dt / C_hvac) * (Q_hp + Q_from_ptc - Q_to_cabin)

    wb = WhiteBox(
        inputs=[
            T_vent.source,
            T_ptc_elem.source,
            T_cabin.source,
            T_ambient.source,
            u_hvac.source,
            u_blower.source,
            u_recirc.source,
            v_vehicle.source,
        ],
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

    Simplified energy balance (MPC does NOT know T_mass):
        C_cabin * dT/dt = m_dot*c_p*(T_vent - T_cabin)
                        + f_solar_air * Q_solar
                        + Q_passengers
                        + Q_transmission

    Note: The convection term with T_mass is OMITTED because T_mass is
    unmeasurable in a real vehicle. The hidden thermal mass acts as an
    unmodeled disturbance. The MPC compensates through feedback.
    """

    T_cab = T_cabin.source[0]
    T_v = T_vent.source[0]
    T_amb = T_ambient.source[0]
    u_bl = u_blower.source[0]
    I_solar = solar_radiation.source[0]
    n_pass = n_passengers.source[0]
    head = heading.source[0]

    # Mass flow from HVAC duct
    m_dot = m_dot_blower_max * u_bl

    # Heat from HVAC duct
    Q_from_hvac = m_dot * c_p_air * (T_v - T_cab)

    # Transmission through envelope
    Q_transmission = (UA_opaque + UA_window) * (T_amb - T_cab)

    # Solar gain (air portion only)
    solar_factor = 0.3 + 0.2 * ca.fabs(ca.sin(head))
    Q_solar_total = A_window * tau_window * I_solar * solar_factor
    Q_solar_air = f_solar_air * Q_solar_total

    # Passenger heat load
    Q_passengers = n_pass * 90.0

    # Total heat (NO T_mass convection - hidden from MPC)
    Q_total = Q_from_hvac + Q_transmission + Q_solar_air + Q_passengers

    # Temperature change
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

    # Fresh air volumetric flow (scaled by blower, min fresh air guarantee)
    fresh_frac = ca.fmax(min_fresh_frac, 1 - u_rec)
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
