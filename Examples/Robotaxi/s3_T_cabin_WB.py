"""
Step 3: WhiteBox Models for Robotaxi Cabin

Three physics-based predictors for the MPC controller:
1. T_cabin (air node) - cabin air temperature change
2. T_mass (mass node) - interior mass temperature change
3. C_CO2 - CO2 concentration change

Two-node thermal model with fresh air/recirculation and CO2 tracking.
"""

from configuration import *
from ddmpc.modeling.process_models.physics_based.white_box import WhiteBox
import casadi as ca


# =============================================================================
# MODEL PARAMETERS (must match cabin_simulator.py)
# =============================================================================

# Air node
C_cabin = 50000.0           # Air thermal capacity [J/K]

# Transmission (separate paths)
UA_opaque = 15.0             # Opaque envelope [W/K]
UA_window = 12.5             # Windows [W/K]
A_window = 2.5               # Window area [m²]
tau_window = 0.6             # Window transmittance [-]

# Interior mass
C_mass = 120000.0            # Mass thermal capacity [J/K]
h_conv = 10.0                # Convective HTC [W/(m²K)]
A_int = 8.0                  # Interior surface area [m²]
f_solar_air = 0.3            # Solar fraction to air
f_solar_mass = 0.7           # Solar fraction to mass

# Heat Pump
Q_hvac_max_cool = 5000.0     # Max HP cooling [W]
Q_hvac_max_heat = 4000.0     # Max HP heating [W]
alpha_plr = 0.3              # COP partial load factor [-]

# PTC Heater
Q_ptc_max = 6000.0           # Max PTC power [W]

# Fresh air / recirculation
m_dot_blower_max = 0.08     # Max blower flow [kg/s]
c_p_air = 1005.0             # Specific heat [J/(kg*K)]

# CO2
V_cabin = 3.0                # Cabin volume [m³]
R_CO2_ppm = 5.0              # = 5e-6 * 1e6, CO2 rate [ppm*m³/s per person]
C_ambient = 420.0            # Ambient CO2 [ppm]
rho_air = 1.2                # Air density [kg/m³]

# Step size
dt = one_minute               # 60 seconds


# =============================================================================
# WHITEBOX 1: T_cabin (Air Node)
# =============================================================================

def create_whitebox_T_cabin(hvac_mode: str = 'cooling') -> WhiteBox:
    """
    WhiteBox predictor for cabin air temperature change.

    Energy balance (air node):
        C_air * dT/dt = Q_hvac + Q_ptc + f_solar_air*Q_solar + Q_passengers
                      + Q_transmission + Q_conv(T_mass - T_air) + Q_fresh
    """

    # CasADi symbolic variables
    T_cab = T_cabin.source[0]
    T_m = T_mass.source[0]
    T_amb = T_ambient.source[0]
    u = u_hvac.source[0]
    u_p = u_ptc.source[0]
    u_bl = u_blower.source[0]
    u_rec = u_recirc.source[0]
    I_solar = solar_radiation.source[0]
    n_pass = n_passengers.source[0]
    v = v_vehicle.source[0]
    head = heading.source[0]

    # Radiator efficiency
    v_ref = 15.0
    eta_radiator = 0.5 + 0.5 * (1 - ca.exp(-v / v_ref))

    # Solar factor based on heading
    solar_factor = 0.3 + 0.2 * ca.fabs(ca.sin(head))

    # Heat flows [W]
    Q_transmission = (UA_opaque + UA_window) * (T_amb - T_cab)
    Q_solar_total = A_window * tau_window * I_solar * solar_factor
    Q_solar_air = f_solar_air * Q_solar_total
    Q_passengers = n_pass * 90.0
    Q_conv = h_conv * A_int * (T_m - T_cab)

    # Fresh air thermal load (scaled by blower)
    m_dot_fresh = m_dot_blower_max * u_bl * (1 - u_rec)
    Q_fresh = m_dot_fresh * c_p_air * (T_amb - T_cab)

    # HVAC with PLR-dependent COP (raw, before blower scaling)
    if hvac_mode == 'cooling':
        Q_hvac_raw = -u * Q_hvac_max_cool * eta_radiator
    else:
        # Heating: COP amplifies thermal output, PLR improves COP
        COP_base = 3.0  # Simplified average heating COP for WB
        COP_eff = COP_base * (1 + alpha_plr * (1 - u))
        Q_hvac_raw = u * Q_hvac_max_heat * eta_radiator * COP_eff

    # PTC heater (COP=1, independent control, raw)
    Q_ptc_raw = u_p * Q_ptc_max

    # Blower coupling: without blower, no heat reaches the cabin
    Q_hvac = u_bl * Q_hvac_raw
    Q_ptc = u_bl * Q_ptc_raw

    # Total and temperature change
    Q_total = Q_hvac + Q_ptc + Q_solar_air + Q_passengers + Q_transmission + Q_conv + Q_fresh
    dT = (dt / C_cabin) * Q_total

    wb = WhiteBox(
        inputs=[
            T_cabin.source,
            T_mass.source,
            T_ambient.source,
            u_hvac.source,
            u_ptc.source,
            u_blower.source,
            u_recirc.source,
            solar_radiation.source,
            n_passengers.source,
            v_vehicle.source,
            heading.source,
        ],
        output=T_cabin_change,
        output_expression=dT,
        step_size=dt,
    )
    return wb


# =============================================================================
# WHITEBOX 2: T_mass (Interior Mass Node)
# =============================================================================

def create_whitebox_T_mass() -> WhiteBox:
    """
    WhiteBox predictor for interior mass temperature change.

    Energy balance (mass node):
        C_mass * dT/dt = f_solar_mass * Q_solar + Q_conv(T_air - T_mass)
    """

    T_cab = T_cabin.source[0]
    T_m = T_mass.source[0]
    I_solar = solar_radiation.source[0]
    head = heading.source[0]

    solar_factor = 0.3 + 0.2 * ca.fabs(ca.sin(head))

    Q_solar_total = A_window * tau_window * I_solar * solar_factor
    Q_solar_mass = f_solar_mass * Q_solar_total
    Q_conv = h_conv * A_int * (T_cab - T_m)

    Q_total = Q_solar_mass + Q_conv
    dT_mass = (dt / C_mass) * Q_total

    wb = WhiteBox(
        inputs=[
            T_cabin.source,
            T_mass.source,
            solar_radiation.source,
            heading.source,
        ],
        output=T_mass_change,
        output_expression=dT_mass,
        step_size=dt,
    )
    return wb


# =============================================================================
# WHITEBOX 3: CO2 Concentration
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

    # Fresh air volumetric flow (scaled by blower)
    m_dot_fresh = m_dot_blower_max * u_bl * (1 - u_rec)
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
    print("Robotaxi Cabin - WhiteBox Models")
    print("=" * 60)

    # Create all three predictors
    wb_T_cabin = create_whitebox_T_cabin(hvac_mode='cooling')
    wb_T_mass = create_whitebox_T_mass()
    wb_CO2 = create_whitebox_CO2()

    print("\n1. T_cabin WhiteBox (air node):")
    print(f"   Inputs: {[str(inp.source) for inp in wb_T_cabin.inputs]}")
    print(f"   Output: {wb_T_cabin.output.source}")

    print("\n2. T_mass WhiteBox (mass node):")
    print(f"   Inputs: {[str(inp.source) for inp in wb_T_mass.inputs]}")
    print(f"   Output: {wb_T_mass.output.source}")

    print("\n3. CO2 WhiteBox:")
    print(f"   Inputs: {[str(inp.source) for inp in wb_CO2.inputs]}")
    print(f"   Output: {wb_CO2.output.source}")

    print("\nAll models ready for MPC.")
