from ddmpc import *
import numpy as np

"""
Robotaxi Cabin Climate Control Configuration

This configuration defines the features (variables) for thermal comfort control
in an autonomous robotaxi cabin, considering:
- Passenger load changes (boarding/alighting)
- Solar radiation dependent on driving direction
- Ambient temperature variations
- State of Charge (SOC) influencing cost function
- Vehicle velocity affecting radiator efficiency
"""

# =============================================================================
# Time Configuration
# =============================================================================
time_offset = 0  # Can be adjusted for simulation start time

# =============================================================================
# Modes for Cabin Temperature Control
# =============================================================================
# Comfort bounds: 20-24°C (293.15-297.15 K)
T_cabin_steady = Steady(
    day_start=6,
    day_end=22,
    day_target=295.15,      # 22°C target during operation
    night_target=293.15     # 20°C during idle/night
)

T_cabin_identification = Identification(
    day_start=6,
    day_end=22,
    day_lb=291.15,          # 18°C lower bound
    day_ub=299.15,          # 26°C upper bound
    night_lb=288.15,        # 15°C night lower
    night_ub=299.15,        # 26°C night upper
    min_interval=one_minute * 5,
    max_interval=one_minute * 30,
    min_change=0.5,
    max_change=2
)

T_cabin_economic = Economic(
    day_start=6,
    day_end=22,
    day_lb=291.15,          # 18°C - wider bounds for energy saving
    day_ub=299.15,          # 26°C
    night_lb=288.15,
    night_ub=299.15
)

# =============================================================================
# CONTROLLED VARIABLES
# =============================================================================

# Cabin air temperature [K]
T_cabin = Controlled(
    source=Readable(
        name="T_cabin",
        read_name="cabin_temperature",  # Sensor reading
        plt_opts=PlotOptions(color=red, line=line_solid),
    ),
    mode=T_cabin_steady,
)
T_cabin_change = Connection(Change(base=T_cabin))

# Interior mass temperature [K] - tracked for physics coupling
T_mass_steady = Steady(
    day_start=6,
    day_end=22,
    day_target=295.15,      # Same target as air (equilibrium goal)
    night_target=293.15,
)

T_mass = Controlled(
    source=Readable(
        name="T_mass",
        read_name="mass_temperature",
        plt_opts=PlotOptions(color=dark_red, line=line_dotted, label="T_mass"),
    ),
    mode=T_mass_steady,
)
T_mass_change = Connection(Change(base=T_mass))

# CO2 concentration [ppm]
C_CO2_economic = Economic(
    day_start=0,
    day_end=24,
    day_lb=400,       # Slightly below ambient
    day_ub=1000,      # ASHRAE recommended upper limit
    night_lb=400,
    night_ub=1200,    # Relaxed at night
)

C_CO2 = Controlled(
    source=Readable(
        name="C_CO2",
        read_name="co2_concentration",
        plt_opts=PlotOptions(color=green, line=line_solid, label="CO2"),
    ),
    mode=C_CO2_economic,
)
C_CO2_change = Connection(Change(base=C_CO2))

# HVAC electrical power consumption [W]
P_hvac = Controlled(
    source=Readable(
        name="P_hvac",
        read_name="hvac_power",
        plt_opts=PlotOptions(color=red, line=line_solid, label="P_hvac"),
    ),
    mode=Steady(day_target=0, night_target=0),  # Minimize power consumption
)

# =============================================================================
# CONTROL VARIABLES (Manipulated)
# =============================================================================

# HVAC compressor modulation signal [0-1]
u_hvac = Control(
    source=Readable(
        name="u_hvac",
        read_name="hvac_modulation",
        plt_opts=PlotOptions(color=blue, line=line_solid, label="u_hvac"),
    ),
    lb=0,       # Off
    ub=1,       # Maximum capacity
    default=0,
    cutoff=0.05,
)
u_hvac_change = Connection(Change(base=u_hvac))

# PTC heater modulation signal [0-1] (independent from heat pump)
u_ptc = Control(
    source=Readable(
        name="u_ptc",
        read_name="ptc_modulation",
        plt_opts=PlotOptions(color=red, line=line_dotted, label="u_ptc"),
    ),
    lb=0,       # Off
    ub=1,       # Maximum capacity
    default=0,
    cutoff=0.05,
)
u_ptc_change = Connection(Change(base=u_ptc))

# Recirculation control [0-1]: 0=full fresh air, 1=full recirculation
u_recirc = Control(
    source=Readable(
        name="u_recirc",
        read_name="recirc_modulation",
        plt_opts=PlotOptions(color=green, line=line_dotted, label="u_recirc"),
    ),
    lb=0,       # Full fresh air
    ub=1,       # Full recirculation
    default=0.5,
    cutoff=0.05,
)
u_recirc_change = Connection(Change(base=u_recirc))

# =============================================================================
# DISTURBANCES (External inputs with forecasts)
# =============================================================================

# Ambient temperature [K]
T_ambient = Disturbance(
    Readable(
        name="T_ambient",
        read_name="ambient_temperature",
        plt_opts=PlotOptions(color=blue, line=line_solid, label="T_amb"),
    ),
    forecast_name="T_ambient",
)
T_ambient_change = Connection(Change(base=T_ambient))

# Direct solar radiation [W/m²]
solar_radiation = Disturbance(
    Readable(
        name="solar_radiation",
        read_name="solar_irradiance",
        plt_opts=PlotOptions(color=light_red, line=line_solid, label="Solar"),
    ),
    forecast_name="solar_radiation",
)

# Vehicle heading/direction [rad] - for solar gain calculation
heading = Disturbance(
    Readable(
        name="heading",
        read_name="vehicle_heading",
        plt_opts=PlotOptions(color=grey, line=line_solid, label="Heading"),
    ),
    forecast_name="heading",  # From navigation/route planning
)

# Number of passengers [1] - affects internal heat load
n_passengers = Disturbance(
    Readable(
        name="n_passengers",
        read_name="passenger_count",
        plt_opts=PlotOptions(color=green, line=line_solid, label="Passengers"),
    ),
    forecast_name="n_passengers",  # From booking system
)

# Vehicle velocity [m/s] - affects radiator efficiency
v_vehicle = Disturbance(
    Readable(
        name="v_vehicle",
        read_name="vehicle_speed",
        plt_opts=PlotOptions(color=blue, line=line_dotted, label="Velocity"),
    ),
    forecast_name="v_vehicle",  # From route/traffic prediction
)

# State of Charge [0-1]
soc = Disturbance(
    Readable(
        name="SOC",
        read_name="battery_soc",
        plt_opts=PlotOptions(color=green, line=line_solid, label="SOC"),
    ),
    forecast_name="soc",
)

# =============================================================================
# DERIVED/CONNECTED VARIABLES
# =============================================================================

# Internal heat load from passengers [W]
# Approx. 80-100W sensible heat per person
def passenger_heat_load(n):
    """Calculate heat load from passengers (approx. 90W per person)"""
    return n * 90.0

Q_passengers = Connection(Func(base=n_passengers, func=passenger_heat_load, name="Q_passengers"))

# Effective solar gain considering vehicle orientation
# Simplified: assumes sun position and calculates effective window area
def solar_gain_factor(heading_rad):
    """
    Simplified solar gain factor based on heading.
    In reality, this depends on sun position, window areas, and orientation.
    Returns factor [0-1] for how much solar radiation enters the cabin.
    """
    # Simplified model: maximum when sun is perpendicular to side windows
    return 0.3 + 0.2 * np.abs(np.sin(heading_rad))

solar_factor = Connection(Func(base=heading, func=solar_gain_factor, name="solar_factor"))

# Radiator efficiency as function of velocity
def radiator_efficiency(v):
    """
    Radiator/condenser efficiency increases with airflow (vehicle speed).
    At standstill, only fan provides airflow (lower efficiency).
    Efficiency saturates at higher speeds.
    """
    v_ref = 30.0  # Reference velocity [m/s] where efficiency is ~1
    return 0.6 + 0.4 * (1 - np.exp(-v / v_ref))

eta_radiator = Connection(Func(base=v_vehicle, func=radiator_efficiency, name="eta_radiator"))

# SOC-dependent energy cost factor
# Low SOC = higher "cost" to discourage HVAC usage when battery is low
def soc_cost_factor(soc_val):
    """
    Cost multiplier based on SOC.
    Low SOC -> high cost factor to preserve range.
    """
    if soc_val > 0.5:
        return 1.0
    elif soc_val > 0.2:
        return 1.0 + 2.0 * (0.5 - soc_val)  # Linear increase
    else:
        return 3.0 + 5.0 * (0.2 - soc_val)  # Steep increase below 20%

cost_factor_soc = Connection(Func(base=soc, func=soc_cost_factor, name="cost_factor_soc"))

# =============================================================================
# TRACKING VARIABLES (Monitoring only)
# =============================================================================

# Battery power for HVAC [W]
P_battery = Tracking(
    Readable(
        name="P_battery",
        read_name="battery_power_hvac",
        plt_opts=PlotOptions(color=grey, line=line_dotted, label="P_bat"),
    )
)

# Heat pump electrical power [W]
P_hp = Tracking(
    Readable(
        name="P_hp",
        read_name="hp_power",
        plt_opts=PlotOptions(color=blue, line=line_solid, label="P_hp"),
    )
)

# PTC heater electrical power [W]
P_ptc = Tracking(
    Readable(
        name="P_ptc",
        read_name="ptc_power",
        plt_opts=PlotOptions(color=red, line=line_solid, label="P_ptc"),
    )
)

# HVAC operating mode: 1=cooling, 0=idle, -1=heating
hvac_mode_signal = Tracking(
    Readable(
        name="HVAC_mode",
        read_name="hvac_mode",
        plt_opts=PlotOptions(color=grey, line=line_solid, label="Mode"),
    )
)

# Current COP
cop_current = Tracking(
    Readable(
        name="COP",
        read_name="cop",
        plt_opts=PlotOptions(color=green, line=line_dotted, label="COP"),
    )
)

# =============================================================================
# MODEL AND SYSTEM DEFINITION
# =============================================================================

# Create model with all features
model = Model(*Feature.all)

# Import and create the cabin simulator
from cabin_simulator import RobotaxiCabinSimulator

system = RobotaxiCabinSimulator(
    model=model,
    step_size=one_minute,           # 1 minute control steps
    time_offset=time_offset,
    # Air node parameters
    C_cabin=50000.0,                # Air thermal capacity [J/K]
    # Transmission parameters (separate paths)
    UA_opaque=15.0,                 # Opaque envelope [W/K] (roof+walls+floor)
    UA_window=12.5,                 # Windows [W/K]
    A_window=2.5,                   # Window area [m²]
    tau_window=0.6,                 # Window transmittance [-]
    # Interior mass parameters
    C_mass=120000.0,                # Interior mass [J/K] (dashboard, seats, trim)
    h_conv=10.0,                    # Convective HTC interior [W/(m²K)]
    A_int=8.0,                      # Interior surface area [m²]
    f_solar_air=0.3,                # Solar fraction to air
    f_solar_mass=0.7,               # Solar fraction to mass
    # Heat Pump parameters
    Q_hp_max_cool=5000.0,           # Max HP cooling power [W]
    Q_hp_max_heat=4000.0,           # Max HP heating power [W]
    alpha_plr=0.3,                  # COP partial load factor [-]
    # PTC Heater parameters
    Q_ptc_max=6000.0,               # Max PTC power [W]
    T_ptc_threshold=268.15,         # PTC activation below -5°C
    # Fresh air / recirculation
    m_dot_blower=0.08,              # Blower mass flow [kg/s]
    c_p_air=1005.0,                 # Specific heat of air [J/(kg*K)]
    # CO2 parameters
    V_cabin=3.0,                    # Cabin volume [m³]
    R_CO2=5e-6,                     # CO2 per person [m³/s]
    C_CO2_ambient=420.0,            # Ambient CO2 [ppm]
    rho_air=1.2,                    # Air density [kg/m³]
    # Initial conditions
    T_cabin_init=293.15,            # Initial air temp 20°C
    T_mass_init=293.15,             # Initial mass temp 20°C
    T_target=295.15,                # Target 22°C
    C_CO2_init=420.0,               # Initial CO2 [ppm]
)

# =============================================================================
# PLOTTING CONFIGURATION
# =============================================================================

# Plotter for PID baseline controller
pid_plotter = Plotter(
    SubPlot(features=[T_cabin, T_mass], y_label="Temperatures [°C]", shift=273.15),
    SubPlot(features=[C_CO2], y_label="CO2 [ppm]"),
    SubPlot(features=[P_hp, P_ptc], y_label="Power [W]"),
    SubPlot(features=[u_hvac], y_label="HVAC Signal [-]", step=True),
    SubPlot(features=[T_ambient], y_label="Ambient Temp [°C]", shift=273.15),
    SubPlot(features=[n_passengers], y_label="Passengers [-]"),
)

# Plotter for MPC controller
mpc_plotter = Plotter(
    SubPlot(features=[T_cabin, T_mass], y_label="Temperatures [°C]", shift=273.15),
    SubPlot(features=[C_CO2], y_label="CO2 [ppm]"),
    SubPlot(features=[P_hp, P_ptc], y_label="Power [W]"),
    SubPlot(features=[u_hvac, u_ptc, u_recirc], y_label="Controls [-]", step=True),
    SubPlot(features=[T_ambient], y_label="Ambient Temp [°C]", shift=273.15),
    SubPlot(features=[solar_radiation], y_label="Solar [W/m²]"),
    SubPlot(features=[n_passengers], y_label="Passengers [-]"),
    SubPlot(features=[v_vehicle], y_label="Velocity [m/s]"),
    SubPlot(features=[soc], y_label="SOC [-]"),
)