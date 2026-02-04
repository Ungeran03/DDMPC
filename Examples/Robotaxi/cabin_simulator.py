"""
Robotaxi Cabin Thermal Simulator

A two-node thermal model for simulating cabin climate control in electric vehicles.
This serves as a virtual test environment for MPC development.

Thermal Nodes:
1. Air node (T_cabin): Cabin air temperature
2. Mass node (T_mass): Interior surfaces (dashboard, seats, trim)

HVAC System Components:
1. Heat Pump (reversible): Can heat or cool
   - Cooling: COP = 2-3 (depends on conditions)
   - Heating: COP = 1.5-4 (depends on ambient temperature)
   - COP improves at partial load (PLR correction)
2. PTC Heater: Electric resistance heater for very cold conditions
   - COP = 1 (all electrical energy becomes heat)
   - Activated when T_ambient < PTC_threshold and heating needed
3. Fresh Air / Recirculation control (u_recirc)
   - 0 = full fresh air, 1 = full recirculation
   - Fresh air brings thermal load but reduces CO2

Energy Balance (Air Node):
    C_cabin * dT_air/dt = Q_hvac + Q_ptc + f_solar_air*Q_solar + Q_passengers
                        + Q_transmission + Q_conv(T_mass - T_air) + Q_fresh

Energy Balance (Mass Node):
    C_mass * dT_mass/dt = f_solar_mass*Q_solar + Q_conv(T_air - T_mass)

CO2 Balance:
    V_cabin * dC_CO2/dt = n_pass * R_CO2 - m_dot_fresh * (C_CO2 - C_ambient)
"""

import numpy as np
import pandas as pd
from typing import Optional
from ddmpc.systems.base_class import System
from ddmpc.modeling.modeling import Model


class RobotaxiCabinSimulator(System):
    """
    Two-node thermal model of a robotaxi cabin with realistic HVAC.

    Features:
    - Two thermal nodes: air (T_cabin) and interior mass (T_mass)
    - Separate transmission paths: opaque envelope + windows
    - Heat pump with temperature- and PLR-dependent COP
    - PTC heater for very cold conditions
    - Fresh air / recirculation control
    - CO2 air quality tracking
    - Radiator efficiency depends on vehicle speed
    """

    def __init__(
            self,
            model: Model,
            step_size: int,
            time_offset: int = 0,
            # Air node parameters
            C_cabin: float = 50000.0,        # Air thermal capacity [J/K]
            # Transmission parameters (separate paths)
            UA_opaque: float = 15.0,         # Opaque envelope UA [W/K] (roof+walls+floor)
            UA_window: float = 12.5,         # Window UA [W/K]
            A_window: float = 2.5,           # Total window area [m²]
            tau_window: float = 0.6,         # Window transmittance [-]
            # Interior mass parameters
            C_mass: float = 120000.0,        # Interior mass thermal capacity [J/K]
            h_conv: float = 10.0,            # Convective HTC interior [W/(m²K)]
            A_int: float = 8.0,              # Interior surface area [m²]
            f_solar_air: float = 0.3,        # Fraction of solar to air directly
            f_solar_mass: float = 0.7,       # Fraction of solar absorbed by mass
            # Heat Pump parameters
            Q_hp_max_cool: float = 5000.0,   # Max HP cooling power [W]
            Q_hp_max_heat: float = 4000.0,   # Max HP heating power [W]
            alpha_plr: float = 0.3,          # COP partial load improvement factor [-]
            # PTC Heater parameters
            Q_ptc_max: float = 6000.0,       # Max PTC power [W]
            T_ptc_threshold: float = 268.15, # PTC activation threshold [K] (-5°C)
            # Fresh air / recirculation parameters
            m_dot_blower: float = 0.08,      # Blower mass flow rate [kg/s]
            c_p_air: float = 1005.0,         # Specific heat of air [J/(kg*K)]
            # CO2 parameters
            V_cabin: float = 3.0,            # Cabin air volume [m³]
            R_CO2: float = 5e-6,             # CO2 generation per person [m³/s]
            C_CO2_ambient: float = 420.0,    # Ambient CO2 [ppm]
            rho_air: float = 1.2,            # Air density [kg/m³]
            # Initial conditions
            T_cabin_init: float = 293.15,    # Initial cabin air temp [K] (20°C)
            T_mass_init: float = 293.15,     # Initial mass temp [K] (20°C)
            T_target: float = 295.15,        # Target temperature [K] (22°C)
            C_CO2_init: float = 420.0,       # Initial CO2 [ppm]
            # Legacy parameter (ignored, kept for compatibility)
            hvac_mode: str = 'auto',
    ):
        super().__init__(
            model=model,
            step_size=step_size,
            time_offset=time_offset,
        )

        # Air node parameters
        self.C_cabin = C_cabin
        self.UA_opaque = UA_opaque
        self.UA_window = UA_window
        self.A_window = A_window
        self.tau_window = tau_window

        # Interior mass parameters
        self.C_mass = C_mass
        self.h_conv = h_conv
        self.A_int = A_int
        self.f_solar_air = f_solar_air
        self.f_solar_mass = f_solar_mass

        # Heat Pump parameters
        self.Q_hp_max_cool = Q_hp_max_cool
        self.Q_hp_max_heat = Q_hp_max_heat
        self.alpha_plr = alpha_plr

        # PTC Heater parameters
        self.Q_ptc_max = Q_ptc_max
        self.T_ptc_threshold = T_ptc_threshold

        # Fresh air / recirculation parameters
        self.m_dot_blower = m_dot_blower
        self.c_p_air = c_p_air

        # CO2 parameters
        self.V_cabin = V_cabin
        self.R_CO2 = R_CO2
        self.C_CO2_ambient = C_CO2_ambient
        self.rho_air = rho_air

        # Target temperature for automatic mode selection
        self.T_target = T_target

        # State variables
        self.T_cabin = T_cabin_init
        self.T_cabin_init = T_cabin_init
        self.T_mass = T_mass_init
        self.T_mass_init = T_mass_init
        self.C_CO2 = C_CO2_init
        self.C_CO2_init = C_CO2_init

        # Control inputs
        self.u_hvac = 0.0       # Main HVAC modulation [0-1]
        self.u_ptc = 0.0        # PTC heater modulation [0-1] (can be auto or manual)
        self.u_recirc = 0.5     # Recirculation [0-1]: 0=fresh, 1=recirc

        # Operating mode (determined automatically)
        self.mode = 'idle'      # 'cooling', 'heating', 'heating_ptc', 'idle'

        # Disturbance scenario
        self.scenario = None

        # Energy tracking
        self.E_hp_total = 0.0   # Total HP electrical energy [J]
        self.E_ptc_total = 0.0  # Total PTC electrical energy [J]

    def setup(self, start_time: int, scenario: str = 'summer_city', hvac_mode: str = None, **kwargs):
        """
        Initialize simulation at given start time.

        Args:
            start_time: Unix timestamp
            scenario: 'summer_city', 'winter_highway', 'winter_city', 'mild_mixed'
            hvac_mode: 'cooling', 'heating', or 'auto' (default: based on scenario)
        """
        self.time = start_time
        self.C_CO2 = self.C_CO2_init
        self.u_hvac = 0.0
        self.u_ptc = 0.0
        self.u_recirc = 0.5
        self.mode = 'idle'
        self.scenario = scenario
        self.E_hp_total = 0.0
        self.E_ptc_total = 0.0
        self._ptc_externally_controlled = False

        # Set HVAC mode based on scenario if not specified
        if hvac_mode is not None:
            self.scenario_mode = hvac_mode
        elif 'summer' in scenario:
            self.scenario_mode = 'cooling'
        elif 'winter' in scenario:
            self.scenario_mode = 'heating'
        else:
            self.scenario_mode = 'auto'

        # Scenario-dependent initial conditions (parked vehicle)
        if 'summer' in scenario:
            # Parked in sun: cabin air hot, dashboard/mass even hotter
            self.T_cabin = 273.15 + 35.0   # 35°C
            self.T_mass = 273.15 + 45.0    # 45°C (dashboard in direct sun)
        elif 'winter' in scenario:
            # Parked in cold: cabin at ambient temperature
            dist = self._get_disturbances(start_time)
            self.T_cabin = dist['T_ambient']
            self.T_mass = dist['T_ambient']
        else:
            # Mild: close to ambient
            dist = self._get_disturbances(start_time)
            self.T_cabin = dist['T_ambient']
            self.T_mass = dist['T_ambient']

        print(f"Robotaxi Cabin Simulator initialized")
        print(f"  Scenario: {scenario}")
        print(f"  HVAC mode: {self.scenario_mode.upper()}")
        print(f"  Start time: {start_time}")
        print(f"  Initial T_cabin: {self.T_cabin - 273.15:.1f}°C")
        print(f"  Initial T_mass:  {self.T_mass - 273.15:.1f}°C")
        print(f"  Target T_cabin: {self.T_target - 273.15:.1f}°C")
        print(f"  PTC threshold: {self.T_ptc_threshold - 273.15:.1f}°C")

    def advance(self):
        """
        Advance simulation by one time step using Euler integration.
        Two-node model: air (T_cabin) + interior mass (T_mass) + CO2 tracking.
        """
        # Get current disturbances
        dist = self._get_disturbances(self.time)

        T_amb = dist['T_ambient']
        I_solar = dist['solar_radiation']
        heading = dist['heading']
        n_pass = dist['n_passengers']
        v = dist['v_vehicle']

        # Determine operating mode based on scenario_mode
        if self.scenario_mode == 'cooling':
            self.mode = 'cooling'
        elif self.scenario_mode == 'heating':
            if T_amb < self.T_ptc_threshold:
                self.mode = 'heating_ptc'
            else:
                self.mode = 'heating'
        else:  # 'auto'
            T_error = self.T_cabin - self.T_target
            if T_error > 0:
                self.mode = 'cooling'
            elif T_error < 0:
                if T_amb < self.T_ptc_threshold:
                    self.mode = 'heating_ptc'
                else:
                    self.mode = 'heating'
            else:
                self.mode = 'idle'

        # =====================================================================
        # Heat flows [W]
        # =====================================================================

        # 1. Transmission through cabin shell (separate paths)
        Q_transmission = (self.UA_opaque + self.UA_window) * (T_amb - self.T_cabin)

        # 2. Solar gain (total, then split between air and mass)
        solar_factor = 0.3 + 0.2 * abs(np.sin(heading))
        Q_solar_total = self.A_window * self.tau_window * I_solar * solar_factor
        Q_solar_air = self.f_solar_air * Q_solar_total
        Q_solar_mass = self.f_solar_mass * Q_solar_total

        # 3. Passenger heat load (~90W sensible heat per person)
        Q_passengers = n_pass * 90.0

        # 4. Convective exchange between air and interior mass
        Q_conv_mass_to_air = self.h_conv * self.A_int * (self.T_mass - self.T_cabin)

        # 5. Fresh air thermal load
        m_dot_fresh = self.m_dot_blower * (1 - self.u_recirc)
        Q_fresh = m_dot_fresh * self.c_p_air * (T_amb - self.T_cabin)

        # 6. Heat Pump (with PLR-dependent COP)
        eta_radiator = self._radiator_efficiency(v)
        COP_cool = self._cop_cooling(T_amb, self.T_cabin)
        COP_heat = self._cop_heating(T_amb)

        Q_hp = 0.0
        P_hp_elec = 0.0

        if self.mode == 'cooling':
            # PLR correction: COP improves at partial load
            COP_cool_eff = COP_cool * (1 + self.alpha_plr * (1 - self.u_hvac))
            Q_hp = -self.u_hvac * self.Q_hp_max_cool * eta_radiator
            P_hp_elec = abs(Q_hp) / COP_cool_eff

        elif self.mode in ['heating', 'heating_ptc']:
            COP_heat_eff = COP_heat * (1 + self.alpha_plr * (1 - self.u_hvac))
            Q_hp = self.u_hvac * self.Q_hp_max_heat * eta_radiator * COP_heat_eff
            P_hp_elec = self.u_hvac * self.Q_hp_max_heat * eta_radiator

        # 7. PTC Heater (independent control, only available when T_amb < threshold)
        #    If no controller sets u_ptc explicitly, mirror u_hvac (PID compatibility)
        Q_ptc = 0.0
        P_ptc_elec = 0.0

        if self.mode == 'heating_ptc':
            if not getattr(self, '_ptc_externally_controlled', False):
                self.u_ptc = self.u_hvac
            Q_ptc = self.u_ptc * self.Q_ptc_max
            P_ptc_elec = Q_ptc

        # Track energy consumption
        self.E_hp_total += P_hp_elec * self.step_size
        self.E_ptc_total += P_ptc_elec * self.step_size

        # =====================================================================
        # Euler integration: Air node
        # =====================================================================
        Q_total_air = (Q_hp + Q_ptc + Q_solar_air + Q_passengers
                       + Q_transmission + Q_conv_mass_to_air + Q_fresh)
        dT_air = Q_total_air * self.step_size / self.C_cabin
        self.T_cabin += dT_air

        # =====================================================================
        # Euler integration: Mass node
        # =====================================================================
        Q_total_mass = Q_solar_mass - Q_conv_mass_to_air
        dT_mass = Q_total_mass * self.step_size / self.C_mass
        self.T_mass += dT_mass

        # =====================================================================
        # CO2 dynamics
        # =====================================================================
        Q_vol_fresh = m_dot_fresh / self.rho_air  # Volumetric fresh air [m³/s]
        CO2_generation = n_pass * self.R_CO2 * 1e6 / self.V_cabin  # [ppm/s]
        CO2_ventilation = (Q_vol_fresh / self.V_cabin) * (self.C_CO2 - self.C_CO2_ambient)
        dC_CO2 = (CO2_generation - CO2_ventilation) * self.step_size
        self.C_CO2 = max(self.C_CO2_ambient, self.C_CO2 + dC_CO2)

        # Store for read()
        self._last_P_hp = P_hp_elec
        self._last_P_ptc = P_ptc_elec
        self._last_Q_hp = Q_hp
        self._last_Q_ptc = Q_ptc
        self._last_COP = COP_cool if self.mode == 'cooling' else COP_heat

        # Advance time
        self.time += self.step_size

    def read(self) -> dict:
        """Read current state and disturbances."""
        dist = self._get_disturbances(self.time)

        # Total HVAC electrical power
        P_hvac_total = getattr(self, '_last_P_hp', 0) + getattr(self, '_last_P_ptc', 0)

        return {
            'time': self.time,
            # States
            'cabin_temperature': self.T_cabin,
            'mass_temperature': self.T_mass,
            'co2_concentration': self.C_CO2,
            'hvac_power': P_hvac_total,
            # Controls
            'hvac_modulation': self.u_hvac,
            'ptc_modulation': self.u_ptc,
            'recirc_modulation': self.u_recirc,
            # Disturbances
            'ambient_temperature': dist['T_ambient'],
            'solar_irradiance': dist['solar_radiation'],
            'vehicle_heading': dist['heading'],
            'passenger_count': dist['n_passengers'],
            'vehicle_speed': dist['v_vehicle'],
            'battery_soc': dist['soc'],
            # Additional info
            'battery_power_hvac': P_hvac_total,
            'hp_power': getattr(self, '_last_P_hp', 0),
            'ptc_power': getattr(self, '_last_P_ptc', 0),
            'hvac_mode': 1 if self.mode == 'cooling' else (-1 if 'heating' in self.mode else 0),
            'cop': getattr(self, '_last_COP', 1.0),
            'fresh_air_flow': self.m_dot_blower * (1 - self.u_recirc),
        }

    def write(self, values: dict):
        """Write control values to simulator."""
        if 'hvac_modulation' in values:
            self.u_hvac = np.clip(values['hvac_modulation'], 0.0, 1.0)
        if 'ptc_modulation' in values:
            self.u_ptc = np.clip(values['ptc_modulation'], 0.0, 1.0)
            self._ptc_externally_controlled = True
        if 'recirc_modulation' in values:
            self.u_recirc = np.clip(values['recirc_modulation'], 0.0, 1.0)

    def _get_forecast(self, horizon_in_seconds: int) -> pd.DataFrame:
        """Generate forecast for MPC."""
        n_steps = int(horizon_in_seconds / self.step_size) + 1
        times = [self.time + i * self.step_size for i in range(n_steps)]

        data = []
        for t in times:
            dist = self._get_disturbances(t)
            dist['time'] = t
            data.append(dist)

        return pd.DataFrame(data)

    def _get_passengers(self, time: int) -> int:
        """
        Passenger count from booking schedule (0-4 passengers).

        Robotaxi stops every ~10 min, passengers board/alight.
        The schedule is deterministic (reproducible) but varies 0-4.
        MPC has access to this schedule via forecast.
        """
        # Stops every 10 minutes (600 s)
        stop_index = int(time // 600)
        # Deterministic schedule covering a full shift
        schedule = [1, 3, 2, 0, 4, 2, 1, 3, 0, 2, 4, 1, 3, 2, 0,
                    1, 4, 3, 2, 1, 0, 3, 2, 4, 1, 0, 3, 4, 2, 1]
        return schedule[stop_index % len(schedule)]

    def _get_disturbances(self, time: int) -> dict:
        """
        Generate disturbance values for given time based on scenario.
        """
        # Time of day [0-24h]
        hour = (time % 86400) / 3600.0

        # Passenger count from booking schedule (same for all scenarios)
        n_pass = self._get_passengers(time)

        if self.scenario == 'summer_city':
            # Hot summer day, city driving
            T_amb = 303.15 + 5 * np.sin((hour - 6) * np.pi / 12)  # 30-35°C
            I_solar = max(0, 800 * np.sin((hour - 6) * np.pi / 12))
            v = 8.0 + 5 * np.sin(time / 300)
            soc = max(0.2, 0.8 - (time % 14400) / 72000)

        elif self.scenario == 'winter_highway':
            # Cold winter, highway driving
            T_amb = 263.15 + 5 * np.sin((hour - 6) * np.pi / 12)  # -10 to -5°C
            I_solar = max(0, 200 * np.sin((hour - 8) * np.pi / 8))
            v = 25.0 + 5 * np.sin(time / 600)
            soc = max(0.2, 0.9 - (time % 18000) / 90000)

        elif self.scenario == 'winter_city':
            # Cold winter, city driving (more PTC usage due to low speed)
            T_amb = 268.15 + 3 * np.sin((hour - 6) * np.pi / 12)  # -5 to -2°C
            I_solar = max(0, 150 * np.sin((hour - 8) * np.pi / 8))
            v = 6.0 + 4 * np.sin(time / 300)  # Slow city traffic
            soc = max(0.15, 0.7 - (time % 10800) / 36000)  # Faster depletion in winter

        else:  # 'mild_mixed'
            # Mild weather, mixed driving
            T_amb = 293.15 + 4 * np.sin((hour - 6) * np.pi / 12)  # 20-24°C
            I_solar = max(0, 500 * np.sin((hour - 6) * np.pi / 12))
            v = 12.0 + 8 * np.sin(time / 400)
            soc = max(0.3, 0.85 - (time % 10800) / 54000)

        heading = (time / 100) % (2 * np.pi)

        return {
            'T_ambient': T_amb,
            'solar_radiation': I_solar,
            'heading': heading,
            'n_passengers': float(n_pass),
            'v_vehicle': max(0, v),
            'soc': soc,
        }

    def _radiator_efficiency(self, v: float) -> float:
        """
        Radiator/condenser efficiency as function of vehicle velocity.
        """
        v_ref = 15.0
        return 0.5 + 0.5 * (1 - np.exp(-v / v_ref))

    def _cop_cooling(self, T_amb: float, T_cabin: float) -> float:
        """
        COP for cooling mode (air conditioning).
        COP decreases as temperature difference increases.
        Typical values: 2-4 for automotive AC.
        """
        dT = T_amb - T_cabin  # Positive when hot outside
        # Carnot-inspired but simplified
        COP_ideal = (T_cabin) / max(dT, 5)  # Avoid division by zero
        COP = min(4.0, max(1.5, 0.4 * COP_ideal))  # Realistic bounds
        return COP

    def _cop_heating(self, T_amb: float) -> float:
        """
        COP for heat pump heating mode.
        COP decreases significantly at low ambient temperatures.
        Below ~-10°C, heat pump becomes very inefficient.
        """
        T_amb_C = T_amb - 273.15  # Convert to Celsius

        if T_amb_C > 10:
            COP = 4.0  # Mild conditions
        elif T_amb_C > 0:
            COP = 3.0 + (T_amb_C / 10)  # 3.0 to 4.0
        elif T_amb_C > -10:
            COP = 2.0 + (T_amb_C + 10) / 10  # 2.0 to 3.0
        else:
            COP = max(1.2, 2.0 + (T_amb_C + 10) / 20)  # Below 2.0, min 1.2

        return COP

    def summary(self):
        """Print system summary."""
        print("\n" + "=" * 50)
        print("Robotaxi Cabin Simulator Summary")
        print("=" * 50)
        print(f"  Air thermal capacity: {self.C_cabin:.0f} J/K")
        print(f"  Mass thermal capacity: {self.C_mass:.0f} J/K")
        print(f"  UA opaque: {self.UA_opaque:.1f} W/K")
        print(f"  UA window: {self.UA_window:.1f} W/K")
        print(f"  UA total:  {self.UA_opaque + self.UA_window:.1f} W/K")
        print(f"  Window area: {self.A_window:.1f} m²")
        print(f"  Interior surface: {self.A_int:.1f} m²")
        print(f"  Max HP cooling: {self.Q_hp_max_cool:.0f} W")
        print(f"  Max HP heating: {self.Q_hp_max_heat:.0f} W")
        print(f"  PLR factor: {self.alpha_plr:.1f}")
        print(f"  Max PTC power: {self.Q_ptc_max:.0f} W")
        print(f"  PTC threshold: {self.T_ptc_threshold - 273.15:.0f}°C")
        print(f"  Blower flow: {self.m_dot_blower:.3f} kg/s")
        print(f"  Cabin volume: {self.V_cabin:.1f} m³")
        print(f"  Step size: {self.step_size} s")
        print(f"  Scenario: {self.scenario}")
        if self.E_hp_total > 0 or self.E_ptc_total > 0:
            print(f"  Total HP energy: {self.E_hp_total / 3600:.1f} Wh")
            print(f"  Total PTC energy: {self.E_ptc_total / 3600:.1f} Wh")
        print("=" * 50)
