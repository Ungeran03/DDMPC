"""
Robotaxi Cabin Thermal Simulator

A single-zone thermal model for simulating cabin climate control in electric vehicles.
This serves as a virtual test environment for MPC development.

HVAC System Components:
1. Heat Pump (reversible): Can heat or cool
   - Cooling: COP = 2-3 (depends on conditions)
   - Heating: COP = 1.5-4 (depends on ambient temperature)
2. PTC Heater: Electric resistance heater for very cold conditions
   - COP = 1 (all electrical energy becomes heat)
   - Activated when T_ambient < PTC_threshold and heating needed

Energy Balance:
    C_cabin * dT/dt = Q_hvac + Q_ptc + Q_solar + Q_passengers + Q_transmission
"""

import numpy as np
import pandas as pd
from typing import Optional
from ddmpc.systems.base_class import System
from ddmpc.modeling.modeling import Model


class RobotaxiCabinSimulator(System):
    """
    Single-zone thermal model of a robotaxi cabin with realistic HVAC.

    Features:
    - Automatic heating/cooling based on temperature error
    - Heat pump with temperature-dependent COP
    - PTC heater for very cold conditions
    - Radiator efficiency depends on vehicle speed
    """

    def __init__(
            self,
            model: Model,
            step_size: int,
            time_offset: int = 0,
            # Cabin parameters
            C_cabin: float = 50000.0,       # Thermal capacity [J/K] (~50 kg air equivalent)
            UA: float = 80.0,               # Heat transfer coefficient [W/K]
            A_window: float = 2.5,          # Total window area [m²]
            tau_window: float = 0.6,        # Window transmittance [-]
            # Heat Pump parameters
            Q_hp_max_cool: float = 5000.0,  # Max HP cooling power [W]
            Q_hp_max_heat: float = 4000.0,  # Max HP heating power [W]
            # PTC Heater parameters
            Q_ptc_max: float = 6000.0,      # Max PTC power [W]
            T_ptc_threshold: float = 268.15, # PTC activation threshold [K] (-5°C)
            # Initial conditions
            T_cabin_init: float = 293.15,   # Initial cabin temp [K] (20°C)
            T_target: float = 295.15,       # Target temperature [K] (22°C)
            # Legacy parameter (ignored, kept for compatibility)
            hvac_mode: str = 'auto',
    ):
        super().__init__(
            model=model,
            step_size=step_size,
            time_offset=time_offset,
        )

        # Cabin parameters
        self.C_cabin = C_cabin
        self.UA = UA
        self.A_window = A_window
        self.tau_window = tau_window

        # Heat Pump parameters
        self.Q_hp_max_cool = Q_hp_max_cool
        self.Q_hp_max_heat = Q_hp_max_heat

        # PTC Heater parameters
        self.Q_ptc_max = Q_ptc_max
        self.T_ptc_threshold = T_ptc_threshold

        # Target temperature for automatic mode selection
        self.T_target = T_target

        # State variables
        self.T_cabin = T_cabin_init
        self.T_cabin_init = T_cabin_init

        # Control inputs
        self.u_hvac = 0.0       # Main HVAC modulation [0-1]
        self.u_ptc = 0.0        # PTC heater modulation [0-1] (can be auto or manual)

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
        self.T_cabin = self.T_cabin_init
        self.u_hvac = 0.0
        self.u_ptc = 0.0
        self.mode = 'idle'
        self.scenario = scenario
        self.E_hp_total = 0.0
        self.E_ptc_total = 0.0

        # Set HVAC mode based on scenario if not specified
        if hvac_mode is not None:
            self.scenario_mode = hvac_mode
        elif 'summer' in scenario:
            self.scenario_mode = 'cooling'
        elif 'winter' in scenario:
            self.scenario_mode = 'heating'
        else:
            self.scenario_mode = 'auto'

        print(f"Robotaxi Cabin Simulator initialized")
        print(f"  Scenario: {scenario}")
        print(f"  HVAC mode: {self.scenario_mode.upper()}")
        print(f"  Start time: {start_time}")
        print(f"  Initial T_cabin: {self.T_cabin - 273.15:.1f}°C")
        print(f"  Target T_cabin: {self.T_target - 273.15:.1f}°C")
        print(f"  PTC threshold: {self.T_ptc_threshold - 273.15:.1f}°C")

    def advance(self):
        """
        Advance simulation by one time step using Euler integration.
        """
        # Get current disturbances
        dist = self._get_disturbances(self.time)

        T_amb = dist['T_ambient']
        I_solar = dist['solar_radiation']
        heading = dist['heading']
        n_pass = dist['n_passengers']
        v = dist['v_vehicle']

        # Determine operating mode based on scenario_mode
        # The MODE is fixed by scenario (cooling/heating)
        # The INTENSITY is controlled by u_hvac from the controller
        # If u_hvac is near zero, effectively idle

        if self.scenario_mode == 'cooling':
            self.mode = 'cooling'

        elif self.scenario_mode == 'heating':
            # Check if PTC boost is needed
            if T_amb < self.T_ptc_threshold:
                self.mode = 'heating_ptc'
            else:
                self.mode = 'heating'

        else:  # 'auto' - system decides based on temperature error
            T_error = self.T_cabin - self.T_target
            if T_error > 0:  # Too warm -> need cooling
                self.mode = 'cooling'
            elif T_error < 0:  # Too cold -> need heating
                if T_amb < self.T_ptc_threshold:
                    self.mode = 'heating_ptc'
                else:
                    self.mode = 'heating'
            else:
                self.mode = 'idle'

        # Calculate heat flows [W]

        # 1. Transmission through cabin shell
        Q_transmission = self.UA * (T_amb - self.T_cabin)

        # 2. Solar gain
        solar_factor = 0.3 + 0.2 * abs(np.sin(heading))
        Q_solar = self.A_window * self.tau_window * I_solar * solar_factor

        # 3. Passenger heat load (~90W sensible heat per person)
        Q_passengers = n_pass * 90.0

        # 4. Heat Pump
        eta_radiator = self._radiator_efficiency(v)
        COP_cool = self._cop_cooling(T_amb, self.T_cabin)
        COP_heat = self._cop_heating(T_amb)

        Q_hp = 0.0
        P_hp_elec = 0.0

        if self.mode == 'cooling':
            # Cooling: Q_hp is negative (removes heat from cabin)
            Q_hp = -self.u_hvac * self.Q_hp_max_cool * eta_radiator
            P_hp_elec = abs(Q_hp) / COP_cool

        elif self.mode in ['heating', 'heating_ptc']:
            # Heating: Q_hp is positive (adds heat to cabin)
            Q_hp = self.u_hvac * self.Q_hp_max_heat * eta_radiator * COP_heat
            P_hp_elec = self.u_hvac * self.Q_hp_max_heat * eta_radiator

        # 5. PTC Heater (only in heating_ptc mode or when manually activated)
        Q_ptc = 0.0
        P_ptc_elec = 0.0

        if self.mode == 'heating_ptc':
            # Auto PTC: use u_hvac to also control PTC proportionally
            Q_ptc = self.u_hvac * self.Q_ptc_max  # COP = 1, so Q = P
            P_ptc_elec = Q_ptc

        # Track energy consumption
        self.E_hp_total += P_hp_elec * self.step_size
        self.E_ptc_total += P_ptc_elec * self.step_size

        # Total heat flow to cabin
        Q_total = Q_hp + Q_ptc + Q_solar + Q_passengers + Q_transmission

        # Euler integration: dT = Q * dt / C
        dT = Q_total * self.step_size / self.C_cabin
        self.T_cabin += dT

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
            'hvac_power': P_hvac_total,
            # Controls
            'hvac_modulation': self.u_hvac,
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
        }

    def write(self, values: dict):
        """Write control values to simulator."""
        if 'hvac_modulation' in values:
            self.u_hvac = np.clip(values['hvac_modulation'], 0.0, 1.0)

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

    def _get_disturbances(self, time: int) -> dict:
        """
        Generate disturbance values for given time based on scenario.
        """
        # Time of day [0-24h]
        hour = (time % 86400) / 3600.0

        if self.scenario == 'summer_city':
            # Hot summer day, city driving
            T_amb = 303.15 + 5 * np.sin((hour - 6) * np.pi / 12)  # 30-35°C
            I_solar = max(0, 800 * np.sin((hour - 6) * np.pi / 12))
            v = 8.0 + 5 * np.sin(time / 300)
            n_pass = 1 + int(2 * (0.5 + 0.5 * np.sin(time / 600)))
            soc = max(0.2, 0.8 - (time % 14400) / 72000)

        elif self.scenario == 'winter_highway':
            # Cold winter, highway driving
            T_amb = 263.15 + 5 * np.sin((hour - 6) * np.pi / 12)  # -10 to -5°C
            I_solar = max(0, 200 * np.sin((hour - 8) * np.pi / 8))
            v = 25.0 + 5 * np.sin(time / 600)
            n_pass = 2
            soc = max(0.2, 0.9 - (time % 18000) / 90000)

        elif self.scenario == 'winter_city':
            # Cold winter, city driving (more PTC usage due to low speed)
            T_amb = 268.15 + 3 * np.sin((hour - 6) * np.pi / 12)  # -5 to -2°C
            I_solar = max(0, 150 * np.sin((hour - 8) * np.pi / 8))
            v = 6.0 + 4 * np.sin(time / 300)  # Slow city traffic
            n_pass = 1 + int(2 * (0.5 + 0.5 * np.sin(time / 600)))
            soc = max(0.15, 0.7 - (time % 10800) / 36000)  # Faster depletion in winter

        else:  # 'mild_mixed'
            # Mild weather, mixed driving
            T_amb = 293.15 + 4 * np.sin((hour - 6) * np.pi / 12)  # 20-24°C
            I_solar = max(0, 500 * np.sin((hour - 6) * np.pi / 12))
            v = 12.0 + 8 * np.sin(time / 400)
            n_pass = 1 + int(1.5 * (0.5 + 0.5 * np.sin(time / 500)))
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
        print(f"  Thermal capacity: {self.C_cabin:.0f} J/K")
        print(f"  UA value: {self.UA:.1f} W/K")
        print(f"  Window area: {self.A_window:.1f} m²")
        print(f"  Max HP cooling: {self.Q_hp_max_cool:.0f} W")
        print(f"  Max HP heating: {self.Q_hp_max_heat:.0f} W")
        print(f"  Max PTC power: {self.Q_ptc_max:.0f} W")
        print(f"  PTC threshold: {self.T_ptc_threshold - 273.15:.0f}°C")
        print(f"  Step size: {self.step_size} s")
        print(f"  Scenario: {self.scenario}")
        if self.E_hp_total > 0 or self.E_ptc_total > 0:
            print(f"  Total HP energy: {self.E_hp_total / 3600:.1f} Wh")
            print(f"  Total PTC energy: {self.E_ptc_total / 3600:.1f} Wh")
        print("=" * 50)
