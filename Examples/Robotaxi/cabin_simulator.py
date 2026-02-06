"""
Robotaxi Cabin Thermal Simulator

A four-node thermal model for simulating cabin climate control in electric vehicles.
This serves as a virtual test environment for MPC development.

Thermal Nodes:
1. T_ptc:   PTC heater element temperature
2. T_vent:  HVAC duct air temperature (output of heater/AC before cabin)
3. T_cabin: Cabin air temperature
4. T_mass:  Interior surfaces (dashboard, seats, trim) - not measurable

HVAC System Components:
1. Heat Pump (reversible): Can heat or cool
   - Cooling: COP = 2-3 (depends on conditions)
   - Heating: COP = 1.5-4 (depends on ambient temperature)
   - COP improves at partial load (PLR correction)
   - Heat/cool goes to T_vent, then blower transports to cabin
2. PTC Heater: Electric resistance heater for very cold conditions
   - Electrical power heats T_ptc element
   - T_ptc transfers heat to T_vent via h_ptc coefficient
   - Activated when T_ambient < PTC_threshold
3. Fresh Air / Recirculation control (u_recirc)
   - 0 = full fresh air, 1 = full recirculation
   - Inlet air to HVAC: T_inlet = fresh_frac * T_amb + (1-fresh_frac) * T_cabin

Energy Balance (PTC Element):
    C_ptc * dT_ptc/dt = u_ptc * Q_ptc_max - h_ptc * (T_ptc - T_vent)

Energy Balance (HVAC Duct):
    C_hvac * dT_vent/dt = Q_hp + h_ptc*(T_ptc - T_vent) - m_dot*c_p*(T_vent - T_inlet)

Energy Balance (Cabin Air):
    C_cabin * dT_cabin/dt = m_dot*c_p*(T_vent - T_cabin) + f_solar_air*Q_solar
                          + Q_passengers + Q_transmission + Q_conv(T_mass - T_cabin)

Energy Balance (Mass Node):
    C_mass * dT_mass/dt = f_solar_mass*Q_solar + Q_conv(T_cabin - T_mass)

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
    Four-node thermal model of a robotaxi cabin with realistic HVAC.

    Features:
    - Four thermal nodes: T_ptc (element), T_vent (duct), T_cabin (air), T_mass (interior)
    - Separate transmission paths: opaque envelope + windows
    - Heat pump with temperature- and PLR-dependent COP
    - PTC heater with element dynamics (T_ptc → T_vent)
    - HVAC duct dynamics (T_vent) with inlet mixing
    - Fresh air / recirculation control with T_inlet mixing
    - CO2 air quality tracking
    - Radiator efficiency depends on vehicle speed
    """

    def __init__(
            self,
            model: Model,
            step_size: int,
            time_offset: int = 0,
            # Cabin air node parameters
            C_cabin: float = 50000.0,        # Air thermal capacity [J/K]
            # Transmission parameters (separate paths)
            UA_opaque: float = 15.0,         # Opaque envelope UA [W/K] (roof+walls+floor)
            UA_window: float = 12.5,         # Window UA [W/K]
            A_window: float = 2.5,           # Total window area [m²]
            tau_window: float = 0.6,         # Window transmittance [-]
            # Interior mass parameters (not measurable, only in simulator)
            C_mass: float = 120000.0,        # Interior mass thermal capacity [J/K]
            h_conv: float = 10.0,            # Convective HTC interior [W/(m²K)]
            A_int: float = 8.0,              # Interior surface area [m²]
            f_solar_air: float = 0.3,        # Fraction of solar to air directly
            f_solar_mass: float = 0.7,       # Fraction of solar absorbed by mass
            # HVAC duct parameters (T_vent node)
            C_hvac: float = 100.0,           # HVAC duct thermal capacity [J/K]
            # Heat Pump parameters
            Q_hp_max_cool: float = 5000.0,   # Max HP cooling power [W]
            Q_hp_max_heat: float = 4000.0,   # Max HP heating power [W]
            alpha_plr: float = 0.3,          # COP partial load improvement factor [-]
            # PTC Heater parameters (T_ptc node)
            Q_ptc_max: float = 6000.0,       # Max PTC electrical power [W]
            C_ptc: float = 1500.0,           # PTC element thermal capacity [J/K]
            h_ptc: float = 200.0,            # PTC-to-vent heat transfer coeff [W/K]
            T_ptc_threshold: float = 268.15, # PTC activation threshold [K] (-5°C)
            # Blower / fresh air parameters
            m_dot_blower_max: float = 0.08,  # Max blower mass flow rate [kg/s]
            c_p_air: float = 1005.0,         # Specific heat of air [J/(kg*K)]
            min_fresh_frac: float = 0.1,     # Min fresh air fraction even at full recirc [-]
            # Vent temperature limits (physical constraints)
            T_vent_min: float = 278.15,      # Min vent temp [K] (5°C) - evaporator icing limit
            T_vent_max: float = 338.15,      # Max vent temp [K] (65°C) - safety/comfort limit
            # CO2 parameters
            V_cabin: float = 3.0,            # Cabin air volume [m³]
            R_CO2: float = 5e-6,             # CO2 generation per person [m³/s]
            C_CO2_ambient: float = 420.0,    # Ambient CO2 [ppm]
            rho_air: float = 1.2,            # Air density [kg/m³]
            # Initial conditions
            T_cabin_init: float = 293.15,    # Initial cabin air temp [K] (20°C)
            T_mass_init: float = 293.15,     # Initial mass temp [K] (20°C)
            T_vent_init: float = 293.15,     # Initial duct air temp [K] (20°C)
            T_ptc_init: float = 293.15,      # Initial PTC element temp [K] (20°C)
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

        # Interior mass parameters (not measurable, only in simulator)
        self.C_mass = C_mass
        self.h_conv = h_conv
        self.A_int = A_int
        self.f_solar_air = f_solar_air
        self.f_solar_mass = f_solar_mass

        # HVAC duct parameters (T_vent node)
        self.C_hvac = C_hvac

        # Heat Pump parameters
        self.Q_hp_max_cool = Q_hp_max_cool
        self.Q_hp_max_heat = Q_hp_max_heat
        self.alpha_plr = alpha_plr

        # PTC Heater parameters (T_ptc node)
        self.Q_ptc_max = Q_ptc_max
        self.C_ptc = C_ptc
        self.h_ptc = h_ptc
        self.T_ptc_threshold = T_ptc_threshold

        # Blower / fresh air parameters
        self.m_dot_blower_max = m_dot_blower_max
        self.c_p_air = c_p_air
        self.min_fresh_frac = min_fresh_frac

        # Vent temperature limits
        self.T_vent_min = T_vent_min
        self.T_vent_max = T_vent_max

        # CO2 parameters
        self.V_cabin = V_cabin
        self.R_CO2 = R_CO2
        self.C_CO2_ambient = C_CO2_ambient
        self.rho_air = rho_air

        # Target temperature for automatic mode selection
        self.T_target = T_target

        # State variables (4 thermal nodes + CO2)
        self.T_cabin = T_cabin_init
        self.T_cabin_init = T_cabin_init
        self.T_mass = T_mass_init
        self.T_mass_init = T_mass_init
        self.T_vent = T_vent_init
        self.T_vent_init = T_vent_init
        self.T_ptc = T_ptc_init
        self.T_ptc_init = T_ptc_init
        self.C_CO2 = C_CO2_init
        self.C_CO2_init = C_CO2_init

        # Control inputs
        self.u_hvac = 0.0       # Compressor RPM modulation [0-1]
        self.u_ptc = 0.0        # PTC heater modulation [0-1]
        self.u_blower = 0.1     # Blower fan speed [0.1-1.0], start at minimum
        self.u_recirc = 0.5     # Recirculation [0-1]: 0=fresh, 1=recirc

        # Operating mode (determined automatically)
        self.mode = 'idle'      # 'cooling', 'heating', 'heating_ptc', 'idle'

        # Disturbance scenario
        self.scenario = None

        # Energy tracking
        self.E_hp_total = 0.0   # Total HP electrical energy [J]
        self.E_ptc_total = 0.0  # Total PTC electrical energy [J]

    def setup(self, start_time: int, scenario: str = 'summer_city', hvac_mode: str = None,
              duration: int = 4 * 3600, seed: int = None,
              profile_overrides: dict = None, init_overrides: dict = None, **kwargs):
        """
        Initialize simulation at given start time.

        Args:
            start_time: Unix timestamp
            scenario: 'summer_city', 'winter_highway', 'winter_city', 'mild_mixed'
            hvac_mode: 'cooling', 'heating', or 'auto' (default: based on scenario)
            duration: Simulation duration [s] for pre-generating drive cycle
            seed: Random seed for reproducibility (None = random)
            profile_overrides: Dict of external profile functions for deterministic scenarios.
                Keys: 'T_ambient', 'solar_radiation', 'n_passengers', 'v_vehicle', 'heading', 'soc'
                Values: Callable(t_seconds_since_start) -> value
            init_overrides: Dict of initial condition overrides.
                Keys: 'T_cabin', 'T_mass', 'T_vent', 'T_ptc', 'C_CO2'
        """
        self.time = start_time
        self.start_time = start_time
        self.C_CO2 = self.C_CO2_init
        self.u_hvac = 0.0
        self.u_ptc = 0.0
        self.u_blower = 0.1  # Start at minimum for fair comparison
        self.u_recirc = 0.5
        self.mode = 'idle'
        self.scenario = scenario
        self.E_hp_total = 0.0
        self.E_ptc_total = 0.0
        self.E_blower_total = 0.0

        # Store external profile overrides (for deterministic paper scenarios)
        self._profile_overrides = profile_overrides or {}

        # Set HVAC mode based on scenario if not specified
        if hvac_mode is not None:
            self.scenario_mode = hvac_mode
        elif 'summer' in scenario:
            self.scenario_mode = 'cooling'
        elif 'winter' in scenario:
            self.scenario_mode = 'heating'
        else:
            self.scenario_mode = 'auto'

        # Generate stochastic drive cycle (velocity + passengers)
        # Only used if no external profiles provided
        if 'v_vehicle' not in self._profile_overrides or 'n_passengers' not in self._profile_overrides:
            self._generate_drive_cycle(duration, seed)

        # Scenario-dependent initial conditions (parked vehicle)
        if 'summer' in scenario:
            # Parked in sun: cabin air hot, dashboard/mass even hotter
            self.T_cabin = 273.15 + 35.0   # 35°C
            self.T_mass = 273.15 + 45.0    # 45°C (dashboard in direct sun)
            self.T_vent = 273.15 + 35.0    # Duct at cabin temp
            self.T_ptc = 273.15 + 35.0     # PTC element at cabin temp
        elif 'winter' in scenario:
            # Parked in cold: cabin at ambient temperature
            dist = self._get_disturbances(start_time)
            self.T_cabin = dist['T_ambient']
            self.T_mass = dist['T_ambient']
            self.T_vent = dist['T_ambient']
            self.T_ptc = dist['T_ambient']
        else:
            # Mild: close to ambient
            dist = self._get_disturbances(start_time)
            self.T_cabin = dist['T_ambient']
            self.T_mass = dist['T_ambient']
            self.T_vent = dist['T_ambient']
            self.T_ptc = dist['T_ambient']

        # Apply initial condition overrides (for paper scenarios)
        if init_overrides:
            if 'T_cabin' in init_overrides:
                self.T_cabin = init_overrides['T_cabin']
            if 'T_mass' in init_overrides:
                self.T_mass = init_overrides['T_mass']
            if 'T_vent' in init_overrides:
                self.T_vent = init_overrides['T_vent']
            if 'T_ptc' in init_overrides:
                self.T_ptc = init_overrides['T_ptc']
            if 'C_CO2' in init_overrides:
                self.C_CO2 = init_overrides['C_CO2']

        # Determine if using external profiles
        using_external = len(self._profile_overrides) > 0

        print(f"Robotaxi Cabin Simulator initialized (4-node model)")
        print(f"  Scenario: {scenario}" + (" (with external profiles)" if using_external else ""))
        print(f"  HVAC mode: {self.scenario_mode.upper()}")
        print(f"  Start time: {start_time}")
        print(f"  Initial T_cabin: {self.T_cabin - 273.15:.1f}°C")
        print(f"  Initial T_vent:  {self.T_vent - 273.15:.1f}°C")
        print(f"  Initial T_mass:  {self.T_mass - 273.15:.1f}°C")
        print(f"  Initial CO2:     {self.C_CO2:.0f} ppm")
        print(f"  Target T_cabin: {self.T_target - 273.15:.1f}°C")
        print(f"  PTC threshold: {self.T_ptc_threshold - 273.15:.1f}°C")
        if using_external:
            print(f"  External profiles: {list(self._profile_overrides.keys())}")

    def advance(self):
        """
        Advance simulation by one time step using Euler integration.
        Four-node model: T_ptc (element) + T_vent (duct) + T_cabin (air) + T_mass (interior) + CO2.
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
        # Air flow and inlet temperature
        # =====================================================================
        fresh_frac = max(self.min_fresh_frac, 1 - self.u_recirc)
        m_dot = self.m_dot_blower_max * self.u_blower  # Total air flow through HVAC

        # Inlet air temperature: mix of fresh air and recirculated cabin air
        T_inlet = fresh_frac * T_amb + (1 - fresh_frac) * self.T_cabin

        # Fresh air mass flow for CO2 calculation
        m_dot_fresh = m_dot * fresh_frac

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

        # 5. Heat Pump (with PLR-dependent COP) - heat goes to T_vent
        # Note: HP heat transfer to air depends on blower (convection coefficient
        # increases with air velocity). We scale Q_hp by u_blower to model this.
        eta_radiator = self._radiator_efficiency(v)
        COP_cool = self._cop_cooling(T_amb, self.T_cabin)
        COP_heat = self._cop_heating(T_amb)

        Q_hp = 0.0  # Heat to T_vent
        P_hp_elec = 0.0

        if self.mode == 'cooling':
            # PLR correction: COP improves at partial load
            COP_cool_eff = COP_cool * (1 + self.alpha_plr * (1 - self.u_hvac))
            # Q_hp scales with blower: no air flow = no heat transfer to air
            Q_hp = -self.u_hvac * self.u_blower * self.Q_hp_max_cool * eta_radiator
            # Electrical power is what the compressor draws, not affected by blower
            P_hp_elec = abs(self.u_hvac * self.Q_hp_max_cool * eta_radiator) / COP_cool_eff

        elif self.mode in ['heating', 'heating_ptc']:
            # Q_hp_max_heat is max THERMAL power [W], COP converts thermal to electrical
            COP_heat_eff = COP_heat * (1 + self.alpha_plr * (1 - self.u_hvac))
            # Q_hp scales with blower: no air flow = no heat transfer to air
            Q_hp = self.u_hvac * self.u_blower * self.Q_hp_max_heat * eta_radiator
            P_hp_elec = (self.u_hvac * self.Q_hp_max_heat * eta_radiator) / COP_heat_eff

        # 6. PTC Heater (independent control, only available when T_amb < threshold)
        #    If no controller sets u_ptc, mirror u_hvac (PID compatibility)
        P_ptc_elec = 0.0

        if self.mode == 'heating_ptc':
            if not hasattr(self, '_ptc_externally_controlled') or not self._ptc_externally_controlled:
                self.u_ptc = self.u_hvac
            P_ptc_elec = self.u_ptc * self.Q_ptc_max

        # Track energy consumption
        self.E_hp_total += P_hp_elec * self.step_size
        self.E_ptc_total += P_ptc_elec * self.step_size

        # =====================================================================
        # Sub-stepped Euler integration for fast dynamics (T_ptc, T_vent)
        # =====================================================================
        # The HVAC duct (C_hvac=5000 J/K) and PTC element (C_ptc=1500 J/K) have
        # fast time constants relative to the control step (60s). We use
        # sub-stepping for numerical stability.
        n_substeps = 60  # 1-second sub-steps
        dt_sub = self.step_size / n_substeps

        for _ in range(n_substeps):
            # PTC element: C_ptc * dT_ptc/dt = P_ptc_elec - h_ptc * (T_ptc - T_vent)
            Q_ptc_to_vent = self.h_ptc * (self.T_ptc - self.T_vent)
            dT_ptc = (P_ptc_elec - Q_ptc_to_vent) * dt_sub / self.C_ptc
            self.T_ptc += dT_ptc

            # HVAC duct: C_hvac * dT_vent/dt = Q_hp + Q_ptc_to_vent - m_dot*c_p*(T_vent - T_inlet)
            Q_to_cabin = m_dot * self.c_p_air * (self.T_vent - T_inlet)
            dT_vent = (Q_hp + Q_ptc_to_vent - Q_to_cabin) * dt_sub / self.C_hvac
            self.T_vent += dT_vent

            # Physical limits on T_vent (evaporator icing / safety)
            self.T_vent = np.clip(self.T_vent, self.T_vent_min, self.T_vent_max)

        # Final Q values for logging (use last sub-step values)
        Q_ptc_to_vent = self.h_ptc * (self.T_ptc - self.T_vent)

        # =====================================================================
        # Euler integration: Cabin air node (T_cabin) - slower dynamics, full step OK
        # =====================================================================
        # C_cabin * dT/dt = m_dot*c_p*(T_vent - T_cabin) + Q_solar_air + Q_passengers
        #                 + Q_transmission + Q_conv
        Q_from_hvac = m_dot * self.c_p_air * (self.T_vent - self.T_cabin)
        Q_total_air = Q_from_hvac + Q_solar_air + Q_passengers + Q_transmission + Q_conv_mass_to_air
        dT_air = Q_total_air * self.step_size / self.C_cabin
        self.T_cabin += dT_air

        # =====================================================================
        # Euler integration: Mass node (T_mass) - slow dynamics
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
        self._last_Q_ptc_to_vent = Q_ptc_to_vent
        self._last_Q_from_hvac = Q_from_hvac
        self._last_T_inlet = T_inlet
        self._last_COP = COP_cool if self.mode == 'cooling' else COP_heat

        # Advance time
        self.time += self.step_size

    def read(self) -> dict:
        """Read current state and disturbances."""
        dist = self._get_disturbances(self.time)

        # Total HVAC electrical power
        P_hvac_total = getattr(self, '_last_P_hp', 0) + getattr(self, '_last_P_ptc', 0)

        # Fresh air flow calculation
        fresh_frac = max(self.min_fresh_frac, 1 - self.u_recirc)
        m_dot_fresh = self.m_dot_blower_max * self.u_blower * fresh_frac

        return {
            'time': self.time,
            # States (4 thermal nodes + CO2)
            'cabin_temperature': self.T_cabin,
            'vent_temperature': self.T_vent,
            'ptc_temperature': self.T_ptc,
            'mass_temperature': self.T_mass,  # Not measurable, for debugging only
            'co2_concentration': self.C_CO2,
            'hvac_power': P_hvac_total,
            # Controls
            'hvac_modulation': self.u_hvac,
            'ptc_modulation': self.u_ptc,
            'blower_modulation': self.u_blower,
            'recirc_modulation': self.u_recirc,
            # Disturbances
            'ambient_temperature': dist['T_ambient'],
            'solar_irradiance': dist['solar_radiation'],
            'vehicle_heading': dist['heading'],
            'passenger_count': dist['n_passengers'],
            'vehicle_speed': dist['v_vehicle'],
            'battery_soc': dist['soc'],
            # Additional info
            'inlet_temperature': getattr(self, '_last_T_inlet', self.T_cabin),
            'battery_power_hvac': P_hvac_total,
            'hp_power': getattr(self, '_last_P_hp', 0),
            'ptc_power': getattr(self, '_last_P_ptc', 0),
            'hvac_mode': 1 if self.mode == 'cooling' else (-1 if 'heating' in self.mode else 0),
            'cop': getattr(self, '_last_COP', 1.0),
            'fresh_air_flow': m_dot_fresh,
        }

    def write(self, values: dict):
        """Write control values to simulator."""
        if 'hvac_modulation' in values:
            self.u_hvac = np.clip(values['hvac_modulation'], 0.0, 1.0)
        if 'ptc_modulation' in values:
            self.u_ptc = np.clip(values['ptc_modulation'], 0.0, 1.0)
            self._ptc_externally_controlled = True
        if 'blower_modulation' in values:
            self.u_blower = np.clip(values['blower_modulation'], 0.1, 1.0)
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

    def _generate_drive_cycle(self, duration: int, seed: int = None):
        """
        Generate stochastic drive cycle for urban robotaxi.

        Based on realistic urban robotaxi behavior:
        - STOP:       25% - Traffic lights, pickup/dropoff, congestion
        - SLOW_URBAN: 35% - Dense city (10-30 km/h)
        - FAST_URBAN: 30% - Main roads (30-50 km/h)
        - SUBURBAN:    5% - Rare faster roads (50-70 km/h)
        - DECELERATE:  5% - Approaching stops/turns

        CRITICAL: Passengers can ONLY board/alight when vehicle is stopped (v=0).
        """
        if seed is not None:
            np.random.seed(seed)

        n_steps = duration // self.step_size + 1
        self._v_profile = np.zeros(n_steps)
        self._n_passengers = np.zeros(n_steps, dtype=int)

        # Acceleration/deceleration (m/s per step)
        max_accel = 1.0 * self.step_size  # ~1 m/s²
        max_decel = 1.5 * self.step_size  # ~1.5 m/s²

        # State
        idx = 0
        current_v = 0.0
        current_passengers = np.random.randint(0, 5)
        last_passenger_change_idx = -int(120 / self.step_size)

        # Helper: decelerate to stop - MUST end with v=0 written to array
        def decel_to_stop():
            nonlocal idx, current_v
            # Decelerate until nearly stopped
            while current_v > 0.5 and idx < n_steps:
                current_v = max(0, current_v - max_decel)
                self._v_profile[idx] = current_v
                self._n_passengers[idx] = current_passengers
                idx += 1
            # Write final v=0 step (CRITICAL: must happen before passenger change)
            if idx < n_steps:
                current_v = 0.0
                self._v_profile[idx] = 0.0
                self._n_passengers[idx] = current_passengers
                idx += 1

        # Helper: accelerate to target
        def accel_to(target_v):
            nonlocal idx, current_v
            while current_v < target_v - 0.3 and idx < n_steps:
                current_v = min(target_v, current_v + max_accel)
                self._v_profile[idx] = current_v
                self._n_passengers[idx] = current_passengers
                idx += 1

        # Helper: passenger change during stop (called ONCE per stop, not per step)
        def do_passenger_change():
            nonlocal current_passengers, last_passenger_change_idx
            time_since_change = (idx - last_passenger_change_idx) * self.step_size
            # Minimum 3 minutes between changes
            if time_since_change > 180:
                change = np.random.choice([-2, -1, 1, 2])
                current_passengers = max(0, min(4, current_passengers + change))
                last_passenger_change_idx = idx

        # Helper: execute a stop with potential passenger change
        def execute_stop(min_duration: float, max_duration: float, force_passenger_change: bool = False):
            nonlocal idx, current_passengers, last_passenger_change_idx
            stop_steps = max(1, int((min_duration + np.random.random() * (max_duration - min_duration)) / self.step_size))
            end_idx = min(idx + stop_steps, n_steps)

            # Decide passenger change at START of stop (not during)
            time_since_change = (idx - last_passenger_change_idx) * self.step_size
            will_change = force_passenger_change or (time_since_change > 180 and np.random.random() < 0.4)

            if will_change:
                change = np.random.choice([-2, -1, 1, 2])
                current_passengers = max(0, min(4, current_passengers + change))
                last_passenger_change_idx = idx

            # Fill stop duration with v=0 and constant passengers
            while idx < end_idx:
                self._v_profile[idx] = 0.0
                self._n_passengers[idx] = current_passengers
                idx += 1

        while idx < n_steps:
            segment_type = np.random.random()

            if segment_type < 0.25:
                # === STOP (25%): Traffic light, pickup/dropoff ===
                decel_to_stop()
                execute_stop(min_duration=15, max_duration=105)

            elif segment_type < 0.60:
                # === SLOW URBAN (35%): 10-30 km/h → 2.8-8.3 m/s ===
                target_v = 2.8 + np.random.random() * 5.5
                cruise_steps = int((30 + np.random.random() * 120) / self.step_size)
                accel_to(target_v)
                for k in range(cruise_steps):
                    if idx >= n_steps:
                        break
                    noise = 1.2 * np.sin(2 * np.pi * k / 50) + np.random.randn() * 0.6
                    self._v_profile[idx] = max(0.5, target_v + noise)  # Never fully stop while cruising
                    self._n_passengers[idx] = current_passengers
                    current_v = self._v_profile[idx]
                    idx += 1

            elif segment_type < 0.90:
                # === FAST URBAN (30%): 30-50 km/h → 8.3-13.9 m/s ===
                target_v = 8.3 + np.random.random() * 5.6
                cruise_steps = int((45 + np.random.random() * 150) / self.step_size)
                accel_to(target_v)
                for k in range(cruise_steps):
                    if idx >= n_steps:
                        break
                    noise = 0.8 * np.sin(2 * np.pi * k / 70) + np.random.randn() * 0.4
                    self._v_profile[idx] = max(0.5, target_v + noise)
                    self._n_passengers[idx] = current_passengers
                    current_v = self._v_profile[idx]
                    idx += 1

            elif segment_type < 0.95:
                # === SUBURBAN (5%): 50-70 km/h → 13.9-19.4 m/s ===
                target_v = 13.9 + np.random.random() * 5.5
                cruise_steps = int((60 + np.random.random() * 90) / self.step_size)
                accel_to(target_v)
                for k in range(cruise_steps):
                    if idx >= n_steps:
                        break
                    noise = 0.5 * np.sin(2 * np.pi * k / 90) + np.random.randn() * 0.25
                    self._v_profile[idx] = max(0.5, target_v + noise)
                    self._n_passengers[idx] = current_passengers
                    current_v = self._v_profile[idx]
                    idx += 1

            else:
                # === DECELERATE (5%): slow down for turn/obstacle ===
                target_v = max(0.5, current_v * (0.2 + np.random.random() * 0.3))
                while current_v > target_v + 0.3 and idx < n_steps:
                    current_v = max(target_v, current_v - max_decel)
                    self._v_profile[idx] = current_v
                    self._n_passengers[idx] = current_passengers
                    idx += 1

            # Force a stop for passenger change every ~8-12 minutes
            time_since_change = (idx - last_passenger_change_idx) * self.step_size
            if time_since_change > 480 + np.random.random() * 240:
                decel_to_stop()
                execute_stop(min_duration=30, max_duration=90, force_passenger_change=True)

        # Smooth velocity only (NOT passengers) - and preserve v=0 at stops
        if n_steps > 5:
            v_smoothed = np.convolve(self._v_profile, np.ones(3)/3, mode='same')
            # Only apply smoothing where v > 0.5 (preserve stops)
            mask_driving = self._v_profile > 0.5
            self._v_profile[mask_driving] = v_smoothed[mask_driving]
            self._v_profile = np.maximum(0, self._v_profile)

        # VERIFICATION: Ensure passengers only change when stopped
        # np.diff at index i means n_passengers[i] != n_passengers[i+1]
        # The NEW passenger count starts at index i+1, so check v_profile[i+1]
        passenger_changes = np.where(np.diff(self._n_passengers) != 0)[0]
        for change_idx in passenger_changes:
            new_idx = change_idx + 1  # Index where new passenger count starts
            if new_idx < len(self._v_profile) and self._v_profile[new_idx] > 0.1:
                # This should never happen - fix by forcing stop
                print(f"WARNING: Passenger change at idx={new_idx} with v={self._v_profile[new_idx]:.1f} m/s - fixing")
                self._v_profile[new_idx] = 0.0

    def _get_passengers(self, time: int) -> int:
        """Get passenger count from pre-generated drive cycle."""
        idx = (time - self.start_time) // self.step_size
        if hasattr(self, '_n_passengers') and 0 <= idx < len(self._n_passengers):
            return int(self._n_passengers[idx])
        # Fallback for times outside pre-generated range
        return np.random.randint(0, 5)

    def _get_velocity_profile(self, time: int) -> float:
        """Get velocity from pre-generated drive cycle."""
        idx = (time - self.start_time) // self.step_size
        if hasattr(self, '_v_profile') and 0 <= idx < len(self._v_profile):
            return float(self._v_profile[idx])
        # Fallback
        return 5.0 + np.random.random() * 5.0

    def _get_disturbances(self, time: int) -> dict:
        """
        Generate disturbance values for given time based on scenario.

        If external profile overrides are provided (via setup), those are used
        instead of the scenario-based calculations. This enables deterministic
        paper scenarios.
        """
        # Time since simulation start [s] - for external profiles
        t_rel = time - self.start_time

        # Time of day [0-24h] - for scenario-based profiles
        hour = (time % 86400) / 3600.0

        # Check for external profile overrides
        overrides = getattr(self, '_profile_overrides', {})

        # --- Passenger count ---
        if 'n_passengers' in overrides:
            n_pass = int(overrides['n_passengers'](t_rel))
        else:
            n_pass = self._get_passengers(time)

        # --- Vehicle velocity ---
        if 'v_vehicle' in overrides:
            v = overrides['v_vehicle'](t_rel)
        else:
            v = self._get_velocity_profile(time)

        # --- Ambient temperature ---
        if 'T_ambient' in overrides:
            T_amb = overrides['T_ambient'](t_rel)
        elif self.scenario == 'summer_city':
            T_amb = 303.15 + 5 * np.sin((hour - 6) * np.pi / 12)  # 30-35°C
        elif self.scenario == 'winter_highway':
            T_amb = 263.15 + 5 * np.sin((hour - 6) * np.pi / 12)  # -10 to -5°C
        elif self.scenario == 'winter_city':
            T_amb = 268.15 + 3 * np.sin((hour - 6) * np.pi / 12)  # -5 to -2°C
        else:  # 'mild_mixed'
            T_amb = 293.15 + 4 * np.sin((hour - 6) * np.pi / 12)  # 20-24°C

        # --- Solar radiation ---
        if 'solar_radiation' in overrides:
            I_solar = overrides['solar_radiation'](t_rel)
        elif self.scenario == 'summer_city':
            I_solar = max(0, 800 * np.sin((hour - 6) * np.pi / 12))
        elif self.scenario == 'winter_highway':
            I_solar = max(0, 200 * np.sin((hour - 8) * np.pi / 8))
        elif self.scenario == 'winter_city':
            I_solar = max(0, 150 * np.sin((hour - 8) * np.pi / 8))
        else:
            I_solar = max(0, 500 * np.sin((hour - 6) * np.pi / 12))

        # --- Heading ---
        if 'heading' in overrides:
            heading = overrides['heading'](t_rel)
        else:
            heading = (time / 100) % (2 * np.pi)

        # --- SOC ---
        if 'soc' in overrides:
            soc = overrides['soc'](t_rel)
        elif self.scenario == 'summer_city':
            soc = max(0.2, 0.8 - (time % 14400) / 72000)
        elif self.scenario == 'winter_highway':
            soc = max(0.2, 0.9 - (time % 18000) / 90000)
        elif self.scenario == 'winter_city':
            soc = max(0.15, 0.7 - (time % 10800) / 36000)
        else:
            soc = max(0.3, 0.85 - (time % 10800) / 54000)

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
        print("Robotaxi Cabin Simulator Summary (4-node model)")
        print("=" * 50)
        print("Thermal Nodes:")
        print(f"  C_cabin (air):   {self.C_cabin:.0f} J/K")
        print(f"  C_hvac (duct):   {self.C_hvac:.0f} J/K")
        print(f"  C_ptc (element): {self.C_ptc:.0f} J/K")
        print(f"  C_mass (interior): {self.C_mass:.0f} J/K")
        print("Envelope:")
        print(f"  UA opaque: {self.UA_opaque:.1f} W/K")
        print(f"  UA window: {self.UA_window:.1f} W/K")
        print(f"  UA total:  {self.UA_opaque + self.UA_window:.1f} W/K")
        print(f"  Window area: {self.A_window:.1f} m²")
        print("HVAC:")
        print(f"  Max HP cooling: {self.Q_hp_max_cool:.0f} W")
        print(f"  Max HP heating: {self.Q_hp_max_heat:.0f} W")
        print(f"  Max PTC power:  {self.Q_ptc_max:.0f} W")
        print(f"  h_ptc (element→duct): {self.h_ptc:.0f} W/K")
        print(f"  PTC threshold: {self.T_ptc_threshold - 273.15:.0f}°C")
        print(f"  Blower max flow: {self.m_dot_blower_max:.3f} kg/s")
        print("Other:")
        print(f"  Cabin volume: {self.V_cabin:.1f} m³")
        print(f"  Step size: {self.step_size} s")
        print(f"  Scenario: {self.scenario}")
        if self.E_hp_total > 0 or self.E_ptc_total > 0:
            print(f"  Total HP energy: {self.E_hp_total / 3600:.1f} Wh")
            print(f"  Total PTC energy: {self.E_ptc_total / 3600:.1f} Wh")
        print("=" * 50)
