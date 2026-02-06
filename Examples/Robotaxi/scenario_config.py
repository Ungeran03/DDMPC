"""
Scenario Configuration for Paper Experiments

This module provides a clean way to configure MPC scenarios for the paper.
Each scenario defines:
- Which MVs are active (with bounds)
- MPC objective weights
- Deterministic disturbance profiles
- Initial conditions
- Metrics to track

Usage:
    from scenario_config import ScenarioConfig, preconditioning_scenario

    config = preconditioning_scenario()
    # Pass config to runner or use directly
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
import numpy as np


@dataclass
class MVConfig:
    """Configuration for a single Manipulated Variable."""
    active: bool = True
    lb: float = 0.0
    ub: float = 1.0
    weight_change: float = 5.0  # Weight on delta-u in MPC objective


@dataclass
class ScenarioConfig:
    """
    Complete configuration for a paper scenario.

    Attributes:
        name: Short identifier (e.g., 'preconditioning')
        description: Human-readable description

        # Simulation settings
        duration_hours: Simulation duration [h]
        start_time_hours: Start time of day [h] (e.g., 10 = 10:00 AM)
        hvac_mode: 'cooling' or 'heating' (for WhiteBox model selection)

        # Initial conditions [K]
        T_cabin_init: Initial cabin air temperature
        T_mass_init: Initial interior mass temperature
        T_vent_init: Initial HVAC duct temperature
        T_ptc_init: Initial PTC element temperature
        C_CO2_init: Initial CO2 concentration [ppm]

        # MV configuration
        mv_config: Dict mapping MV name to MVConfig

        # MPC objective weights
        weights: Dict mapping feature name to weight

        # Disturbance profiles (Callable: time_seconds -> value)
        # If None, uses default scenario-based calculation
        profile_T_ambient: T_ambient(t) [K]
        profile_solar: solar_radiation(t) [W/m²]
        profile_passengers: n_passengers(t) [int]
        profile_velocity: v_vehicle(t) [m/s]
        profile_heading: heading(t) [rad]
        profile_soc: soc(t) [0-1]

        # Metrics to compute after simulation
        metrics: List of metric names to compute

        # Comparison settings
        run_pid_baseline: Whether to run PID for comparison
        pid_params: PID tuning parameters if different from defaults
    """

    # Identification
    name: str
    description: str

    # Simulation
    duration_hours: float = 1.0
    start_time_hours: float = 10.0
    hvac_mode: str = 'cooling'  # 'cooling' or 'heating'

    # Initial conditions
    T_cabin_init: float = 308.15  # 35°C (hot parked car)
    T_mass_init: float = 318.15   # 45°C (dashboard in sun)
    T_vent_init: float = 308.15   # Same as cabin
    T_ptc_init: float = 308.15    # Same as cabin
    C_CO2_init: float = 420.0     # Ambient

    # MV configuration
    mv_config: Dict[str, MVConfig] = field(default_factory=lambda: {
        'u_hvac': MVConfig(active=True, lb=0, ub=1, weight_change=10),
        'u_ptc': MVConfig(active=False, lb=0, ub=1, weight_change=10),
        'u_blower': MVConfig(active=True, lb=0.1, ub=1, weight_change=5),
        'u_recirc': MVConfig(active=True, lb=0, ub=1, weight_change=5),
    })

    # MPC weights (base values, T_cabin scaled by passengers)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'T_cabin': 100.0,      # Band violation penalty
        'T_vent': 1.0,         # Intermediate state tracking
        'T_ptc': 0.1,          # PTC element tracking
        'C_CO2': 50.0,         # CO2 above soft target
        'energy': 0.001,       # Energy penalty (scaled for Watts)
    })

    # Temperature comfort band [K] - no penalty inside this band
    T_comfort_lb: float = 273.15 + 20.0  # 20°C lower comfort bound
    T_comfort_ub: float = 273.15 + 24.0  # 24°C upper comfort bound

    # Passenger-dependent comfort weighting
    # w_eff = w_base * (1 + alpha_passenger * n_passengers)
    alpha_passenger: float = 0.5  # 0.5 means: 4 passengers -> 3x weight

    # CO2 settings
    CO2_soft_target: float = 800.0   # Soft target [ppm] - no penalty below
    CO2_limit: float = 1200.0        # Hard constraint [ppm]

    # Energy approximation parameters
    Q_hp_max: float = 5000.0    # Max HP power [W] (cooling or heating)
    Q_ptc_max: float = 6000.0   # Max PTC power [W]
    COP_nominal: float = 2.5    # Nominal COP for energy approximation

    # Disturbance profiles (None = use scenario defaults)
    profile_T_ambient: Optional[Callable[[float], float]] = None
    profile_solar: Optional[Callable[[float], float]] = None
    profile_passengers: Optional[Callable[[float], int]] = None
    profile_velocity: Optional[Callable[[float], float]] = None
    profile_heading: Optional[Callable[[float], float]] = None
    profile_soc: Optional[Callable[[float], float]] = None

    # Metrics
    metrics: List[str] = field(default_factory=lambda: [
        'T_mean', 'T_std', 'T_max_dev',
        'CO2_max', 'CO2_mean',
        'energy_total_Wh',
    ])

    # PID baseline
    run_pid_baseline: bool = True
    pid_params: Optional[Dict[str, Any]] = None

    def get_profile_overrides(self) -> Dict[str, Callable]:
        """Return dict of non-None profile overrides for simulator."""
        overrides = {}
        if self.profile_T_ambient is not None:
            overrides['T_ambient'] = self.profile_T_ambient
        if self.profile_solar is not None:
            overrides['solar_radiation'] = self.profile_solar
        if self.profile_passengers is not None:
            overrides['n_passengers'] = self.profile_passengers
        if self.profile_velocity is not None:
            overrides['v_vehicle'] = self.profile_velocity
        if self.profile_heading is not None:
            overrides['heading'] = self.profile_heading
        if self.profile_soc is not None:
            overrides['soc'] = self.profile_soc
        return overrides

    def get_active_mvs(self) -> List[str]:
        """Return list of active MV names."""
        return [name for name, cfg in self.mv_config.items() if cfg.active]

    def summary(self):
        """Print scenario summary."""
        print(f"\n{'='*60}")
        print(f"Scenario: {self.name}")
        print(f"{'='*60}")
        print(f"Description: {self.description}")
        print(f"\nSimulation:")
        print(f"  Duration: {self.duration_hours} hours")
        print(f"  Start time: {self.start_time_hours}:00")
        print(f"  HVAC mode: {self.hvac_mode}")
        print(f"\nInitial Conditions:")
        print(f"  T_cabin: {self.T_cabin_init - 273.15:.1f}°C")
        print(f"  T_mass:  {self.T_mass_init - 273.15:.1f}°C")
        print(f"  CO2:     {self.C_CO2_init:.0f} ppm")
        print(f"\nActive MVs:")
        for name, cfg in self.mv_config.items():
            if cfg.active:
                print(f"  {name}: [{cfg.lb}, {cfg.ub}], weight_change={cfg.weight_change}")
        print(f"\nMPC Objective:")
        print(f"  Temperature band: [{self.T_comfort_lb - 273.15:.0f}, {self.T_comfort_ub - 273.15:.0f}]°C")
        print(f"  Passenger scaling: w_eff = w_base × (1 + {self.alpha_passenger} × n_pass)")
        print(f"  CO2 soft target: {self.CO2_soft_target:.0f} ppm")
        print(f"  Weights: T={self.weights.get('T_cabin', 0)}, CO2={self.weights.get('C_CO2', 0)}, E={self.weights.get('energy', 0)}")
        print(f"\nConstraints:")
        print(f"  CO2 hard limit: {self.CO2_limit} ppm")
        print(f"\nExternal Profiles:")
        profiles = self.get_profile_overrides()
        if profiles:
            for name in profiles:
                print(f"  {name}: custom function")
        else:
            print("  (using scenario defaults)")
        print(f"{'='*60}\n")


# =============================================================================
# SCENARIO 1: Pre-Conditioning Before Boarding
# =============================================================================

def preconditioning_scenario() -> ScenarioConfig:
    """
    Scenario 1: Pre-Conditioning Before Boarding

    The robotaxi knows from the booking system that 3 passengers will board
    at t=5min. MPC should pre-cool the cabin and pre-ventilate (lower CO2)
    before they arrive.

    Timeline:
    - t=0-5min:   Empty cabin, parked in sun (35°C)
    - t=5min:     3 passengers board (vehicle stopped)
    - t=5-25min:  3 passengers in cabin
    - t=25min:    Passengers alight
    - t=25-30min: Empty again

    Key insight: MPC sees n_passengers forecast and acts BEFORE boarding.
    PID only reacts AFTER temperature/CO2 changes.
    """

    # Deterministic passenger profile
    def passenger_profile(t_sec: float) -> int:
        """3 passengers board at t=8min, alight at t=25min."""
        if t_sec < 8 * 60:  # Longer pre-conditioning window
            return 0
        elif t_sec < 25 * 60:
            return 3
        else:
            return 0

    # Vehicle stationary for this scenario (focus on passenger boarding)
    def velocity_profile(t_sec: float) -> float:
        """Vehicle stopped for passenger exchange, then slow city driving."""
        if t_sec < 5 * 60:
            return 0.0  # Parked, waiting
        elif t_sec < 6 * 60:
            return 0.0  # Stopped for boarding
        else:
            return 5.0  # Slow city driving ~18 km/h

    # Constant ambient (simplify for first scenario)
    def ambient_profile(t_sec: float) -> float:
        """Constant hot ambient: 32°C."""
        return 273.15 + 32.0

    # Constant solar
    def solar_profile(t_sec: float) -> float:
        """Strong sun: 800 W/m²."""
        return 800.0

    # Constant heading
    def heading_profile(t_sec: float) -> float:
        """Constant heading."""
        return 0.0

    # SOC not relevant for this scenario
    def soc_profile(t_sec: float) -> float:
        """High SOC (not a constraint here)."""
        return 0.8

    return ScenarioConfig(
        name="preconditioning",
        description="Pre-Conditioning Before Passenger Boarding",

        duration_hours=0.5,  # 30 min is enough
        start_time_hours=10.0,
        hvac_mode='cooling',

        # Moderately warm cabin (after short parking in shade)
        # MPC should pre-cool knowing passengers board at t=5min
        T_cabin_init=273.15 + 28.0,  # 28°C - above target but achievable
        T_mass_init=273.15 + 30.0,   # 30°C
        T_vent_init=273.15 + 28.0,
        T_ptc_init=273.15 + 28.0,
        C_CO2_init=420.0,

        # All cooling-relevant MVs active
        mv_config={
            'u_hvac': MVConfig(active=True, lb=0, ub=1, weight_change=10),
            'u_ptc': MVConfig(active=False),  # Summer, not needed
            'u_blower': MVConfig(active=True, lb=0.1, ub=1, weight_change=5),
            'u_recirc': MVConfig(active=True, lb=0, ub=1, weight_change=5),
        },

        # MPC objective weights (tuned for aggressive cooling + good CO2)
        weights={
            'T_cabin': 500.0,      # High weight for temperature tracking
            'T_vent': 1.0,         # Intermediate state
            'T_ptc': 0.0,          # Not tracking PTC in cooling
            'C_CO2': 30.0,         # Moderate CO2 weight to stay away from limit
            'energy': 0.0001,      # Lower energy penalty for faster response
        },

        # Temperature comfort band
        T_comfort_lb=273.15 + 20.0,  # 20°C
        T_comfort_ub=273.15 + 24.0,  # 24°C

        # Passenger-dependent comfort weighting
        alpha_passenger=0.5,  # w_eff = w_base * (1 + 0.5 * n_pass)

        # CO2 settings
        CO2_soft_target=800.0,   # No penalty below this
        CO2_limit=1200.0,        # Hard constraint

        # Deterministic profiles
        profile_T_ambient=ambient_profile,
        profile_solar=solar_profile,
        profile_passengers=passenger_profile,
        profile_velocity=velocity_profile,
        profile_heading=heading_profile,
        profile_soc=soc_profile,

        metrics=[
            'T_at_boarding',      # T_cabin when passengers board (t=5min)
            'CO2_at_boarding',    # CO2 when passengers board
            'T_overshoot',        # Max deviation after boarding
            'T_settling_time',    # Time to reach ±1K of target after boarding
            'energy_total_Wh',
            'energy_preconditioning_Wh',  # Energy used before boarding
        ],

        run_pid_baseline=True,
        pid_params={
            'Kp': 0.3,
            'Ti': 100.0,
            'reverse_act': True,
        },
    )


# =============================================================================
# SCENARIO 2: Highway Speed Anticipation
# =============================================================================

def highway_anticipation_scenario() -> ScenarioConfig:
    """
    Scenario 2: Highway Speed Anticipation

    MPC knows the vehicle will accelerate to highway speed in 3 minutes.
    It can delay cooling until eta_radiator improves (0.5 -> 0.9).

    Timeline:
    - t=0-3min:   City driving, v=5 m/s, eta=0.65
    - t=3-5min:   Acceleration to highway
    - t=5-25min:  Highway cruising, v=25 m/s, eta=0.90
    - t=25-30min: Deceleration back to city
    """

    def velocity_profile(t_sec: float) -> float:
        """Accelerate to highway at t=3min."""
        if t_sec < 3 * 60:
            return 5.0  # City: ~18 km/h
        elif t_sec < 5 * 60:
            # Smooth acceleration
            progress = (t_sec - 3*60) / (2*60)
            return 5.0 + progress * 20.0
        elif t_sec < 25 * 60:
            return 25.0  # Highway: ~90 km/h
        else:
            # Deceleration
            progress = (t_sec - 25*60) / (5*60)
            return max(5.0, 25.0 - progress * 20.0)

    def passenger_profile(t_sec: float) -> int:
        """2 passengers throughout."""
        return 2

    def ambient_profile(t_sec: float) -> float:
        """Hot ambient: 33°C."""
        return 273.15 + 33.0

    def solar_profile(t_sec: float) -> float:
        """Moderate sun."""
        return 600.0

    return ScenarioConfig(
        name="highway_anticipation",
        description="Highway Speed Anticipation for Radiator Efficiency",

        duration_hours=0.5,
        start_time_hours=14.0,
        hvac_mode='cooling',

        # Warm cabin (just got in)
        T_cabin_init=273.15 + 30.0,
        T_mass_init=273.15 + 35.0,
        T_vent_init=273.15 + 30.0,
        T_ptc_init=273.15 + 30.0,
        C_CO2_init=600.0,  # Some CO2 from previous passengers

        mv_config={
            'u_hvac': MVConfig(active=True, lb=0, ub=1, weight_change=10),
            'u_ptc': MVConfig(active=False),
            'u_blower': MVConfig(active=True, lb=0.1, ub=1, weight_change=5),
            'u_recirc': MVConfig(active=True, lb=0, ub=1, weight_change=5),
        },

        weights={
            'T_cabin': 100.0,
            'T_vent': 1.0,
            'T_ptc': 0.0,
            'C_CO2': 50.0,
            'energy': 0.001,  # Higher energy penalty to encourage waiting for better eta
        },

        profile_T_ambient=ambient_profile,
        profile_solar=solar_profile,
        profile_passengers=passenger_profile,
        profile_velocity=velocity_profile,

        metrics=[
            'energy_city_Wh',      # Energy during city phase
            'energy_highway_Wh',   # Energy during highway phase
            'energy_total_Wh',
            'T_mean',
            'T_max_dev',
            'avg_eta_radiator',
        ],
    )


# =============================================================================
# SCENARIO 3: Temperature Peak Shaving
# =============================================================================

def peak_shaving_scenario() -> ScenarioConfig:
    """
    Scenario 3: Temperature Peak Shaving

    MPC sees that T_ambient will rise from 25°C to 35°C over 15 minutes.
    It pre-cools before the peak and uses T_mass as thermal buffer.

    Timeline:
    - t=0-5min:   Mild ambient (25°C)
    - t=5-20min:  Rising ambient (25°C -> 35°C)
    - t=20-30min: Peak ambient (35°C)
    """

    def ambient_profile(t_sec: float) -> float:
        """Rising ambient temperature."""
        if t_sec < 5 * 60:
            return 273.15 + 25.0
        elif t_sec < 20 * 60:
            progress = (t_sec - 5*60) / (15*60)
            return 273.15 + 25.0 + progress * 10.0
        else:
            return 273.15 + 35.0

    def solar_profile(t_sec: float) -> float:
        """Rising solar with ambient."""
        if t_sec < 5 * 60:
            return 400.0
        elif t_sec < 20 * 60:
            progress = (t_sec - 5*60) / (15*60)
            return 400.0 + progress * 500.0
        else:
            return 900.0

    def passenger_profile(t_sec: float) -> int:
        """2 passengers throughout."""
        return 2

    def velocity_profile(t_sec: float) -> float:
        """Moderate city driving."""
        return 8.0  # ~29 km/h

    return ScenarioConfig(
        name="peak_shaving",
        description="Temperature Peak Shaving with Thermal Mass",

        duration_hours=0.5,
        start_time_hours=12.0,
        hvac_mode='cooling',

        # Start at comfortable temperature
        T_cabin_init=273.15 + 23.0,
        T_mass_init=273.15 + 24.0,
        T_vent_init=273.15 + 23.0,
        T_ptc_init=273.15 + 23.0,
        C_CO2_init=500.0,

        mv_config={
            'u_hvac': MVConfig(active=True, lb=0, ub=1, weight_change=10),
            'u_ptc': MVConfig(active=False),
            'u_blower': MVConfig(active=True, lb=0.1, ub=1, weight_change=5),
            'u_recirc': MVConfig(active=True, lb=0, ub=1, weight_change=5),
        },

        weights={
            'T_cabin': 100.0,
            'T_vent': 1.0,
            'T_ptc': 0.0,
            'C_CO2': 30.0,  # Lower priority, focus on temperature
        },

        profile_T_ambient=ambient_profile,
        profile_solar=solar_profile,
        profile_passengers=passenger_profile,
        profile_velocity=velocity_profile,

        metrics=[
            'max_u_hvac',          # Peak compressor load
            'energy_precool_Wh',   # Energy during pre-cooling
            'energy_peak_Wh',      # Energy during peak
            'energy_total_Wh',
            'T_mean',
            'T_max',
        ],
    )


# =============================================================================
# SCENARIO 4: CO2 Management
# =============================================================================

def co2_management_scenario() -> ScenarioConfig:
    """
    Scenario 4: CO2 vs. Energy Trade-off

    4 passengers in stop-and-go traffic. MPC must coordinate u_blower
    and u_recirc to keep CO2 < 1200 ppm while minimizing energy.

    PID baseline has u_recirc=0.5 fixed and BlowerPI only reacts to
    temperature, leading to CO2 peaks of 2000-4000 ppm.
    """

    def passenger_profile(t_sec: float) -> int:
        """Full cabin: 4 passengers."""
        return 4

    def velocity_profile(t_sec: float) -> float:
        """Stop-and-go traffic."""
        # Simulate traffic: alternate between stop and slow
        cycle = int(t_sec / 60) % 4  # 4-minute cycle
        if cycle == 0:
            return 0.0  # Stopped
        elif cycle == 1:
            return 3.0  # Crawling
        elif cycle == 2:
            return 6.0  # Moving
        else:
            return 2.0  # Slowing

    def ambient_profile(t_sec: float) -> float:
        """Moderate temperature."""
        return 273.15 + 28.0

    def solar_profile(t_sec: float) -> float:
        """Moderate sun."""
        return 500.0

    return ScenarioConfig(
        name="co2_management",
        description="CO2 vs. Energy Trade-off with Full Cabin",

        duration_hours=0.5,
        start_time_hours=17.0,  # Rush hour
        hvac_mode='cooling',

        # Start comfortable but with elevated CO2
        T_cabin_init=273.15 + 24.0,
        T_mass_init=273.15 + 26.0,
        T_vent_init=273.15 + 24.0,
        T_ptc_init=273.15 + 24.0,
        C_CO2_init=800.0,  # Already elevated

        mv_config={
            'u_hvac': MVConfig(active=True, lb=0, ub=1, weight_change=10),
            'u_ptc': MVConfig(active=False),
            'u_blower': MVConfig(active=True, lb=0.1, ub=1, weight_change=5),
            'u_recirc': MVConfig(active=True, lb=0, ub=1, weight_change=5),
        },

        weights={
            'T_cabin': 80.0,
            'T_vent': 1.0,
            'T_ptc': 0.0,
            'C_CO2': 100.0,  # High priority on CO2!
        },

        CO2_limit=1200.0,

        profile_T_ambient=ambient_profile,
        profile_solar=solar_profile,
        profile_passengers=passenger_profile,
        profile_velocity=velocity_profile,

        metrics=[
            'CO2_max',
            'CO2_mean',
            'CO2_time_above_1000',   # Time [min] with CO2 > 1000
            'CO2_violations',        # Time [min] with CO2 > 1200
            'energy_ventilation_Wh', # Energy for fresh air
            'energy_total_Wh',
            'T_mean',
        ],
    )


# =============================================================================
# SCENARIO 5: SOC-Dependent Comfort Relaxation
# =============================================================================

def soc_relaxation_scenario() -> ScenarioConfig:
    """
    Scenario 5: SOC-Dependent Comfort Relaxation

    Battery is low (starting at 25%, dropping to ~15%). MPC should
    gradually relax comfort constraints to preserve range.

    This requires stage parameter support in the MPC.
    """

    def soc_profile(t_sec: float) -> float:
        """Declining SOC: 25% -> 15% over 30 min."""
        initial_soc = 0.25
        drain_rate = 0.10 / (30 * 60)  # 10% over 30 min
        return max(0.10, initial_soc - drain_rate * t_sec)

    def passenger_profile(t_sec: float) -> int:
        """2 passengers."""
        return 2

    def velocity_profile(t_sec: float) -> float:
        """Normal city driving."""
        return 8.0

    def ambient_profile(t_sec: float) -> float:
        """Hot ambient."""
        return 273.15 + 33.0

    def solar_profile(t_sec: float) -> float:
        """Strong sun."""
        return 700.0

    return ScenarioConfig(
        name="soc_relaxation",
        description="SOC-Dependent Comfort Relaxation",

        duration_hours=0.5,
        start_time_hours=15.0,
        hvac_mode='cooling',

        # Start at target
        T_cabin_init=273.15 + 22.0,
        T_mass_init=273.15 + 24.0,
        T_vent_init=273.15 + 22.0,
        T_ptc_init=273.15 + 22.0,
        C_CO2_init=500.0,

        mv_config={
            'u_hvac': MVConfig(active=True, lb=0, ub=1, weight_change=10),
            'u_ptc': MVConfig(active=False),
            'u_blower': MVConfig(active=True, lb=0.1, ub=1, weight_change=5),
            'u_recirc': MVConfig(active=True, lb=0, ub=1, weight_change=5),
        },

        # Weights will be modified by SOC via stage parameter
        weights={
            'T_cabin': 100.0,  # This gets scaled by w_comfort(soc)
            'T_vent': 1.0,
            'T_ptc': 0.0,
            'C_CO2': 50.0,
        },

        profile_T_ambient=ambient_profile,
        profile_solar=solar_profile,
        profile_passengers=passenger_profile,
        profile_velocity=velocity_profile,
        profile_soc=soc_profile,

        metrics=[
            'T_mean',
            'T_max_dev',
            'comfort_violation_Kmin',  # Integral of |T - T_target| [K*min]
            'energy_total_Wh',
            'energy_saved_vs_baseline_Wh',
            'final_soc',
        ],
    )


# =============================================================================
# SCENARIO REGISTRY
# =============================================================================

SCENARIOS = {
    'preconditioning': preconditioning_scenario,
    'highway_anticipation': highway_anticipation_scenario,
    'peak_shaving': peak_shaving_scenario,
    'co2_management': co2_management_scenario,
    'soc_relaxation': soc_relaxation_scenario,
}


def get_scenario(name: str) -> ScenarioConfig:
    """Get scenario by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]()


def list_scenarios():
    """Print available scenarios."""
    print("\nAvailable Paper Scenarios:")
    print("-" * 40)
    for name, factory in SCENARIOS.items():
        config = factory()
        print(f"  {name}: {config.description}")
    print()
