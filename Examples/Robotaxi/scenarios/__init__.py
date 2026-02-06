"""
Paper Scenarios for Robotaxi Cabin Climate Control

Each scenario demonstrates a specific MPC advantage over PID control.

Scenarios:
1. preconditioning: Pre-Conditioning Before Passenger Boarding
2. highway_anticipation: Highway Speed Anticipation
3. peak_shaving: Temperature Peak Shaving
4. co2_management: CO2 vs. Energy Trade-off
5. soc_relaxation: SOC-Dependent Comfort Relaxation
"""

from scenario_config import (
    ScenarioConfig,
    MVConfig,
    get_scenario,
    list_scenarios,
    SCENARIOS,
)
