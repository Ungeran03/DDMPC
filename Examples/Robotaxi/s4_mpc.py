"""
Step 4: Model Predictive Control for Robotaxi Cabin Climate

This script implements MPC for optimal cabin climate control considering:
- Thermal comfort constraints
- Energy minimization
- Predictive optimization using forecasts

Comparison: MPC vs PID baseline
"""

from configuration import *
from s3_T_cabin_WB import create_whitebox_predictor

# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

# Simulation duration
simulation_hours = 4
simulation_duration = simulation_hours * one_hour

# Start time
start_time = 6 * one_hour  # Start at 6:00 AM

# Scenario
scenario = 'summer_city'

# =============================================================================
# MPC CONFIGURATION
# =============================================================================

# Horizons
prediction_horizon = 20     # 20 steps = 20 minutes ahead
control_horizon = 10        # 10 control moves

# =============================================================================
# CREATE WHITEBOX PREDICTOR
# =============================================================================

# Create predictor for cooling mode
wb_T_cabin = create_whitebox_predictor(hvac_mode='cooling')

# =============================================================================
# COST FUNCTIONS
# =============================================================================

# Comfort cost: penalize deviation from target temperature
cost_comfort = Quadratic(
    feature=T_cabin,
    weight=100.0,           # High weight for comfort
)

# Energy cost: penalize HVAC power consumption
cost_energy = Linear(
    feature=P_hvac,
    weight=0.001,           # Smaller weight for energy
)

# Smoothness cost: penalize rapid control changes
cost_smoothness = Quadratic(
    feature=u_hvac_change,
    weight=10.0,
)

# =============================================================================
# BUILD MPC CONTROLLER
# =============================================================================

mpc_controller = ModelPredictive(
    predictors=[wb_T_cabin],
    costs=[cost_comfort, cost_energy, cost_smoothness],
    prediction_horizon=prediction_horizon,
    control_horizon=control_horizon,
    step_size=one_minute,
)

# =============================================================================
# PID BASELINE FOR COMPARISON
# =============================================================================

pid_controller = PID(
    y=T_cabin,
    u=u_hvac,
    step_size=one_minute,
    Kp=0.5,
    Ti=100.0,
    Td=0.0,
    reverse_act=True,
)

# =============================================================================
# RUN SIMULATION
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Robotaxi Cabin - MPC vs PID Comparison")
    print("=" * 60)

    results = {}

    # --- Run PID Baseline ---
    print("\n" + "-" * 40)
    print("Running PID Baseline...")
    print("-" * 40)

    system.setup(start_time=start_time, scenario=scenario)

    dc_pid = system.run(
        duration=simulation_duration,
        controllers=(pid_controller,),
    )
    results['PID'] = dc_pid.df.copy()

    # --- Run MPC ---
    print("\n" + "-" * 40)
    print("Running MPC...")
    print("-" * 40)

    system.close()  # Reset system
    system.setup(start_time=start_time, scenario=scenario)

    dc_mpc = system.run(
        duration=simulation_duration,
        controllers=(mpc_controller,),
    )
    results['MPC'] = dc_mpc.df.copy()

    # =============================================================================
    # COMPARE RESULTS
    # =============================================================================

    print("\n" + "=" * 60)
    print("COMPARISON: MPC vs PID")
    print("=" * 60)

    T_target = T_cabin_steady.day_target - 273.15

    for name, df in results.items():
        T_mean = df['cabin_temperature'].mean() - 273.15
        T_std = df['cabin_temperature'].std()
        T_max_dev = (df['cabin_temperature'] - T_cabin_steady.day_target).abs().max()
        P_total = df['hvac_power'].sum() * system.step_size / 3600  # Wh
        u_mean = df['hvac_modulation'].mean()

        print(f"\n{name}:")
        print(f"  Mean Temp:     {T_mean:.2f}°C (target: {T_target:.1f}°C)")
        print(f"  Temp Std:      {T_std:.3f} K")
        print(f"  Max Deviation: {T_max_dev:.2f} K")
        print(f"  Total Energy:  {P_total:.0f} Wh")
        print(f"  Avg HVAC:      {u_mean*100:.1f}%")

    # Calculate improvements
    E_pid = results['PID']['hvac_power'].sum()
    E_mpc = results['MPC']['hvac_power'].sum()
    energy_saving = (E_pid - E_mpc) / E_pid * 100

    print(f"\n{'='*40}")
    print(f"Energy Saving (MPC vs PID): {energy_saving:.1f}%")
    print(f"{'='*40}")

    # =============================================================================
    # PLOT RESULTS
    # =============================================================================

    print("\nPlotting MPC results...")
    mpc_plotter.plot(results['MPC'])

    # Optional: Plot comparison
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    # ...
