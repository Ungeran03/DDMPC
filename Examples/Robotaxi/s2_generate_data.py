"""
Step 2: Generate Training Data / Test PID Baseline

This script runs a PID controller to:
1. Test the cabin simulator
2. Collect baseline data for comparison with MPC
3. (Optional) Generate training data for data-driven models
"""

from configuration import *
from controllers import FixedPID, BlowerPI

# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

# Simulation duration
simulation_hours = 4
simulation_duration = simulation_hours * one_hour

# Start time (Unix timestamp - here we use 0 for simplicity)
start_time = 6 * one_hour  # Start at 6:00 AM

# Scenario: 'summer_city', 'winter_highway', 'mild_mixed'
scenario = 'winter_highway'

# =============================================================================
# PID CONTROLLER
# =============================================================================

# PID for cooling: if T_cabin > target, increase u_hvac
# u_recirc stays at default (0.5) during PID operation
if scenario in ('winter_highway', 'winter_city'):
    # Heating: low Kp needed due to low UA (27.5 W/K) and PLR-COP amplification
    pid_controller = FixedPID(
        y=T_cabin,
        u=u_hvac,
        step_size=one_minute,
        Kp=0.04,
        Ti=100.0,
        Td=0.0,
        reverse_act=False,  # Heating: increase u when T < target
    )
else:
    # Cooling (summer, mild)
    pid_controller = FixedPID(
        y=T_cabin,
        u=u_hvac,
        step_size=one_minute,
        Kp=0.3,
        Ti=100.0,
        Td=0.0,
        reverse_act=True,   # Cooling: increase u when T > target
    )

# Blower: proportional to abs(error) + minimum ventilation floor
blower_pi = BlowerPI(
    y=T_cabin,
    u=u_blower,
    step_size=one_minute,
    Kp=1.0,
    tau=300.0,
    min_vent=0.5,
)

# =============================================================================
# RUN SIMULATION
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Robotaxi Cabin - PID Baseline Simulation")
    print("=" * 60)

    # Setup system
    system.setup(start_time=start_time, scenario=scenario)
    system.summary()

    # Run simulation
    print(f"\nRunning {simulation_hours}h simulation with PID controller...")
    print(f"Target temperature: {T_cabin_steady.day_target - 273.15:.1f}°C")
    print()

    data_container = system.run(
        duration=simulation_duration,
        controllers=(pid_controller, blower_pi),
    )

    # Get results
    df = data_container.df

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Simulation Results")
    print("=" * 60)

    T_mean = df['cabin_temperature'].mean() - 273.15
    T_min = df['cabin_temperature'].min() - 273.15
    T_max = df['cabin_temperature'].max() - 273.15
    T_target = T_cabin_steady.day_target - 273.15

    print(f"Cabin Temperature:")
    print(f"  Target: {T_target:.1f}°C")
    print(f"  Mean:   {T_mean:.1f}°C")
    print(f"  Min:    {T_min:.1f}°C")
    print(f"  Max:    {T_max:.1f}°C")

    P_mean = df['hvac_power'].mean()
    P_total = df['hvac_power'].sum() * system.step_size / 3600  # Wh

    print(f"\nHVAC Energy:")
    print(f"  Mean Power: {P_mean:.0f} W")
    print(f"  Total Energy: {P_total:.0f} Wh")

    u_mean = df['hvac_modulation'].mean()
    print(f"\nHVAC Utilization: {u_mean*100:.1f}%")

    # CO2 results
    if 'co2_concentration' in df.columns:
        co2_max = df['co2_concentration'].max()
        co2_mean = df['co2_concentration'].mean()
        print(f"\nCO2:")
        print(f"  Mean: {co2_mean:.0f} ppm")
        print(f"  Max:  {co2_max:.0f} ppm")
        print(f"  Limit: 1200 ppm")

    # Mass temperature
    if 'mass_temperature' in df.columns:
        T_mass_mean = df['mass_temperature'].mean() - 273.15
        print(f"\nMass Temperature: {T_mass_mean:.1f}°C (mean)")

    # Plot results
    print("\nPlotting results...")
    pid_plotter.plot(df)

    # Save data (optional)
    # save_DataHandler(data_container, 'pid_baseline_data')
    # print("Data saved to: pid_baseline_data")
