"""
Step 4: Model Predictive Control for Robotaxi Cabin Climate

Enhanced MPC with:
- Two-node thermal model (air + interior mass)
- Fresh air / recirculation control
- CO2 air quality tracking
- PLR-dependent COP

Comparison: MPC vs PID baseline
"""

from configuration import *
from s3_T_cabin_WB import create_whitebox_T_cabin, create_whitebox_T_mass, create_whitebox_CO2
from controllers import FixedPID, BlowerPI
from ddmpc.controller.model_predictive.nlp import NLP, Objective, Constraint
from ddmpc.controller.model_predictive.costs import Quadratic, AbsoluteLinear

# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

simulation_hours = 4
simulation_duration = simulation_hours * one_hour
start_time = 6 * one_hour  # Start at 6:00 AM
scenario = 'summer_city'

# =============================================================================
# CREATE WHITEBOX PREDICTORS
# =============================================================================

wb_T_cabin = create_whitebox_T_cabin(hvac_mode='cooling')
wb_T_mass = create_whitebox_T_mass()
wb_CO2 = create_whitebox_CO2()

# =============================================================================
# SET MODES FOR MPC
# =============================================================================

T_cabin.mode = T_cabin_economic     # Economic mode: comfort bounds
C_CO2.mode = C_CO2_economic         # Economic mode: CO2 bounds

# =============================================================================
# MPC CONTROLLER
# =============================================================================

mpc_controller = ModelPredictive(
    step_size=one_minute,
    nlp=NLP(
        model=model,
        N=20,                               # 20 min prediction horizon
        control_change_step=1,
        objectives=[
            # Comfort: penalize T_cabin bound violations
            Objective(feature=T_cabin, cost=Quadratic(weight=100.0)),
            # Mass: low weight (physics coupling only)
            Objective(feature=T_mass, cost=Quadratic(weight=1.0)),
            # CO2: penalize bound violations
            Objective(feature=C_CO2, cost=Quadratic(weight=50.0)),
            # Control smoothness
            Objective(feature=u_hvac_change, cost=Quadratic(weight=10.0)),
            Objective(feature=u_ptc_change, cost=Quadratic(weight=10.0)),
            Objective(feature=u_blower_change, cost=Quadratic(weight=5.0)),
            Objective(feature=u_recirc_change, cost=Quadratic(weight=5.0)),
        ],
        constraints=[
            # Control bounds
            Constraint(feature=u_hvac, lb=0, ub=1),
            Constraint(feature=u_ptc, lb=0, ub=1),
            Constraint(feature=u_blower, lb=0.1, ub=1),
            Constraint(feature=u_recirc, lb=0, ub=1),
            # Hard CO2 safety limit
            Constraint(feature=C_CO2, lb=0, ub=1200),
        ],
    ),
    forecast_callback=system.get_forecast,
    solution_plotter=mpc_plotter,
    show_solution_plot=True,
    save_solution_plot=False,
    save_solution_data=True,
)

# =============================================================================
# PID BASELINE FOR COMPARISON
# =============================================================================

# PID only controls u_hvac; u_recirc stays at default (0.5)
if scenario in ('winter_highway', 'winter_city'):
    pid_controller = FixedPID(
        y=T_cabin, u=u_hvac, step_size=one_minute,
        Kp=0.04, Ti=100.0, Td=0.0, reverse_act=False,
    )
else:
    pid_controller = FixedPID(
        y=T_cabin, u=u_hvac, step_size=one_minute,
        Kp=0.3, Ti=100.0, Td=0.0, reverse_act=True,
    )

# Blower for PID baseline
blower_pi = BlowerPI(
    y=T_cabin, u=u_blower, step_size=one_minute,
    Kp=1.0, tau=300.0, min_vent=0.5,
)

# =============================================================================
# SOLVER OPTIONS
# =============================================================================

solver_options = {
    "verbose": False,
    "ipopt.print_level": 2,
    "ipopt.max_iter": 1000,
    "expand": True,
    'ipopt.tol': 1e-2,
    'ipopt.acceptable_tol': 1e-1,
}

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
        controllers=(pid_controller, blower_pi),
    )
    results['PID'] = dc_pid.df.copy()

    # --- Run MPC ---
    print("\n" + "-" * 40)
    print("Running MPC...")
    print("-" * 40)

    system.close()
    system.setup(start_time=start_time, scenario=scenario)

    # Build NLP with all three predictors
    mpc_controller.nlp.build(
        solver_options=solver_options,
        predictors=[wb_T_cabin, wb_T_mass, wb_CO2],
    )
    mpc_controller.nlp.summary()

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

    T_target_C = T_cabin_steady.day_target - 273.15

    for name, df in results.items():
        T_mean = df['cabin_temperature'].mean() - 273.15
        T_std = df['cabin_temperature'].std()
        T_max_dev = (df['cabin_temperature'] - T_cabin_steady.day_target).abs().max()
        P_total = df['hvac_power'].sum() * system.step_size / 3600  # Wh
        u_mean = df['hvac_modulation'].mean()
        co2_max = df['co2_concentration'].max() if 'co2_concentration' in df.columns else 0
        co2_mean = df['co2_concentration'].mean() if 'co2_concentration' in df.columns else 0

        print(f"\n{name}:")
        print(f"  Mean Temp:     {T_mean:.2f}°C (target: {T_target_C:.1f}°C)")
        print(f"  Temp Std:      {T_std:.3f} K")
        print(f"  Max Deviation: {T_max_dev:.2f} K")
        print(f"  Total Energy:  {P_total:.0f} Wh")
        print(f"  Avg HVAC:      {u_mean*100:.1f}%")
        print(f"  Max CO2:       {co2_max:.0f} ppm")
        print(f"  Mean CO2:      {co2_mean:.0f} ppm")

    # Calculate improvements
    E_pid = results['PID']['hvac_power'].sum()
    E_mpc = results['MPC']['hvac_power'].sum()
    if E_pid > 0:
        energy_saving = (E_pid - E_mpc) / E_pid * 100
        print(f"\n{'='*40}")
        print(f"Energy Saving (MPC vs PID): {energy_saving:.1f}%")
        print(f"{'='*40}")

    # =============================================================================
    # PLOT RESULTS
    # =============================================================================

    print("\nPlotting MPC results...")
    mpc_plotter.plot(results['MPC'])
