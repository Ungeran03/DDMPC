"""
Scenario 1: Pre-Conditioning Before Passenger Boarding

Demonstrates MPC's ability to use passenger forecast for proactive climate control.

Timeline:
- t=0-5min:   Empty cabin, parked in sun (35°C)
- t=5min:     3 passengers board (MPC knew this from t=0!)
- t=5-25min:  3 passengers in cabin
- t=25min:    Passengers alight
- t=25-30min: Empty again

Key insight:
- MPC sees n_passengers in forecast → pre-cools and pre-ventilates BEFORE boarding
- PID only reacts AFTER temperature/CO2 changes → overshoot and discomfort

Metrics:
- T_cabin at boarding (t=5min)
- CO2 at boarding
- Temperature overshoot after boarding
- Energy consumption
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration import *
from scenario_config import preconditioning_scenario, ScenarioConfig, MVConfig
from s3_T_cabin_WB import (
    create_whitebox_T_ptc,
    create_whitebox_T_vent,
    create_whitebox_T_cabin,
    create_whitebox_CO2,
)
from controllers import FixedPID, BlowerPI, PTCRelay
from ddmpc.controller.model_predictive.nlp import NLP, Objective, Constraint
from ddmpc.controller.model_predictive.costs import Quadratic

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime


def save_results(metrics_all: dict, config: ScenarioConfig, save_path: str = None):
    """Save scenario results as JSON for reproducibility."""
    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(__file__), '..',
            f's1_{config.name}_results.json'
        )

    results = {
        'scenario': config.name,
        'description': config.description,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'duration_hours': config.duration_hours,
            'hvac_mode': config.hvac_mode,
            'T_cabin_init_C': config.T_cabin_init - 273.15,
            'T_mass_init_C': config.T_mass_init - 273.15,
            'weights': config.weights,
        },
        'metrics': metrics_all,
    }

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {save_path}")
    return save_path


def build_mpc_from_config(config: ScenarioConfig):
    """
    Build MPC controller from scenario configuration.

    Uses simple Quadratic costs for reliable IPOPT convergence:
    - Temperature: Quadratic toward target (22°C)
    - CO2: Quadratic toward target (600 ppm ambient level)
    - Energy: Penalize u_hvac directly
    - Smoothness: Control change penalties
    """

    # Create WhiteBox predictors
    wb_T_ptc = create_whitebox_T_ptc()
    wb_T_vent = create_whitebox_T_vent(hvac_mode=config.hvac_mode)
    wb_T_cabin = create_whitebox_T_cabin()
    wb_CO2 = create_whitebox_CO2()

    # Build objectives using simple Quadratic costs
    objectives = []

    # --- Temperature: Quadratic toward target (22°C = 295.15 K) ---
    if config.weights.get('T_cabin', 0) > 0:
        objectives.append(Objective(
            feature=T_cabin,
            cost=Quadratic(weight=config.weights['T_cabin'])
        ))

    # T_vent: Intermediate state tracking
    if config.weights.get('T_vent', 0) > 0:
        objectives.append(Objective(feature=T_vent, cost=Quadratic(weight=config.weights['T_vent'])))

    # T_ptc: PTC element tracking
    if config.weights.get('T_ptc', 0) > 0:
        objectives.append(Objective(feature=T_ptc_elem, cost=Quadratic(weight=config.weights['T_ptc'])))

    # --- CO2: Quadratic toward low target (600 ppm) ---
    # Penalizes high CO2; hard constraint at 1200 ppm enforces the Pettenkofer limit
    if config.weights.get('C_CO2', 0) > 0:
        objectives.append(Objective(
            feature=C_CO2,
            cost=Quadratic(
                weight=config.weights['C_CO2'],
                norm=100.0,  # Scale: 100 ppm deviation = 1 unit
            )
        ))

    # --- Energy: Penalize u_hvac ---
    if config.weights.get('energy', 0) > 0:
        objectives.append(Objective(
            feature=u_hvac,
            cost=Quadratic(weight=config.weights['energy'] * config.Q_hp_max / config.COP_nominal)
        ))

    # --- Smoothness: Control change penalties ---
    mv_cfg = config.mv_config
    if mv_cfg['u_hvac'].active:
        objectives.append(Objective(feature=u_hvac_change, cost=Quadratic(weight=mv_cfg['u_hvac'].weight_change)))
    if mv_cfg.get('u_ptc', MVConfig(active=False)).active:
        objectives.append(Objective(feature=u_ptc_change, cost=Quadratic(weight=mv_cfg['u_ptc'].weight_change)))
    if mv_cfg['u_blower'].active:
        objectives.append(Objective(feature=u_blower_change, cost=Quadratic(weight=mv_cfg['u_blower'].weight_change)))
    if mv_cfg['u_recirc'].active:
        objectives.append(Objective(feature=u_recirc_change, cost=Quadratic(weight=mv_cfg['u_recirc'].weight_change)))

    # Build constraints from config
    constraints = []

    # Control bounds (box constraints - these MUST be respected)
    if mv_cfg['u_hvac'].active:
        constraints.append(Constraint(feature=u_hvac, lb=mv_cfg['u_hvac'].lb, ub=mv_cfg['u_hvac'].ub))
    if mv_cfg.get('u_ptc', MVConfig(active=False)).active:
        constraints.append(Constraint(feature=u_ptc, lb=mv_cfg['u_ptc'].lb, ub=mv_cfg['u_ptc'].ub))
    if mv_cfg['u_blower'].active:
        constraints.append(Constraint(feature=u_blower, lb=mv_cfg['u_blower'].lb, ub=mv_cfg['u_blower'].ub))
    if mv_cfg['u_recirc'].active:
        constraints.append(Constraint(feature=u_recirc, lb=mv_cfg['u_recirc'].lb, ub=mv_cfg['u_recirc'].ub))

    # CO2 hard constraint (Pettenkofer limit)
    constraints.append(Constraint(feature=C_CO2, lb=0, ub=config.CO2_limit))

    # Create MPC controller
    mpc = ModelPredictive(
        step_size=one_minute,
        nlp=NLP(
            model=model,
            N=20,  # 20-minute horizon
            control_change_step=1,
            objectives=objectives,
            constraints=constraints,
        ),
        forecast_callback=system.get_forecast,
        solution_plotter=mpc_plotter,
        show_solution_plot=False,
        save_solution_plot=False,
        save_solution_data=False,
    )

    # Solver options with higher iteration limit for reliable convergence
    solver_options = {
        "verbose": False,
        "ipopt.print_level": 0,
        "ipopt.max_iter": 1000,  # Higher limit for difficult initial conditions
        "expand": True,
    }

    # Select predictors based on active MVs
    predictors = [wb_T_vent, wb_T_cabin, wb_CO2]
    if mv_cfg.get('u_ptc', MVConfig(active=False)).active:
        predictors.insert(0, wb_T_ptc)

    mpc.nlp.build(solver_options=solver_options, predictors=predictors)

    return mpc


def build_pid_from_config(config: ScenarioConfig):
    """Build PID controllers from scenario configuration."""

    pid_params = config.pid_params or {}

    # Main temperature PID
    pid = FixedPID(
        y=T_cabin, u=u_hvac, step_size=one_minute,
        Kp=pid_params.get('Kp', 0.3),
        Ti=pid_params.get('Ti', 100.0),
        Td=pid_params.get('Td', 0.0),
        reverse_act=pid_params.get('reverse_act', True),
    )

    # Blower PI
    blower = BlowerPI(
        y=T_cabin, u=u_blower, T_amb_feature=T_ambient, step_size=one_minute,
        Kp=0.5, Ti=150.0, deadband=0.3,
    )

    # PTC relay (only for heating scenarios)
    ptc = PTCRelay(
        y=T_cabin, u=u_ptc, T_amb_feature=T_ambient,
        step_size=one_minute,
    )

    return pid, blower, ptc


def compute_metrics(df: pd.DataFrame, config: ScenarioConfig, boarding_time_sec: int = 8 * 60) -> dict:
    """Compute scenario-specific metrics."""

    metrics = {}

    # Find boarding index
    boarding_idx = int(boarding_time_sec / 60)  # Assuming 1-minute steps

    # T_cabin at boarding
    if boarding_idx < len(df):
        metrics['T_at_boarding_C'] = df['cabin_temperature'].iloc[boarding_idx] - 273.15
    else:
        metrics['T_at_boarding_C'] = np.nan

    # CO2 at boarding
    if 'co2_concentration' in df.columns and boarding_idx < len(df):
        metrics['CO2_at_boarding'] = df['co2_concentration'].iloc[boarding_idx]
    else:
        metrics['CO2_at_boarding'] = np.nan

    # Temperature overshoot after boarding (for cooling: undershoot below target)
    T_target = 295.15  # 22°C
    if boarding_idx < len(df):
        T_after_boarding = df['cabin_temperature'].iloc[boarding_idx:]
        if config.hvac_mode == 'cooling':
            # Overshoot = how much above target after cooldown
            metrics['T_overshoot_K'] = (T_after_boarding - T_target).max()
        else:
            # Undershoot = how much below target
            metrics['T_overshoot_K'] = (T_target - T_after_boarding).max()
    else:
        metrics['T_overshoot_K'] = np.nan

    # Mean temperature
    metrics['T_mean_C'] = df['cabin_temperature'].mean() - 273.15

    # Max CO2
    if 'co2_concentration' in df.columns:
        metrics['CO2_max'] = df['co2_concentration'].max()
        metrics['CO2_mean'] = df['co2_concentration'].mean()
    else:
        metrics['CO2_max'] = np.nan
        metrics['CO2_mean'] = np.nan

    # Total energy
    metrics['energy_total_Wh'] = df['hvac_power'].sum() * 60 / 3600  # Assuming 1-min steps

    # Energy before boarding (pre-conditioning)
    if boarding_idx < len(df):
        metrics['energy_preconditioning_Wh'] = df['hvac_power'].iloc[:boarding_idx].sum() * 60 / 3600
    else:
        metrics['energy_preconditioning_Wh'] = np.nan

    return metrics


def plot_comparison(results: dict, config: ScenarioConfig, save_path: str = None):
    """Plot MPC vs PID comparison for this scenario."""

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    boarding_time = 8  # minutes
    alighting_time = 25  # minutes

    colors = {'MPC': 'blue', 'PID': 'red'}

    for name, df in results.items():
        t_min = np.arange(len(df))  # Time in minutes
        color = colors[name]

        # Temperature (only T_cabin)
        axes[0].plot(t_min, df['cabin_temperature'] - 273.15, color=color, label=name)

        # CO2
        if 'co2_concentration' in df.columns:
            axes[1].plot(t_min, df['co2_concentration'], color=color, label=name)

        # u_hvac
        axes[2].plot(t_min, df['hvac_modulation'], color=color, label=name, drawstyle='steps-post')

        # u_blower
        axes[3].plot(t_min, df['blower_modulation'], color=color, label=name, drawstyle='steps-post')

        # u_recirc (only MPC controls this)
        if 'recirc_modulation' in df.columns:
            axes[4].plot(t_min, df['recirc_modulation'], color=color, label=name, drawstyle='steps-post')

    # Add target lines and boarding markers
    axes[0].axhline(y=22, color='green', linestyle=':', label='Target')
    axes[0].axvline(x=boarding_time, color='grey', linestyle='--', alpha=0.5, label='Boarding')
    axes[0].axvline(x=alighting_time, color='grey', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Temperature [°C]')
    axes[0].legend(loc='upper right')
    axes[0].set_title(f'Scenario 1: {config.description}')

    axes[1].axhline(y=1200, color='red', linestyle=':', label='Limit')
    axes[1].axvline(x=boarding_time, color='grey', linestyle='--', alpha=0.5)
    axes[1].axvline(x=alighting_time, color='grey', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('CO2 [ppm]')
    axes[1].legend(loc='upper right')

    for ax in axes[2:5]:
        ax.axvline(x=boarding_time, color='grey', linestyle='--', alpha=0.5)
        ax.axvline(x=alighting_time, color='grey', linestyle='--', alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper right')

    axes[2].set_ylabel('u_hvac [-]')
    axes[3].set_ylabel('u_blower [-]')
    axes[4].set_ylabel('u_recirc [-]')
    axes[4].set_xlabel('Time [min]')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def run_scenario():
    """Run Scenario 1: Pre-Conditioning Before Boarding."""

    print("=" * 70)
    print("SCENARIO 1: Pre-Conditioning Before Passenger Boarding")
    print("=" * 70)

    # Load scenario configuration
    config = preconditioning_scenario()
    config.summary()

    # Calculate simulation parameters
    duration_sec = int(config.duration_hours * 3600)
    start_time = int(config.start_time_hours * 3600)

    # Get profile overrides and initial conditions
    profile_overrides = config.get_profile_overrides()
    init_overrides = {
        'T_cabin': config.T_cabin_init,
        'T_mass': config.T_mass_init,
        'T_vent': config.T_vent_init,
        'T_ptc': config.T_ptc_init,
        'C_CO2': config.C_CO2_init,
    }

    results = {}

    # --- Run PID Baseline ---
    if config.run_pid_baseline:
        print("\n" + "-" * 50)
        print("Running PID Baseline...")
        print("-" * 50)

        system.setup(
            start_time=start_time,
            scenario='summer_city',  # Base scenario for defaults
            hvac_mode=config.hvac_mode,
            duration=duration_sec,
            profile_overrides=profile_overrides,
            init_overrides=init_overrides,
        )

        pid, blower, ptc = build_pid_from_config(config)

        dc_pid = system.run(
            duration=duration_sec,
            controllers=(pid, blower, ptc),
        )
        results['PID'] = dc_pid.df.copy()

        system.close()

    # --- Run MPC ---
    print("\n" + "-" * 50)
    print("Running MPC...")
    print("-" * 50)

    system.setup(
        start_time=start_time,
        scenario='summer_city',
        hvac_mode=config.hvac_mode,
        duration=duration_sec,
        profile_overrides=profile_overrides,
        init_overrides=init_overrides,
    )

    mpc = build_mpc_from_config(config)
    mpc.nlp.summary()

    dc_mpc = system.run(
        duration=duration_sec,
        controllers=(mpc,),
    )
    results['MPC'] = dc_mpc.df.copy()

    # --- Compute and Display Metrics ---
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    metrics_all = {}
    for name, df in results.items():
        metrics = compute_metrics(df, config)
        metrics_all[name] = metrics

        print(f"\n{name}:")
        print(f"  T at boarding (t=8min): {metrics['T_at_boarding_C']:.1f}°C (target: 22°C)")
        print(f"  CO2 at boarding:        {metrics['CO2_at_boarding']:.0f} ppm")
        print(f"  T overshoot:            {metrics['T_overshoot_K']:.2f} K")
        print(f"  T mean:                 {metrics['T_mean_C']:.1f}°C")
        print(f"  CO2 max:                {metrics['CO2_max']:.0f} ppm")
        print(f"  Energy total:           {metrics['energy_total_Wh']:.0f} Wh")
        print(f"  Energy pre-conditioning:{metrics['energy_preconditioning_Wh']:.0f} Wh")

    # --- Calculate Improvements ---
    if 'PID' in metrics_all and 'MPC' in metrics_all:
        print("\n" + "-" * 50)
        print("MPC IMPROVEMENTS vs PID:")
        print("-" * 50)

        T_diff = metrics_all['PID']['T_at_boarding_C'] - metrics_all['MPC']['T_at_boarding_C']
        CO2_diff = metrics_all['PID']['CO2_at_boarding'] - metrics_all['MPC']['CO2_at_boarding']
        energy_diff = (metrics_all['PID']['energy_total_Wh'] - metrics_all['MPC']['energy_total_Wh'])
        energy_pct = energy_diff / metrics_all['PID']['energy_total_Wh'] * 100 if metrics_all['PID']['energy_total_Wh'] > 0 else 0

        print(f"  T at boarding:  {T_diff:+.1f}°C closer to target")
        print(f"  CO2 at boarding: {CO2_diff:+.0f} ppm lower")
        print(f"  Energy:          {energy_diff:+.0f} Wh ({energy_pct:+.1f}%)")

    # --- Plot Results ---
    print("\n" + "-" * 50)
    print("Generating comparison plot...")
    print("-" * 50)

    plot_path = os.path.join(os.path.dirname(__file__), '..', 's1_preconditioning_comparison.png')
    plot_comparison(results, config, save_path=plot_path)

    # Save metrics for reproducibility
    save_results(metrics_all, config)

    return results, metrics_all


if __name__ == "__main__":
    results, metrics = run_scenario()
