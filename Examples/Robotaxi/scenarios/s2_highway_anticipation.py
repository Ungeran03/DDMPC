"""
Scenario 2: Highway Speed Anticipation

Demonstrates MPC's ability to use velocity forecast for better radiator efficiency.

Timeline:
- t=0-3min:   City driving, v=5 m/s, eta_radiator=0.65
- t=3-5min:   Acceleration to highway
- t=5-25min:  Highway cruising, v=25 m/s, eta_radiator=0.90
- t=25-30min: Deceleration back to city

Key insight:
- MPC sees v_vehicle in forecast → can delay cooling until eta_radiator improves
- PID ignores velocity → runs compressor immediately at lower efficiency
- eta_radiator(v) = 0.5 + 0.5 * (1 - exp(-v/15))

Metrics:
- Energy during city phase (t=0-3min)
- Energy during highway phase (t=5-25min)
- Total energy consumption
- Temperature tracking quality
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration import *
from scenario_config import highway_anticipation_scenario, ScenarioConfig, MVConfig
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


def save_results(metrics_all: dict, config: ScenarioConfig, dataframes: dict = None, save_path: str = None):
    """
    Save scenario results for reproducibility and paper plots.

    Saves:
    - JSON with metrics and config
    - CSV files with raw time series data (for paper plots)
    """
    base_dir = os.path.join(os.path.dirname(__file__), '..')

    # Save metrics as JSON
    if save_path is None:
        save_path = os.path.join(base_dir, f's2_{config.name}_results.json')

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

    # Save raw data as CSV for paper plots
    if dataframes:
        for name, df in dataframes.items():
            csv_path = os.path.join(base_dir, f's2_{config.name}_{name.lower()}.csv')
            # Select all relevant columns and convert units
            export_df = pd.DataFrame({
                'time_min': np.arange(len(df)),
                # States
                'T_cabin_C': df['cabin_temperature'] - 273.15,
                'T_mass_C': df['mass_temperature'] - 273.15 if 'mass_temperature' in df.columns else np.nan,
                'T_vent_C': df['vent_temperature'] - 273.15 if 'vent_temperature' in df.columns else np.nan,
                'CO2_ppm': df['co2_concentration'] if 'co2_concentration' in df.columns else np.nan,
                # Controls
                'u_hvac': df['hvac_modulation'],
                'u_ptc': df['ptc_modulation'] if 'ptc_modulation' in df.columns else 0.0,
                'u_blower': df['blower_modulation'],
                'u_recirc': df['recirc_modulation'] if 'recirc_modulation' in df.columns else 0.5,
                # Disturbances
                'T_ambient_C': df['ambient_temperature'] - 273.15 if 'ambient_temperature' in df.columns else np.nan,
                'n_passengers': df['passenger_count'] if 'passenger_count' in df.columns else np.nan,
                'v_vehicle_m_s': df['vehicle_speed'] if 'vehicle_speed' in df.columns else np.nan,
                'solar_W_m2': df['solar_irradiance'] if 'solar_irradiance' in df.columns else np.nan,
                # Power
                'P_hvac_W': df['hvac_power'] if 'hvac_power' in df.columns else np.nan,
            })
            export_df.to_csv(csv_path, index=False)
            print(f"Raw data saved to: {csv_path}")

    return save_path


def build_mpc_from_config(config: ScenarioConfig):
    """
    Build MPC controller from scenario configuration.

    Uses simple Quadratic costs for reliable IPOPT convergence.
    """

    # Create WhiteBox predictors
    wb_T_ptc = create_whitebox_T_ptc()
    wb_T_vent = create_whitebox_T_vent(hvac_mode=config.hvac_mode)
    wb_T_cabin = create_whitebox_T_cabin()
    wb_CO2 = create_whitebox_CO2()

    # Build objectives using simple Quadratic costs
    objectives = []

    # Temperature tracking
    if config.weights.get('T_cabin', 0) > 0:
        objectives.append(Objective(
            feature=T_cabin,
            cost=Quadratic(weight=config.weights['T_cabin'])
        ))

    if config.weights.get('T_vent', 0) > 0:
        objectives.append(Objective(feature=T_vent, cost=Quadratic(weight=config.weights['T_vent'])))

    # CO2 tracking
    if config.weights.get('C_CO2', 0) > 0:
        objectives.append(Objective(
            feature=C_CO2,
            cost=Quadratic(weight=config.weights['C_CO2'], norm=100.0)
        ))

    # Energy penalty
    if config.weights.get('energy', 0) > 0:
        objectives.append(Objective(
            feature=u_hvac,
            cost=Quadratic(weight=config.weights['energy'] * config.Q_hp_max / config.COP_nominal)
        ))

    # Smoothness: Control change penalties
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

    # Control bounds
    if mv_cfg['u_hvac'].active:
        constraints.append(Constraint(feature=u_hvac, lb=mv_cfg['u_hvac'].lb, ub=mv_cfg['u_hvac'].ub))
    if mv_cfg.get('u_ptc', MVConfig(active=False)).active:
        constraints.append(Constraint(feature=u_ptc, lb=mv_cfg['u_ptc'].lb, ub=mv_cfg['u_ptc'].ub))
    if mv_cfg['u_blower'].active:
        constraints.append(Constraint(feature=u_blower, lb=mv_cfg['u_blower'].lb, ub=mv_cfg['u_blower'].ub))
    if mv_cfg['u_recirc'].active:
        constraints.append(Constraint(feature=u_recirc, lb=mv_cfg['u_recirc'].lb, ub=mv_cfg['u_recirc'].ub))

    # CO2 hard constraint
    constraints.append(Constraint(feature=C_CO2, lb=0, ub=config.CO2_limit))

    # Create MPC controller
    mpc = ModelPredictive(
        step_size=one_minute,
        nlp=NLP(
            model=model,
            N=20,
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

    # Solver options
    solver_options = {
        "verbose": False,
        "ipopt.print_level": 0,
        "ipopt.max_iter": 1000,
        "expand": True,
    }

    # Select predictors
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


def compute_eta_radiator(v: float) -> float:
    """Compute radiator efficiency from velocity."""
    return 0.5 + 0.5 * (1 - np.exp(-v / 15))


def compute_metrics(df: pd.DataFrame, config: ScenarioConfig) -> dict:
    """Compute scenario-specific metrics for highway anticipation."""

    metrics = {}

    # Phase boundaries (in minutes)
    city_end = 3      # End of initial city phase
    accel_end = 5     # End of acceleration phase
    highway_end = 25  # End of highway phase

    # Energy calculations (assuming 1-minute steps)
    if 'hvac_power' in df.columns:
        # Energy during city phase (t=0-3min)
        city_mask = df.index < city_end
        metrics['energy_city_Wh'] = df.loc[city_mask, 'hvac_power'].sum() * 60 / 3600

        # Energy during acceleration (t=3-5min)
        accel_mask = (df.index >= city_end) & (df.index < accel_end)
        metrics['energy_accel_Wh'] = df.loc[accel_mask, 'hvac_power'].sum() * 60 / 3600

        # Energy during highway phase (t=5-25min)
        highway_mask = (df.index >= accel_end) & (df.index < highway_end)
        metrics['energy_highway_Wh'] = df.loc[highway_mask, 'hvac_power'].sum() * 60 / 3600

        # Total energy
        metrics['energy_total_Wh'] = df['hvac_power'].sum() * 60 / 3600

    # Temperature metrics
    T_target = 295.15  # 22C
    T_cabin_C = df['cabin_temperature'] - 273.15

    metrics['T_mean_C'] = T_cabin_C.mean()
    metrics['T_max_C'] = T_cabin_C.max()
    metrics['T_min_C'] = T_cabin_C.min()
    metrics['T_at_highway_start_C'] = T_cabin_C.iloc[accel_end] if accel_end < len(df) else np.nan

    # Temperature deviation from target
    T_deviation = (df['cabin_temperature'] - T_target).abs()
    metrics['T_mean_deviation_K'] = T_deviation.mean()
    metrics['T_max_deviation_K'] = T_deviation.max()

    # Mean eta_radiator (if velocity available)
    if 'vehicle_speed' in df.columns:
        v = df['vehicle_speed']
        eta = 0.5 + 0.5 * (1 - np.exp(-v / 15))
        metrics['eta_radiator_city'] = eta.iloc[:city_end].mean() if city_end <= len(df) else np.nan
        metrics['eta_radiator_highway'] = eta.iloc[accel_end:highway_end].mean() if highway_end <= len(df) else np.nan

    # CO2 metrics
    if 'co2_concentration' in df.columns:
        metrics['CO2_max'] = df['co2_concentration'].max()
        metrics['CO2_mean'] = df['co2_concentration'].mean()

    return metrics


def plot_comparison(results: dict, config: ScenarioConfig, save_path: str = None):
    """Plot MPC vs PID comparison for highway anticipation scenario."""

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    # Phase markers
    city_end = 3
    accel_end = 5
    highway_end = 25

    colors = {'MPC': 'blue', 'PID': 'red'}

    for name, df in results.items():
        t_min = np.arange(len(df))
        color = colors[name]

        # Temperature
        axes[0].plot(t_min, df['cabin_temperature'] - 273.15, color=color, label=name)

        # u_hvac
        axes[1].plot(t_min, df['hvac_modulation'], color=color, label=name, drawstyle='steps-post')

        # Vehicle speed
        if 'vehicle_speed' in df.columns:
            axes[2].plot(t_min, df['vehicle_speed'], color=color, label=name)

        # eta_radiator (calculated from velocity)
        if 'vehicle_speed' in df.columns:
            eta = 0.5 + 0.5 * (1 - np.exp(-df['vehicle_speed'] / 15))
            axes[3].plot(t_min, eta, color=color, label=name)

        # HVAC power
        if 'hvac_power' in df.columns:
            axes[4].plot(t_min, df['hvac_power'], color=color, label=name)

    # Add target and phase markers
    axes[0].axhline(y=22, color='green', linestyle=':', label='Target')
    for ax in axes:
        ax.axvline(x=city_end, color='grey', linestyle='--', alpha=0.5)
        ax.axvline(x=accel_end, color='grey', linestyle='--', alpha=0.5)
        ax.axvline(x=highway_end, color='grey', linestyle='--', alpha=0.5)

    # Add phase labels
    axes[0].text(1.5, axes[0].get_ylim()[1], 'City', ha='center', va='bottom', fontsize=9, color='grey')
    axes[0].text(15, axes[0].get_ylim()[1], 'Highway', ha='center', va='bottom', fontsize=9, color='grey')

    axes[0].set_ylabel('T_cabin [C]')
    axes[0].legend(loc='upper right')
    axes[0].set_title(f'Scenario 2: {config.description}')

    axes[1].set_ylabel('u_hvac [-]')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc='upper right')

    axes[2].set_ylabel('v_vehicle [m/s]')
    axes[2].legend(loc='upper right')

    axes[3].set_ylabel('eta_radiator [-]')
    axes[3].set_ylim(0.4, 1.0)
    axes[3].axhline(y=0.65, color='orange', linestyle=':', alpha=0.5, label='City (v=5)')
    axes[3].axhline(y=0.90, color='green', linestyle=':', alpha=0.5, label='Highway (v=25)')
    axes[3].legend(loc='lower right')

    axes[4].set_ylabel('P_hvac [W]')
    axes[4].set_xlabel('Time [min]')
    axes[4].legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def run_scenario():
    """Run Scenario 2: Highway Speed Anticipation."""

    print("=" * 70)
    print("SCENARIO 2: Highway Speed Anticipation")
    print("=" * 70)

    # Load scenario configuration
    config = highway_anticipation_scenario()
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
            scenario='summer_city',
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
        print(f"  Energy (city phase t=0-3min):    {metrics['energy_city_Wh']:.1f} Wh")
        print(f"  Energy (accel phase t=3-5min):   {metrics['energy_accel_Wh']:.1f} Wh")
        print(f"  Energy (highway phase t=5-25min):{metrics['energy_highway_Wh']:.1f} Wh")
        print(f"  Energy (total):                  {metrics['energy_total_Wh']:.1f} Wh")
        print(f"  T at highway start (t=5min):     {metrics['T_at_highway_start_C']:.1f}C")
        print(f"  T mean:                          {metrics['T_mean_C']:.1f}C")
        print(f"  T max deviation:                 {metrics['T_max_deviation_K']:.2f} K")
        if 'eta_radiator_city' in metrics:
            print(f"  eta_radiator (city):             {metrics['eta_radiator_city']:.2f}")
            print(f"  eta_radiator (highway):          {metrics['eta_radiator_highway']:.2f}")

    # --- Calculate Improvements ---
    if 'PID' in metrics_all and 'MPC' in metrics_all:
        print("\n" + "-" * 50)
        print("MPC IMPROVEMENTS vs PID:")
        print("-" * 50)

        energy_city_diff = metrics_all['PID']['energy_city_Wh'] - metrics_all['MPC']['energy_city_Wh']
        energy_total_diff = metrics_all['PID']['energy_total_Wh'] - metrics_all['MPC']['energy_total_Wh']
        energy_pct = energy_total_diff / metrics_all['PID']['energy_total_Wh'] * 100 if metrics_all['PID']['energy_total_Wh'] > 0 else 0

        print(f"  Energy (city phase):   {energy_city_diff:+.1f} Wh saved")
        print(f"  Energy (total):        {energy_total_diff:+.1f} Wh ({energy_pct:+.1f}%)")

        T_dev_diff = metrics_all['PID']['T_max_deviation_K'] - metrics_all['MPC']['T_max_deviation_K']
        print(f"  T max deviation:       {T_dev_diff:+.2f} K better")

    # --- Plot Results ---
    print("\n" + "-" * 50)
    print("Generating comparison plot...")
    print("-" * 50)

    plot_path = os.path.join(os.path.dirname(__file__), '..', 's2_highway_anticipation_comparison.png')
    plot_comparison(results, config, save_path=plot_path)

    # Save metrics and raw data for reproducibility / paper plots
    save_results(metrics_all, config, dataframes=results)

    return results, metrics_all


if __name__ == "__main__":
    results, metrics = run_scenario()
