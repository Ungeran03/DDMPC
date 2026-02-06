"""
Scenario 3: SOC-Dependent Comfort Relaxation

Demonstrates MPC's ability to anticipate battery depletion and save energy
BEFORE the critical threshold is reached.

Timeline:
- t=0min:    SOC=20%  MPC sees forecast, starts planning
- t=40min:   SOC=13%  MPC begins saving (sees 10% threshold in 20-min horizon)
- t=60min:   SOC=10%  Threshold reached
- t=90min:   SOC=5%   Deep energy saving mode
- t=120min:  SOC=0%   End of scenario

Key insight:
- MPC sees SOC forecast over 20-min horizon -> acts BEFORE threshold is reached
- PID has no SOC awareness -> keeps cooling at full blast regardless of battery
- Result: MPC preserves range by relaxing comfort early; PID drains battery

SOC-dependent comfort weight:
    w_comfort = w_base        if soc >= 0.1  (full comfort)
    w_comfort = 0.2 * w_base  if soc < 0.1   (save energy)

Metrics:
- Energy consumption (total and per phase)
- Comfort violations (deviation from target)
- Temperature at end of scenario
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration import *
from scenario_config import soc_relaxation_scenario, ScenarioConfig, MVConfig
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
    """
    base_dir = os.path.join(os.path.dirname(__file__), '..')

    if save_path is None:
        save_path = os.path.join(base_dir, f's3_{config.name}_results.json')

    results = {
        'scenario': config.name,
        'description': config.description,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'duration_hours': config.duration_hours,
            'hvac_mode': config.hvac_mode,
            'T_cabin_init_C': config.T_cabin_init - 273.15,
            'T_mass_init_C': config.T_mass_init - 273.15,
            'soc_threshold': config.soc_threshold,
            'soc_low_weight_factor': config.soc_low_weight_factor,
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
            csv_path = os.path.join(base_dir, f's3_{config.name}_{name.lower()}.csv')
            export_df = pd.DataFrame({
                'time_min': np.arange(len(df)),
                'T_cabin_C': df['cabin_temperature'] - 273.15,
                'T_mass_C': df['mass_temperature'] - 273.15 if 'mass_temperature' in df.columns else np.nan,
                'T_vent_C': df['vent_temperature'] - 273.15 if 'vent_temperature' in df.columns else np.nan,
                'CO2_ppm': df['co2_concentration'] if 'co2_concentration' in df.columns else np.nan,
                'u_hvac': df['hvac_modulation'],
                'u_ptc': df['ptc_modulation'] if 'ptc_modulation' in df.columns else 0.0,
                'u_blower': df['blower_modulation'],
                'u_recirc': df['recirc_modulation'] if 'recirc_modulation' in df.columns else 0.5,
                'T_ambient_C': df['ambient_temperature'] - 273.15 if 'ambient_temperature' in df.columns else np.nan,
                'n_passengers': df['passenger_count'] if 'passenger_count' in df.columns else np.nan,
                'v_vehicle_m_s': df['vehicle_speed'] if 'vehicle_speed' in df.columns else np.nan,
                'solar_W_m2': df['solar_irradiance'] if 'solar_irradiance' in df.columns else np.nan,
                'P_hvac_W': df['hvac_power'] if 'hvac_power' in df.columns else np.nan,
            })
            # Add SOC if available
            if 'soc' in df.columns:
                export_df['soc'] = df['soc']
            export_df.to_csv(csv_path, index=False)
            print(f"Raw data saved to: {csv_path}")

    return save_path


def build_mpc_with_soc_weight(config: ScenarioConfig, T_cabin_weight: float):
    """
    Build MPC controller with specified T_cabin weight.

    The weight is determined by SOC:
    - If min_soc_in_horizon >= threshold: full comfort weight
    - If min_soc_in_horizon < threshold: reduced comfort weight
    """
    # Create WhiteBox predictors
    wb_T_ptc = create_whitebox_T_ptc()
    wb_T_vent = create_whitebox_T_vent(hvac_mode=config.hvac_mode)
    wb_T_cabin = create_whitebox_T_cabin()
    wb_CO2 = create_whitebox_CO2()

    # Build objectives with specified T_cabin weight
    objectives = []

    # Temperature with SOC-dependent weight
    if T_cabin_weight > 0:
        objectives.append(Objective(
            feature=T_cabin,
            cost=Quadratic(weight=T_cabin_weight)
        ))

    # T_vent tracking
    if config.weights.get('T_vent', 0) > 0:
        objectives.append(Objective(feature=T_vent, cost=Quadratic(weight=config.weights['T_vent'])))

    # CO2 penalty
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

    # Smoothness penalties
    mv_cfg = config.mv_config
    if mv_cfg['u_hvac'].active:
        objectives.append(Objective(feature=u_hvac_change, cost=Quadratic(weight=mv_cfg['u_hvac'].weight_change)))
    if mv_cfg['u_blower'].active:
        objectives.append(Objective(feature=u_blower_change, cost=Quadratic(weight=mv_cfg['u_blower'].weight_change)))
    if mv_cfg['u_recirc'].active:
        objectives.append(Objective(feature=u_recirc_change, cost=Quadratic(weight=mv_cfg['u_recirc'].weight_change)))

    # Build constraints
    constraints = []
    if mv_cfg['u_hvac'].active:
        constraints.append(Constraint(feature=u_hvac, lb=mv_cfg['u_hvac'].lb, ub=mv_cfg['u_hvac'].ub))
    if mv_cfg['u_blower'].active:
        constraints.append(Constraint(feature=u_blower, lb=mv_cfg['u_blower'].lb, ub=mv_cfg['u_blower'].ub))
    if mv_cfg['u_recirc'].active:
        constraints.append(Constraint(feature=u_recirc, lb=mv_cfg['u_recirc'].lb, ub=mv_cfg['u_recirc'].ub))
    constraints.append(Constraint(feature=C_CO2, lb=0, ub=config.CO2_limit))

    # Create MPC
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

    solver_options = {
        "verbose": False,
        "ipopt.print_level": 0,
        "ipopt.max_iter": 1000,
        "expand": True,
    }

    predictors = [wb_T_vent, wb_T_cabin, wb_CO2]
    mpc.nlp.build(solver_options=solver_options, predictors=predictors)

    return mpc


class SOCAwareMPCWrapper:
    """
    Wrapper that checks SOC forecast and switches between high/low comfort MPCs.

    This implements anticipation: if SOC will drop below threshold within the
    20-minute horizon, switch to low comfort mode BEFORE it happens.
    """

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.threshold = config.soc_threshold
        self.w_base = config.weights['T_cabin']
        self.w_low = self.w_base * config.soc_low_weight_factor

        # Build both MPCs
        print(f"  Building MPC with high comfort weight ({self.w_base})...")
        self.mpc_high = build_mpc_with_soc_weight(config, self.w_base)
        print(f"  Building MPC with low comfort weight ({self.w_low})...")
        self.mpc_low = build_mpc_with_soc_weight(config, self.w_low)

        self.current_mode = 'high'  # Track which MPC is active
        self.mode_switches = []  # Track when mode switches happen

        # Required attributes for DDMPC system
        self.step_size = one_minute

    def __call__(self, df: pd.DataFrame):
        """
        Called at each MPC step. Checks SOC forecast and selects appropriate MPC.
        """
        # Get current time and SOC forecast
        current_time_sec = len(df) * 60  # Assuming 1-minute steps

        # Get SOC profile from config
        soc_profile = self.config.profile_soc

        # Check minimum SOC over 20-minute horizon
        horizon_socs = []
        for k in range(21):  # 0 to 20 minutes ahead
            future_time = current_time_sec + k * 60
            horizon_socs.append(soc_profile(future_time))

        min_soc_in_horizon = min(horizon_socs)
        current_soc = soc_profile(current_time_sec)

        # Select MPC based on forecast
        old_mode = self.current_mode
        if min_soc_in_horizon < self.threshold:
            self.current_mode = 'low'
            controls, extra = self.mpc_low(df)
        else:
            self.current_mode = 'high'
            controls, extra = self.mpc_high(df)

        # Log mode switch
        if self.current_mode != old_mode:
            t_min = current_time_sec / 60
            self.mode_switches.append((t_min, current_soc, min_soc_in_horizon))
            print(f"  [t={t_min:.0f}min] Mode switch: {old_mode} -> {self.current_mode} "
                  f"(current SOC={current_soc*100:.1f}%, min in horizon={min_soc_in_horizon*100:.1f}%)")

        return controls, extra


def build_pid_from_config(config: ScenarioConfig):
    """Build PID controllers from scenario configuration."""

    pid_params = config.pid_params or {}

    pid = FixedPID(
        y=T_cabin, u=u_hvac, step_size=one_minute,
        Kp=pid_params.get('Kp', 0.3),
        Ti=pid_params.get('Ti', 100.0),
        Td=pid_params.get('Td', 0.0),
        reverse_act=pid_params.get('reverse_act', True),
    )

    blower = BlowerPI(
        y=T_cabin, u=u_blower, T_amb_feature=T_ambient, step_size=one_minute,
        Kp=0.5, Ti=150.0, deadband=0.3,
    )

    ptc = PTCRelay(
        y=T_cabin, u=u_ptc, T_amb_feature=T_ambient,
        step_size=one_minute,
    )

    return pid, blower, ptc


def compute_metrics(df: pd.DataFrame, config: ScenarioConfig) -> dict:
    """Compute scenario-specific metrics for SOC relaxation."""

    metrics = {}

    T_target = 295.15  # 22°C
    threshold_time_min = 60  # SOC reaches 10% at t=60min

    # Temperature statistics
    T_cabin_C = df['cabin_temperature'] - 273.15
    metrics['T_mean_C'] = T_cabin_C.mean()
    metrics['T_max_C'] = T_cabin_C.max()
    metrics['T_min_C'] = T_cabin_C.min()
    metrics['T_final_C'] = T_cabin_C.iloc[-1]

    # Max deviation from target (22°C)
    metrics['T_max_dev_K'] = abs(T_cabin_C - 22.0).max()

    # Comfort violation integral: sum of |T - 22| * dt when |T - 22| > 2
    comfort_band = 2.0  # ±2°C comfort band
    deviations = abs(T_cabin_C - 22.0) - comfort_band
    deviations = deviations.clip(lower=0)  # Only positive violations
    metrics['comfort_violation_Kmin'] = deviations.sum()  # K*min (1-min steps)

    # Energy consumption
    if 'hvac_power' in df.columns:
        P_hvac = df['hvac_power']
        metrics['energy_total_Wh'] = P_hvac.sum() * 60 / 3600  # Assuming 1-min steps

        # Energy before threshold (t < 60min)
        metrics['energy_before_threshold_Wh'] = P_hvac.iloc[:threshold_time_min].sum() * 60 / 3600

        # Energy after threshold (t >= 60min)
        metrics['energy_after_threshold_Wh'] = P_hvac.iloc[threshold_time_min:].sum() * 60 / 3600
    else:
        metrics['energy_total_Wh'] = np.nan
        metrics['energy_before_threshold_Wh'] = np.nan
        metrics['energy_after_threshold_Wh'] = np.nan

    # Average u_hvac
    metrics['u_hvac_mean'] = df['hvac_modulation'].mean()
    metrics['u_hvac_mean_before'] = df['hvac_modulation'].iloc[:threshold_time_min].mean()
    metrics['u_hvac_mean_after'] = df['hvac_modulation'].iloc[threshold_time_min:].mean()

    return metrics


def plot_comparison(results: dict, config: ScenarioConfig, soc_profile: callable, save_path: str = None):
    """Plot MPC vs PID comparison for SOC relaxation scenario."""

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    # Threshold time (SOC = 10% at t=60min)
    threshold_time = 60
    # MPC anticipation time (MPC sees threshold ~20min ahead)
    anticipation_time = 40

    colors = {'MPC': 'blue', 'PID': 'red'}

    for name, df in results.items():
        t_min = np.arange(len(df))
        color = colors[name]

        # Temperature
        axes[0].plot(t_min, df['cabin_temperature'] - 273.15, color=color, label=name, linewidth=1.5)

        # u_hvac
        axes[1].plot(t_min, df['hvac_modulation'], color=color, label=name, drawstyle='steps-post', linewidth=1.5)

        # u_blower
        axes[2].plot(t_min, df['blower_modulation'], color=color, label=name, drawstyle='steps-post', linewidth=1.5)

        # u_recirc (only MPC)
        if 'recirc_modulation' in df.columns:
            axes[3].plot(t_min, df['recirc_modulation'], color=color, label=name, drawstyle='steps-post', linewidth=1.5)

    # Plot SOC profile
    duration_min = int(config.duration_hours * 60)
    t_soc = np.arange(duration_min + 1)
    soc_values = [soc_profile(t * 60) * 100 for t in t_soc]  # Convert to %
    axes[4].plot(t_soc, soc_values, color='green', linewidth=2, label='SOC')
    axes[4].axhline(y=config.soc_threshold * 100, color='red', linestyle='--', label=f'Threshold ({config.soc_threshold*100:.0f}%)')

    # Add markers for key times
    for ax in axes:
        ax.axvline(x=anticipation_time, color='orange', linestyle=':', alpha=0.7, label='MPC sees threshold')
        ax.axvline(x=threshold_time, color='red', linestyle='--', alpha=0.5, label='SOC < 10%')

    # Configure axes
    axes[0].axhline(y=22, color='green', linestyle=':', label='Target')
    axes[0].axhline(y=24, color='grey', linestyle='--', alpha=0.5, label='Comfort band')
    axes[0].axhline(y=20, color='grey', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('T_cabin [°C]')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_title(f'Scenario 3: {config.description}')

    axes[1].set_ylabel('u_hvac [-]')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc='upper right', fontsize=8)

    axes[2].set_ylabel('u_blower [-]')
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc='upper right', fontsize=8)

    axes[3].set_ylabel('u_recirc [-]')
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].legend(loc='upper right', fontsize=8)

    axes[4].set_ylabel('SOC [%]')
    axes[4].set_ylim(-1, 25)
    axes[4].set_xlabel('Time [min]')
    axes[4].legend(loc='upper right', fontsize=8)

    # Add text annotations
    axes[0].annotate('MPC anticipates\n& starts saving',
                     xy=(anticipation_time, 23), xytext=(anticipation_time-15, 26),
                     fontsize=9, color='orange',
                     arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def run_scenario():
    """Run Scenario 3: SOC-Dependent Comfort Relaxation."""

    print("=" * 70)
    print("SCENARIO 3: SOC-Dependent Comfort Relaxation")
    print("=" * 70)

    # Load scenario configuration
    config = soc_relaxation_scenario()
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

    # --- Run PID Baseline (no SOC awareness) ---
    if config.run_pid_baseline:
        print("\n" + "-" * 50)
        print("Running PID Baseline (no SOC awareness)...")
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

        # Add SOC column to PID results for plotting
        soc_values = [config.profile_soc(t * 60) for t in range(len(results['PID']))]
        results['PID']['soc'] = soc_values

        system.close()

    # --- Run MPC with SOC awareness ---
    print("\n" + "-" * 50)
    print("Running MPC with SOC-aware comfort weighting...")
    print("-" * 50)

    system.setup(
        start_time=start_time,
        scenario='summer_city',
        hvac_mode=config.hvac_mode,
        duration=duration_sec,
        profile_overrides=profile_overrides,
        init_overrides=init_overrides,
    )

    # Create SOC-aware MPC wrapper
    soc_mpc = SOCAwareMPCWrapper(config)

    # Run with custom controller wrapper
    dc_mpc = system.run(
        duration=duration_sec,
        controllers=(soc_mpc,),
    )
    results['MPC'] = dc_mpc.df.copy()

    # Add SOC column to MPC results
    soc_values = [config.profile_soc(t * 60) for t in range(len(results['MPC']))]
    results['MPC']['soc'] = soc_values

    # Print mode switches
    if soc_mpc.mode_switches:
        print("\n  MPC Mode Switches:")
        for t_min, current_soc, min_soc in soc_mpc.mode_switches:
            print(f"    t={t_min:.0f}min: switched to low comfort "
                  f"(current SOC={current_soc*100:.1f}%, min in horizon={min_soc*100:.1f}%)")

    # --- Compute and Display Metrics ---
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    metrics_all = {}
    for name, df in results.items():
        metrics = compute_metrics(df, config)
        metrics_all[name] = metrics

        print(f"\n{name}:")
        print(f"  Temperature:")
        print(f"    Mean:   {metrics['T_mean_C']:.1f}°C")
        print(f"    Range:  [{metrics['T_min_C']:.1f}, {metrics['T_max_C']:.1f}]°C")
        print(f"    Final:  {metrics['T_final_C']:.1f}°C")
        print(f"    Max deviation from 22°C: {metrics['T_max_dev_K']:.1f} K")
        print(f"    Comfort violation: {metrics['comfort_violation_Kmin']:.0f} K*min")
        print(f"  Energy:")
        print(f"    Total:            {metrics['energy_total_Wh']:.0f} Wh")
        print(f"    Before threshold: {metrics['energy_before_threshold_Wh']:.0f} Wh")
        print(f"    After threshold:  {metrics['energy_after_threshold_Wh']:.0f} Wh")
        print(f"  Control effort:")
        print(f"    u_hvac mean:        {metrics['u_hvac_mean']:.2f}")
        print(f"    u_hvac mean before: {metrics['u_hvac_mean_before']:.2f}")
        print(f"    u_hvac mean after:  {metrics['u_hvac_mean_after']:.2f}")

    # --- Calculate Improvements ---
    if 'PID' in metrics_all and 'MPC' in metrics_all:
        print("\n" + "-" * 50)
        print("MPC ADVANTAGES vs PID:")
        print("-" * 50)

        energy_diff = metrics_all['PID']['energy_total_Wh'] - metrics_all['MPC']['energy_total_Wh']
        energy_pct = energy_diff / metrics_all['PID']['energy_total_Wh'] * 100 if metrics_all['PID']['energy_total_Wh'] > 0 else 0

        energy_after_diff = metrics_all['PID']['energy_after_threshold_Wh'] - metrics_all['MPC']['energy_after_threshold_Wh']
        energy_after_pct = energy_after_diff / metrics_all['PID']['energy_after_threshold_Wh'] * 100 if metrics_all['PID']['energy_after_threshold_Wh'] > 0 else 0

        print(f"  Energy saved (total): {energy_diff:+.0f} Wh ({energy_pct:+.1f}%)")
        print(f"  Energy saved (after threshold): {energy_after_diff:+.0f} Wh ({energy_after_pct:+.1f}%)")

        T_final_diff = metrics_all['MPC']['T_final_C'] - metrics_all['PID']['T_final_C']
        print(f"  T_final difference: {T_final_diff:+.1f}°C (MPC lets T rise to save energy)")

        print("\n  Key insight: MPC anticipates low SOC ~20min ahead and reduces cooling")
        print("  BEFORE the threshold is reached, preserving battery range.")
        print("  PID has no SOC awareness and keeps cooling at full blast.")

    # --- Plot Results ---
    print("\n" + "-" * 50)
    print("Generating comparison plot...")
    print("-" * 50)

    plot_path = os.path.join(os.path.dirname(__file__), '..', 's3_soc_relaxation_comparison.png')
    plot_comparison(results, config, config.profile_soc, save_path=plot_path)

    # Save metrics and raw data
    save_results(metrics_all, config, dataframes=results)

    return results, metrics_all


if __name__ == "__main__":
    results, metrics = run_scenario()
