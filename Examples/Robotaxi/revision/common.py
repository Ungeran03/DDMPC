"""
Shared helpers for the July 2026 revision experiments (WEVJ review round 1).

Experiments:
  E1  Ablation: identical MPC with constant-hold ("frozen") forecasts
  E2  Robustness: forecast timing errors (S1) and SOC estimation bias (S3)
  E3  Coupled battery/SOC model and range translation (S3)
  E4  Parameter sensitivity of the reported savings (S2)
  E5  True-electrical-power objective vs. quadratic surrogate (S1, S2)

All experiments reuse the exact scenario configurations, PID baseline and
MPC builder of the paper scenarios; only the specific aspect under test is
modified. Results are written to revision/results/.
"""

import sys
import os
import json
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROBOTAXI = os.path.dirname(_HERE)
sys.path.insert(0, _ROBOTAXI)

from configuration import *  # noqa: F401,F403  (system, features, one_minute, ...)
from scenario_config import ScenarioConfig
from mpc_builder import build_mpc_from_config
from controllers import FixedPID, BlowerPI, PTCRelay

RESULTS_DIR = os.path.join(_HERE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# System setup / run helpers (mirror scenarios/s*_*.py exactly)
# ---------------------------------------------------------------------------

def setup_system(config: ScenarioConfig) -> int:
    """Set up the plant exactly like the paper scenario scripts do."""
    duration_sec = int(config.duration_hours * 3600)
    start_time = int(config.start_time_hours * 3600)

    system.setup(
        start_time=start_time,
        scenario='summer_city',
        hvac_mode=config.hvac_mode,
        duration=duration_sec,
        profile_overrides=config.get_profile_overrides(),
        init_overrides={
            'T_cabin': config.T_cabin_init,
            'T_mass': config.T_mass_init,
            'T_vent': config.T_vent_init,
            'T_ptc': config.T_ptc_init,
            'C_CO2': config.C_CO2_init,
        },
    )
    return duration_sec


def build_pid_from_config(config: ScenarioConfig):
    """PID baseline, identical to the scenario scripts."""
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


def run_pid(config: ScenarioConfig) -> pd.DataFrame:
    duration_sec = setup_system(config)
    controllers = build_pid_from_config(config)
    dc = system.run(duration=duration_sec, controllers=controllers)
    df = dc.df.copy()
    system.close()
    return df


def run_mpc(config: ScenarioConfig, mpc_factory) -> pd.DataFrame:
    """
    mpc_factory: callable () -> controller (built AFTER system.setup so the
    forecast callback binds to the freshly configured plant).
    """
    duration_sec = setup_system(config)
    mpc = mpc_factory()
    dc = system.run(duration=duration_sec, controllers=(mpc,))
    df = dc.df.copy()
    system.close()  # reset previous_df so consecutive runs in one process stay independent
    return df


# ---------------------------------------------------------------------------
# Forecast wrappers
# ---------------------------------------------------------------------------

def frozen_forecast(horizon_in_seconds: int) -> pd.DataFrame:
    """
    Constant-hold forecast: every disturbance is held at its CURRENT value
    over the whole horizon. The MPC keeps its full multi-actuator NLP but
    loses all preview information (reviewer 2, comment 1).
    """
    df = system.get_forecast(horizon_in_seconds)
    for col in df.columns:
        if col != 'time':
            df[col] = df[col].iloc[0]
    return df


def make_profile_forecast(column: str, believed_profile) -> callable:
    """
    True forecast for all disturbances EXCEPT `column`, which is replaced by
    the (wrong) believed profile evaluated at absolute forecast times.
    Used for the S1 boarding-time error experiment.
    """
    def biased(horizon_in_seconds: int) -> pd.DataFrame:
        df = system.get_forecast(horizon_in_seconds)
        t_rel = df['time'] - system.start_time
        df[column] = [float(believed_profile(t)) for t in t_rel]
        return df
    return biased


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_csv(df: pd.DataFrame, path: str):
    """Standard CSV export (same columns as the paper scenario scripts)."""
    out = pd.DataFrame({
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
        'P_hp_W': df['hp_power'] if 'hp_power' in df.columns else np.nan,
        'P_blower_W': df['blower_power'] if 'blower_power' in df.columns else np.nan,
    })
    if 'soc' in df.columns:
        out['soc'] = df['soc']
    out.to_csv(path, index=False)
    print(f"  CSV: {path}")


def save_json(payload: dict, name: str):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"  JSON: {path}")
    return path
