"""
E4 — Parameter sensitivity of the reported S2 savings (reviewer 2, comment 3).

Each of the five most influential model parameters is perturbed by +/-20 %;
PID and MPC are re-run on Scenario 2 and the relative savings recomputed.

Perturbation semantics:
  - alpha_plr:          plant only (enters electrical power, not thermal
                        dynamics; the MPC objective does not use it)
  - UA_opaque+UA_window: plant AND predictor (consistent re-identification)
  - m_dot_blower_max:   plant AND predictor (consistent re-identification)
  - C_mass:             plant only (hidden from the MPC by design)
  - h_conv:             plant only; the predictor keeps its nominal 40 W/K
                        phantom-UA bias correction -> deliberate model
                        mismatch test of the quasi-steady T_mass reduction
                        (also addresses reviewer 3, comment 3)
"""

from common import *  # noqa: F401,F403
from scenario_config import highway_anticipation_scenario
from scenarios.s2_highway_anticipation import compute_metrics as metrics_s2
import s3_T_cabin_WB as wb

NOMINAL = {
    'alpha_plr': 0.3,
    'UA': (15.0, 12.5),          # (UA_opaque, UA_window)
    'm_dot_blower_max': 0.08,
    'C_mass': 120000.0,
    'h_conv': 10.0,
}


def apply_variant(param: str, factor: float):
    """Set plant (and where applicable predictor) parameters."""
    if param == 'alpha_plr':
        system.alpha_plr = NOMINAL['alpha_plr'] * factor
    elif param == 'UA':
        ua_o, ua_w = NOMINAL['UA']
        system.UA_opaque = ua_o * factor
        system.UA_window = ua_w * factor
        wb.UA_opaque = ua_o * factor
        wb.UA_window = ua_w * factor
    elif param == 'm_dot_blower_max':
        system.m_dot_blower_max = NOMINAL['m_dot_blower_max'] * factor
        wb.m_dot_blower_max = NOMINAL['m_dot_blower_max'] * factor
    elif param == 'C_mass':
        system.C_mass = NOMINAL['C_mass'] * factor
    elif param == 'h_conv':
        system.h_conv = NOMINAL['h_conv'] * factor
    else:
        raise ValueError(param)


def restore_all():
    system.alpha_plr = NOMINAL['alpha_plr']
    system.UA_opaque, system.UA_window = NOMINAL['UA']
    system.m_dot_blower_max = NOMINAL['m_dot_blower_max']
    system.C_mass = NOMINAL['C_mass']
    system.h_conv = NOMINAL['h_conv']
    wb.UA_opaque, wb.UA_window = NOMINAL['UA']
    wb.m_dot_blower_max = NOMINAL['m_dot_blower_max']


def run_variant(param: str, factor: float) -> dict:
    label = f"{param}_{'+' if factor > 1 else '-'}20pct"
    print("=" * 70)
    print(f"E4 — S2 with {param} x {factor:.1f}")
    print("=" * 70)
    apply_variant(param, factor)
    try:
        cfg = highway_anticipation_scenario()
        df_pid = run_pid(cfg)
        df_mpc = run_mpc(cfg, lambda: build_mpc_from_config(cfg))
        m_pid = metrics_s2(df_pid, cfg)
        m_mpc = metrics_s2(df_mpc, cfg)
    finally:
        restore_all()

    savings = (1 - m_mpc['energy_total_Wh'] / m_pid['energy_total_Wh']) * 100
    out = {
        'param': param,
        'factor': factor,
        'PID_energy_Wh': m_pid['energy_total_Wh'],
        'MPC_energy_Wh': m_mpc['energy_total_Wh'],
        'savings_pct': savings,
        'PID_T_mean_C': m_pid['T_mean_C'],
        'MPC_T_mean_C': m_mpc['T_mean_C'],
        'MPC_T_max_C': m_mpc['T_max_C'],
        'MPC_time_in_band_pct': m_mpc['time_in_band_pct'],
    }
    print(f"  -> PID {m_pid['energy_total_Wh']:.0f} Wh, MPC {m_mpc['energy_total_Wh']:.0f} Wh, "
          f"savings {savings:.1f} %")
    return out


def main():
    results = {'nominal_savings_pct': 55.5, 'variants': []}
    for param in ['alpha_plr', 'UA', 'm_dot_blower_max', 'C_mass', 'h_conv']:
        for factor in [0.8, 1.2]:
            results['variants'].append(run_variant(param, factor))
    save_json(results, 'e4_sensitivity.json')

    print("\nSummary (savings %):")
    for v in results['variants']:
        print(f"  {v['param']:<18} x{v['factor']:.1f}: {v['savings_pct']:.1f} %")
    print("\nE4 done.")


if __name__ == '__main__':
    main()
