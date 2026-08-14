"""
E3 — Coupled battery/SOC model for Scenario 3 and range translation.

The original S3 prescribes SOC(t) linearly (20 % -> 0 % over 2 h). Reviewer 2
(comment 6) correctly notes that HVAC consumption did not feed back into SOC.

Here the SOC becomes a simple integrator:

    SOC(t) = SOC_0 - ( P_base * t + E_hvac(t) ) / E_usable

with E_usable = 60 kWh usable capacity and P_base (traction + autonomy
compute + auxiliaries) calibrated such that the PID run reproduces the
original linear profile: P_base = 0.2*E_usable/2h - mean(P_hvac_PID).

The MPC's weight-set switch now reacts to its own consumption history
(true feedback). The saved energy is additionally translated into remaining
range at the average consumption of the driving mission.

Outputs: coupled S3 MPC run, coupled SOC trajectories for PID/MPC,
range/time extension numbers.
"""

from common import *  # noqa: F401,F403
from scenario_config import soc_relaxation_scenario
from scenarios.s3_soc_relaxation import compute_metrics as metrics_s3, SOCAwareMPCWrapper

E_USABLE_WH = 60000.0          # usable battery capacity [Wh]
SOC_0 = 0.20                   # initial SOC
DURATION_H = 2.0
V_MEAN_MS = 8.0                # S3 city driving speed [m/s]


def load_pid_reference():
    """PID power trace from the paper S3 run (results_backup/current CSVs)."""
    robotaxi_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(robotaxi_dir, 's3_soc_relaxation_pid.csv')
    df = pd.read_csv(path)
    return df


class CoupledSOCMPC(SOCAwareMPCWrapper):
    """SOC wrapper driven by an integrating battery model (true feedback)."""

    def __init__(self, config, P_base_W: float):
        super().__init__(config)
        self.P_base = P_base_W
        self.soc_trace = []

    def _soc_now(self, df):
        t_sec = len(df) * 60
        if 'hvac_power' in df.columns and len(df) > 0:
            e_hvac_Wh = df['hvac_power'].fillna(0.0).sum() * 60 / 3600
            p_hvac_now = float(df['hvac_power'].fillna(0.0).iloc[-1])
        else:
            e_hvac_Wh, p_hvac_now = 0.0, 0.0
        soc = SOC_0 - (self.P_base * t_sec / 3600 + e_hvac_Wh) / E_USABLE_WH
        return soc, p_hvac_now

    def __call__(self, df):
        soc_now, p_hvac_now = self._soc_now(df)
        t_sec = len(df) * 60
        self.soc_trace.append(soc_now)

        # Horizon SOC forecast: current total power held constant
        p_total = self.P_base + p_hvac_now
        soc_horizon_min = soc_now - p_total * (20 * 60) / 3600 / E_USABLE_WH

        old_mode = self.current_mode
        if min(soc_now, soc_horizon_min) < self.threshold:
            self.current_mode = 'saving'
            controls, extra = self.mpc_saving(df)
        else:
            self.current_mode = 'comfort'
            controls, extra = self.mpc_comfort(df)

        if self.current_mode != old_mode:
            t_min = t_sec / 60
            self.mode_switches.append((t_min, soc_now, soc_horizon_min))
            print(f"  [t={t_min:.0f}min] Coupled mode switch: {old_mode} -> {self.current_mode} "
                  f"(SOC={soc_now*100:.1f}%, min in horizon={soc_horizon_min*100:.1f}%)")
        return controls, extra


def main():
    results = {}

    # --- Calibrate P_base from the PID reference run -----------------------
    pid = load_pid_reference()
    e_hvac_pid_Wh = pid['P_hvac_W'].sum() * 60 / 3600
    p_hvac_pid_mean = e_hvac_pid_Wh / DURATION_H
    p_total_nominal = SOC_0 * E_USABLE_WH / DURATION_H       # 6000 W
    p_base = p_total_nominal - p_hvac_pid_mean               # ~5067 W
    consumption_Wh_km = p_total_nominal / (V_MEAN_MS * 3.6)  # per km at 28.8 km/h

    print(f"Calibration: E_usable={E_USABLE_WH/1000:.0f} kWh, "
          f"P_total_nominal={p_total_nominal:.0f} W, P_hvac_PID={p_hvac_pid_mean:.0f} W, "
          f"P_base={p_base:.0f} W, consumption={consumption_Wh_km:.0f} Wh/km")

    results['calibration'] = {
        'E_usable_Wh': E_USABLE_WH,
        'P_total_nominal_W': p_total_nominal,
        'P_hvac_PID_mean_W': p_hvac_pid_mean,
        'P_base_W': p_base,
        'consumption_Wh_per_km': consumption_Wh_km,
    }

    # --- Coupled MPC run ---------------------------------------------------
    print("=" * 70)
    print("E3 — S3 with coupled battery/SOC model")
    print("=" * 70)
    cfg = soc_relaxation_scenario()
    holder = {}

    def factory():
        holder['w'] = CoupledSOCMPC(cfg, P_base_W=p_base)
        return holder['w']

    df = run_mpc(cfg, factory)
    m = metrics_s3(df, cfg)
    m['mode_switches'] = holder['w'].mode_switches

    # --- Coupled SOC trajectories (post-processing, both controllers) ------
    def coupled_soc_series(p_hvac_series):
        e_cum = p_hvac_series.fillna(0.0).cumsum() * 60 / 3600  # Wh
        t_h = np.arange(len(p_hvac_series)) / 60
        return SOC_0 - (p_base * t_h + e_cum) / E_USABLE_WH

    soc_pid = coupled_soc_series(pid['P_hvac_W'])
    soc_mpc = coupled_soc_series(df['hvac_power'].reset_index(drop=True))

    e_hvac_mpc_Wh = df['hvac_power'].fillna(0.0).sum() * 60 / 3600
    delta_e_Wh = e_hvac_pid_Wh - e_hvac_mpc_Wh
    soc_final_pid = float(soc_pid.iloc[-1])
    soc_final_mpc = float(soc_mpc.iloc[-1])
    range_ext_km = (soc_final_mpc - soc_final_pid) * E_USABLE_WH / consumption_Wh_km
    time_ext_min = (soc_final_mpc - soc_final_pid) * E_USABLE_WH / p_total_nominal * 60

    m['energy_hvac_Wh'] = e_hvac_mpc_Wh
    m['soc_final'] = soc_final_mpc
    results['MPC_coupled'] = m
    results['comparison'] = {
        'energy_hvac_PID_Wh': e_hvac_pid_Wh,
        'energy_hvac_MPC_Wh': e_hvac_mpc_Wh,
        'delta_E_Wh': delta_e_Wh,
        'soc_final_PID': soc_final_pid,
        'soc_final_MPC': soc_final_mpc,
        'range_extension_km': range_ext_km,
        'driving_time_extension_min': time_ext_min,
    }

    export_csv(df, os.path.join(RESULTS_DIR, 'e3_s3_mpc_coupled.csv'))
    # store coupled SOC traces for plotting
    pd.DataFrame({
        'time_min': np.arange(len(soc_pid)),
        'soc_coupled_PID': soc_pid,
        'soc_coupled_MPC': soc_mpc[:len(soc_pid)],
    }).to_csv(os.path.join(RESULTS_DIR, 'e3_soc_traces.csv'), index=False)

    print(json.dumps(results['comparison'], indent=2, default=float))
    save_json(results, 'e3_soc_coupling.json')
    print("\nE3 done.")


if __name__ == '__main__':
    main()
