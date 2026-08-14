"""
E1 — Ablation study: forecast-aware MPC vs. identical MPC without preview.

The "MPC-NF" (no forecast) controller is the IDENTICAL nonlinear MPC
(same model, objective, weights, constraints, horizon) but every forecast
is replaced by a constant hold of the current measurement. For S3, the SOC
mode switch additionally becomes reactive (current SOC only, no horizon
minimum). This isolates:

    PID  ->  MPC-NF      benefit of nonlinear multi-actuator optimization
    MPC-NF -> MPC        benefit of the robotaxi forecast information

Addresses reviewer 1 (last comment) and reviewer 2 (comment 1).
"""

from common import *  # noqa: F401,F403
from scenario_config import (
    preconditioning_scenario,
    highway_anticipation_scenario,
    soc_relaxation_scenario,
)
from scenarios.s1_preconditioning import compute_metrics as metrics_s1
from scenarios.s2_highway_anticipation import compute_metrics as metrics_s2
from scenarios.s3_soc_relaxation import compute_metrics as metrics_s3


class ReactiveSOCMPC:
    """
    S3 wrapper without anticipation: selects the weight set on the CURRENT
    SOC only (switch happens when the threshold is actually crossed), and
    both internal MPCs use the frozen forecast.
    """

    def __init__(self, config, forecast_callback):
        self.config = config
        self.threshold = config.soc_threshold
        self.mpc_comfort = build_mpc_from_config(config, forecast_callback=forecast_callback)
        self.mpc_saving = build_mpc_from_config(
            config, weight_overrides=config.weights_saving, forecast_callback=forecast_callback)
        self.current_mode = 'comfort'
        self.mode_switches = []
        self.step_size = one_minute

    def __call__(self, df):
        current_time_sec = len(df) * 60
        soc_now = self.config.profile_soc(current_time_sec)

        old_mode = self.current_mode
        if soc_now < self.threshold:
            self.current_mode = 'saving'
            controls, extra = self.mpc_saving(df)
        else:
            self.current_mode = 'comfort'
            controls, extra = self.mpc_comfort(df)

        if self.current_mode != old_mode:
            t_min = current_time_sec / 60
            self.mode_switches.append((t_min, soc_now))
            print(f"  [t={t_min:.0f}min] Reactive mode switch: {old_mode} -> {self.current_mode} "
                  f"(SOC={soc_now*100:.1f}%)")
        return controls, extra


def main():
    results = {}

    # --- S1 -----------------------------------------------------------------
    print("=" * 70)
    print("E1 ABLATION — S1 Pre-Conditioning, MPC with frozen forecast")
    print("=" * 70)
    cfg1 = preconditioning_scenario()
    df1 = run_mpc(cfg1, lambda: build_mpc_from_config(cfg1, forecast_callback=frozen_forecast))
    m1 = metrics_s1(df1, cfg1)
    results['S1'] = {'MPC_NF': m1}
    export_csv(df1, os.path.join(RESULTS_DIR, 'e1_s1_mpc_nf.csv'))
    print(json.dumps(m1, indent=2, default=float))

    # --- S2 -----------------------------------------------------------------
    print("=" * 70)
    print("E1 ABLATION — S2 Highway Anticipation, MPC with frozen forecast")
    print("=" * 70)
    cfg2 = highway_anticipation_scenario()
    df2 = run_mpc(cfg2, lambda: build_mpc_from_config(cfg2, forecast_callback=frozen_forecast))
    m2 = metrics_s2(df2, cfg2)
    results['S2'] = {'MPC_NF': m2}
    export_csv(df2, os.path.join(RESULTS_DIR, 'e1_s2_mpc_nf.csv'))
    print(json.dumps(m2, indent=2, default=float))

    # --- S3 -----------------------------------------------------------------
    print("=" * 70)
    print("E1 ABLATION — S3 SOC Relaxation, reactive switch + frozen forecast")
    print("=" * 70)
    cfg3 = soc_relaxation_scenario()
    wrapper_holder = {}

    def factory():
        wrapper_holder['w'] = ReactiveSOCMPC(cfg3, frozen_forecast)
        return wrapper_holder['w']

    df3 = run_mpc(cfg3, factory)
    m3 = metrics_s3(df3, cfg3)
    m3['mode_switches'] = wrapper_holder['w'].mode_switches
    results['S3'] = {'MPC_NF': m3}
    export_csv(df3, os.path.join(RESULTS_DIR, 'e1_s3_mpc_nf.csv'))
    print(json.dumps(m3, indent=2, default=float))

    save_json(results, 'e1_ablation.json')
    print("\nE1 done.")


if __name__ == '__main__':
    main()
