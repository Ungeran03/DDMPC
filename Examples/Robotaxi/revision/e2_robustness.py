"""
E2 — Robustness to forecast errors.

(a) S1 boarding-time error: the booking system announces boarding at t=8 min,
    but passengers actually board 3 min late (t=11) or 3 min early (t=5).
    The plant uses the ACTUAL schedule; the MPC forecast uses the BELIEVED one.

(b) S3 SOC estimation bias: the BMS SOC estimate that drives the weight-set
    switch is biased by +2 %-pts (optimistic, switches late) or -2 %-pts
    (pessimistic, switches early). Plant/metrics use the true SOC.

Addresses reviewer 1 (forecast uncertainty question) and reviewer 2 (comment 5).
"""

from common import *  # noqa: F401,F403
from scenario_config import preconditioning_scenario, soc_relaxation_scenario
from scenarios.s1_preconditioning import compute_metrics as metrics_s1
from scenarios.s3_soc_relaxation import compute_metrics as metrics_s3, SOCAwareMPCWrapper


def believed_passenger_profile(t_sec: float) -> int:
    """Nominal booking schedule: 3 passengers from t=8 to t=25 min."""
    if t_sec < 8 * 60:
        return 0
    elif t_sec < 25 * 60:
        return 3
    return 0


def actual_passenger_profile(shift_min: float):
    """Actual boarding shifted by shift_min (alighting unchanged at t=25)."""
    def profile(t_sec: float) -> int:
        if t_sec < (8 + shift_min) * 60:
            return 0
        elif t_sec < 25 * 60:
            return 3
        return 0
    return profile


import dataclasses


class BiasedSOCMPC(SOCAwareMPCWrapper):
    """SOC wrapper whose decision signal is biased by a constant offset."""

    def __init__(self, config, soc_bias: float):
        super().__init__(config)
        self.soc_bias = soc_bias
        true_profile = config.profile_soc
        # Decision signal = true SOC + bias (clipped to [0, 1]);
        # plant profiles and metrics keep the true SOC.
        self.config = dataclasses.replace(
            config,
            profile_soc=lambda t: float(np.clip(true_profile(t) + soc_bias, 0.0, 1.0)),
        )


def main():
    results = {}

    # ------------------------------------------------------------------ (a)
    for label, shift in [('boarding_3min_late', +3.0), ('boarding_3min_early', -3.0)]:
        print("=" * 70)
        print(f"E2a — S1 with {label} (MPC believes t=8 min)")
        print("=" * 70)
        cfg = preconditioning_scenario()
        cfg.profile_passengers = actual_passenger_profile(shift)

        biased_forecast = make_profile_forecast('n_passengers', believed_passenger_profile)
        df = run_mpc(cfg, lambda: build_mpc_from_config(cfg, forecast_callback=biased_forecast))
        m = metrics_s1(df, cfg, boarding_time_sec=int((8 + shift) * 60))
        results[f'S1_{label}'] = m
        export_csv(df, os.path.join(RESULTS_DIR, f'e2_s1_{label}.csv'))
        print(json.dumps(m, indent=2, default=float))

    # ------------------------------------------------------------------ (b)
    for label, bias in [('soc_plus2pct', +0.02), ('soc_minus2pct', -0.02)]:
        print("=" * 70)
        print(f"E2b — S3 with SOC estimation bias {bias:+.0%}")
        print("=" * 70)
        cfg = soc_relaxation_scenario()
        holder = {}

        def factory():
            holder['w'] = BiasedSOCMPC(cfg, bias)
            return holder['w']

        df = run_mpc(cfg, factory)
        m = metrics_s3(df, cfg)
        m['mode_switches'] = holder['w'].mode_switches
        results[f'S3_{label}'] = m
        export_csv(df, os.path.join(RESULTS_DIR, f'e2_s3_{label}.csv'))
        print(json.dumps(m, indent=2, default=float))

    save_json(results, 'e2_robustness.json')
    print("\nE2 done.")


if __name__ == '__main__':
    main()
