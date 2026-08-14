"""
Decisive plausibility test for the ablation wiring:

Capture the PLANNED horizon trajectories (the NLP solution over k = 0..N)
of the forecast-aware MPC and the frozen-forecast MPC at the same closed-loop
state in Scenario 2, during the city phase (t = 0..4 min) when the highway
transition at t = 3..5 min is inside the horizon.

If the forecast reaches the optimizer, the PLANS must differ visibly
(the forecast plan knows eta_rad rises at k >= 5, the frozen plan does not),
even though the EXECUTED first moves coincide.
"""

from common import *  # noqa: F401,F403
from scenario_config import highway_anticipation_scenario

CAPTURE_MIN = 5  # capture plans for the first 5 MPC steps


def run_and_capture(forecast_callback):
    cfg = highway_anticipation_scenario()
    duration_sec = 5 * 60  # only the city/transition phase
    system.setup(
        start_time=int(cfg.start_time_hours * 3600),
        scenario='summer_city',
        hvac_mode=cfg.hvac_mode,
        duration=int(cfg.duration_hours * 3600),
        profile_overrides=cfg.get_profile_overrides(),
        init_overrides={
            'T_cabin': cfg.T_cabin_init, 'T_mass': cfg.T_mass_init,
            'T_vent': cfg.T_vent_init, 'T_ptc': cfg.T_ptc_init,
            'C_CO2': cfg.C_CO2_init,
        },
    )
    mpc = build_mpc_from_config(cfg, forecast_callback=forecast_callback)

    captured = {}
    def capture(df, current_time, _c=captured):
        _c[current_time] = df.copy()
    mpc.save_solution_data = True
    mpc._save_solution = capture

    system.run(duration=duration_sec, controllers=(mpc,))
    system.close()
    return captured


def main():
    print("Running forecast-aware MPC (capture plans)...")
    plans_fc = run_and_capture(None)
    print("Running frozen-forecast MPC (capture plans)...")
    plans_nf = run_and_capture(frozen_forecast)

    t0 = sorted(plans_fc.keys())[0]
    for step, t in enumerate(sorted(plans_fc.keys())[:CAPTURE_MIN]):
        p_fc, p_nf = plans_fc[t], plans_nf[t]
        u_fc = p_fc['hvac_modulation'].to_numpy()
        u_nf = p_nf['hvac_modulation'].to_numpy()
        v_fc = p_fc['vehicle_speed'].to_numpy() if 'vehicle_speed' in p_fc.columns else None
        v_nf = p_nf['vehicle_speed'].to_numpy() if 'vehicle_speed' in p_nf.columns else None
        n = min(len(u_fc), len(u_nf))
        print(f"\n=== MPC step at t = {(t - t0)//60} min ===")
        if v_fc is not None:
            print(f"  v seen by forecast-MPC over horizon: {np.round(v_fc[:8], 1)} ...")
            print(f"  v seen by frozen-MPC  over horizon: {np.round(v_nf[:8], 1)} ...")
        print(f"  planned u_hvac (forecast): {np.round(u_fc[:12], 3)}")
        print(f"  planned u_hvac (frozen):   {np.round(u_nf[:12], 3)}")
        print(f"  max |plan difference| over horizon: {np.abs(u_fc[:n] - u_nf[:n]).max():.4f}")
        print(f"  EXECUTED first move: forecast {u_fc[1] if len(u_fc)>1 else u_fc[0]:.4f} "
              f"vs frozen {u_nf[1] if len(u_nf)>1 else u_nf[0]:.4f}")


if __name__ == '__main__':
    main()
