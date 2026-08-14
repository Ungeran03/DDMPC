"""
E5 — True-electrical-power objective vs. quadratic surrogate
(reviewer 2, comment 4).

The paper's MPC penalizes u_hvac^2 and u_blower^2 as smooth surrogates for
electrical power. Here the surrogates are replaced by the actual electrical
power characteristics used in the plant/energy accounting:

    P_blower ∝ u_blower^3                       (fan affinity law)
    P_hp,el  ∝ u_hvac / (1 + alpha_plr*(1-u))   (PLR-corrected COP)

Both cost shapes are normalized to equal the surrogate's cost at u = 1
(same weight scale), so the comparison isolates the SHAPE of the energy
penalty. S1 and S2 are re-run with the true-power objective; if the actuator
allocation and closed-loop energy stay close to the surrogate solution, the
surrogate is validated.
"""

from common import *  # noqa: F401,F403
from scenario_config import preconditioning_scenario, highway_anticipation_scenario
from scenarios.s1_preconditioning import compute_metrics as metrics_s1
from scenarios.s2_highway_anticipation import compute_metrics as metrics_s2
from ddmpc.controller.model_predictive.costs import Cost


class BlowerElectricalCost(Cost):
    """Cubic fan-affinity electrical power, weight = cost at u=1."""

    def __call__(self, mx):
        return self.weight * mx ** 3

    def __str__(self):
        return 'BlowerElectricalCost'


class HPElectricalCost(Cost):
    """PLR-corrected compressor electrical power, weight = cost at u=1."""

    def __init__(self, weight: float, alpha_plr: float = 0.3):
        super().__init__(weight=weight)
        self.alpha_plr = alpha_plr

    def __call__(self, mx):
        return self.weight * mx / (1 + self.alpha_plr * (1 - mx))

    def __str__(self):
        return 'HPElectricalCost'


def true_power_costs(weights: dict) -> dict:
    return {
        'u_hvac': HPElectricalCost(weight=weights['energy_hvac']),
        'u_blower': BlowerElectricalCost(weight=weights['energy_blower']),
    }


def main():
    results = {}

    # --- S1 -----------------------------------------------------------------
    print("=" * 70)
    print("E5 — S1 with true-electrical-power objective")
    print("=" * 70)
    cfg1 = preconditioning_scenario()
    df1 = run_mpc(cfg1, lambda: build_mpc_from_config(
        cfg1, energy_costs=true_power_costs(cfg1.weights)))
    m1 = metrics_s1(df1, cfg1)
    m1['u_hvac_mean'] = float(df1['hvac_modulation'].mean())
    m1['u_blower_mean'] = float(df1['blower_modulation'].mean())
    results['S1_true_power'] = m1
    export_csv(df1, os.path.join(RESULTS_DIR, 'e5_s1_true_power.csv'))
    print(json.dumps(m1, indent=2, default=float))

    # --- S2 -----------------------------------------------------------------
    print("=" * 70)
    print("E5 — S2 with true-electrical-power objective")
    print("=" * 70)
    cfg2 = highway_anticipation_scenario()
    df2 = run_mpc(cfg2, lambda: build_mpc_from_config(
        cfg2, energy_costs=true_power_costs(cfg2.weights)))
    m2 = metrics_s2(df2, cfg2)
    m2['u_hvac_mean'] = float(df2['hvac_modulation'].mean())
    m2['u_blower_mean'] = float(df2['blower_modulation'].mean())
    results['S2_true_power'] = m2
    export_csv(df2, os.path.join(RESULTS_DIR, 'e5_s2_true_power.csv'))
    print(json.dumps(m2, indent=2, default=float))

    save_json(results, 'e5_true_power.json')
    print("\nE5 done.")


if __name__ == '__main__':
    main()
