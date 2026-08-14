"""
Shared MPC builder for the paper scenarios (S1-S3).

Centralizes the NLP construction so that all scenarios use the same,
verified formulation. Key correctness measures (see paper_revision_notes.md):

1. The NLP uses a REDUCED model containing only the features that are
   predicted, actuated, or forecast. The previous formulation used
   Model(*Feature.all), which turned every Controlled feature without a
   predictor (T_ptc_elem, T_mass, P_hvac and their Change connections)
   into FREE optimization variables over the horizon. The free T_ptc fed
   the T_vent predictor with an unphysical 200 W/K heat source/sink that
   the solver exploited to fake cooling; the remaining dangling variables
   degraded the KKT conditioning and made IPOPT stall at its iteration
   limit.
2. The cooling-mode T_vent WhiteBox does not use T_ptc as an input
   (quasi-stationary PTC element, see s3_T_cabin_WB.py).
3. T_vent is constrained to its physical limits [5, 65]C, matching the
   simulator (evaporator icing / safety clip). Without this, the predictor
   could plan duct temperatures the plant cannot deliver.
4. Inactive MVs (u_ptc in cooling scenarios) are excluded from the NLP
   entirely; the simulator keeps them at their default (0).
5. Electrical energy is penalized for BOTH the heat pump and the blower
   fan, matching the plant model in which the blower draws
   P_blower_max * u_blower^3 (fan affinity law).

Objective structure (implemented == what the paper must state):

    J = sum_k [ w_T * (T_cabin,k - 22C)^2                      (comfort)
              + w_CO2 * (max(0, CO2_k - 800)/100)^2            (air quality)
              + w_E,hvac * u_hvac,k^2                          (HP energy)
              + w_E,blower * u_blower,k^2                      (fan energy)
              + sum_j w_du,j * du_j,k^2 ]                      (smoothness)

The CO2 soft target enters through the Economic mode of the C_CO2 feature
(band [400, 800] ppm, epsilon formulation); the 1200 ppm Pettenkofer limit
is a hard constraint.
"""

from configuration import *
from scenario_config import ScenarioConfig, MVConfig
from s3_T_cabin_WB import (
    create_whitebox_T_ptc,
    create_whitebox_T_vent,
    create_whitebox_T_cabin,
    create_whitebox_CO2,
)
from ddmpc.controller.model_predictive.nlp import NLP, Objective, Constraint
from ddmpc.controller.model_predictive.costs import Quadratic

# Physical T_vent limits, must match cabin_simulator (T_vent_min/max)
T_VENT_MIN = 278.15  # 5C evaporator icing limit
T_VENT_MAX = 338.15  # 65C safety limit


def build_mpc_model(ptc_active: bool) -> Model:
    """
    Reduced feature set for the MPC NLP: predicted states, actuated MVs and
    forecast disturbances only. Everything else (T_mass, P_hvac, tracking
    features, unused connections) stays out of the optimization problem.
    """
    features = [
        # predicted states + their change connections (predictor outputs)
        T_cabin, T_cabin_change,
        T_vent, T_vent_change,
        C_CO2, C_CO2_change,
        # manipulated variables + move connections
        u_hvac, u_hvac_change,
        u_blower, u_blower_change,
        u_recirc, u_recirc_change,
        # forecast disturbances used by the predictors
        T_ambient, solar_radiation, heading, n_passengers, v_vehicle,
    ]
    if ptc_active:
        features += [u_ptc, u_ptc_change, T_ptc_elem, T_ptc_elem_change]

    return Model(*features)


def build_mpc_from_config(config: ScenarioConfig, weight_overrides: dict = None,
                          forecast_callback=None, energy_costs: dict = None):
    """
    Build the MPC controller for a paper scenario.

    :param config: scenario configuration (MVs, weights, limits)
    :param weight_overrides: optional dict overriding config.weights
        (used by the S3 SOC wrapper to build comfort / saving mode NLPs)
    :param forecast_callback: optional replacement for system.get_forecast
        (used by the revision experiments: frozen forecast for the ablation,
        biased forecast for the robustness tests)
    :param energy_costs: optional dict {'u_hvac': Cost, 'u_blower': Cost}
        replacing the default Quadratic energy costs (used by the
        true-electrical-power objective experiment)
    """

    weights = dict(config.weights)
    if weight_overrides:
        weights.update(weight_overrides)

    mv_cfg = config.mv_config
    ptc_active = mv_cfg.get('u_ptc', MVConfig(active=False)).active

    # --- Predictors -------------------------------------------------------
    wb_T_vent = create_whitebox_T_vent(hvac_mode=config.hvac_mode)
    wb_T_cabin = create_whitebox_T_cabin()
    wb_CO2 = create_whitebox_CO2()

    predictors = [wb_T_vent, wb_T_cabin, wb_CO2]
    if ptc_active:
        # exact-discretization PTC element predictor (heating scenarios only)
        predictors.insert(0, create_whitebox_T_ptc())

    # --- Objectives -------------------------------------------------------
    objectives = []

    # Comfort: quadratic tracking of the 22C target (Steady mode epsilons)
    if weights.get('T_cabin', 0) > 0:
        objectives.append(Objective(
            feature=T_cabin,
            cost=Quadratic(weight=weights['T_cabin']),
        ))

    # Air quality: quadratic penalty on CO2 above the 800 ppm soft target
    # (Economic mode epsilon, normalized to 100 ppm units)
    if weights.get('C_CO2', 0) > 0:
        objectives.append(Objective(
            feature=C_CO2,
            cost=Quadratic(weight=weights['C_CO2'], norm=100.0),
        ))

    # Energy: electrical power of heat pump and blower fan
    energy_costs = energy_costs or {}
    if weights.get('energy_hvac', 0) > 0:
        objectives.append(Objective(
            feature=u_hvac,
            cost=energy_costs.get('u_hvac', Quadratic(weight=weights['energy_hvac'])),
        ))
    if weights.get('energy_blower', 0) > 0:
        objectives.append(Objective(
            feature=u_blower,
            cost=energy_costs.get('u_blower', Quadratic(weight=weights['energy_blower'])),
        ))
    if ptc_active and weights.get('energy_ptc', 0) > 0:
        objectives.append(Objective(
            feature=u_ptc,
            cost=Quadratic(weight=weights['energy_ptc']),
        ))

    # Smoothness: control move penalties
    if mv_cfg['u_hvac'].active:
        objectives.append(Objective(feature=u_hvac_change,
                                    cost=Quadratic(weight=mv_cfg['u_hvac'].weight_change)))
    if ptc_active:
        objectives.append(Objective(feature=u_ptc_change,
                                    cost=Quadratic(weight=mv_cfg['u_ptc'].weight_change)))
    if mv_cfg['u_blower'].active:
        objectives.append(Objective(feature=u_blower_change,
                                    cost=Quadratic(weight=mv_cfg['u_blower'].weight_change)))
    if mv_cfg['u_recirc'].active:
        objectives.append(Objective(feature=u_recirc_change,
                                    cost=Quadratic(weight=mv_cfg['u_recirc'].weight_change)))

    # --- Constraints ------------------------------------------------------
    constraints = []

    # MV box constraints (u_ptc excluded from the model when inactive)
    if mv_cfg['u_hvac'].active:
        constraints.append(Constraint(feature=u_hvac, lb=mv_cfg['u_hvac'].lb, ub=mv_cfg['u_hvac'].ub))
    if ptc_active:
        constraints.append(Constraint(feature=u_ptc, lb=mv_cfg['u_ptc'].lb, ub=mv_cfg['u_ptc'].ub))
    if mv_cfg['u_blower'].active:
        constraints.append(Constraint(feature=u_blower, lb=mv_cfg['u_blower'].lb, ub=mv_cfg['u_blower'].ub))
    if mv_cfg['u_recirc'].active:
        constraints.append(Constraint(feature=u_recirc, lb=mv_cfg['u_recirc'].lb, ub=mv_cfg['u_recirc'].ub))

    # CO2 hard constraint (Pettenkofer limit)
    constraints.append(Constraint(feature=C_CO2, lb=0, ub=config.CO2_limit))

    # T_vent physical limits (matches simulator clip)
    constraints.append(Constraint(feature=T_vent, lb=T_VENT_MIN, ub=T_VENT_MAX))

    # --- Assemble ---------------------------------------------------------
    mpc = ModelPredictive(
        step_size=one_minute,
        nlp=NLP(
            model=build_mpc_model(ptc_active),
            N=20,  # 20-minute horizon
            control_change_step=1,
            objectives=objectives,
            constraints=constraints,
        ),
        forecast_callback=forecast_callback if forecast_callback is not None else system.get_forecast,
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

    mpc.nlp.build(solver_options=solver_options, predictors=predictors)

    return mpc
