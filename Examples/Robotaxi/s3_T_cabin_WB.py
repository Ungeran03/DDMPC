"""
Step 3: WhiteBox Model for Cabin Temperature Prediction

This creates a physics-based predictor for the MPC controller.
The model predicts T_cabin_change based on the energy balance.
"""

from configuration import *
from ddmpc.modeling.process_models.physics_based.white_box import WhiteBox
from ddmpc.data_handling.processing_data import TrainingData, Input, Inputs, Output


# =============================================================================
# MODEL PARAMETERS (must match cabin_simulator.py)
# =============================================================================

# Cabin parameters
C_cabin = 50000.0       # Thermal capacity [J/K]
UA = 80.0               # Heat transfer coefficient [W/K]
A_window = 2.5          # Window area [m²]
tau_window = 0.6        # Window transmittance [-]
Q_hvac_max = 5000.0     # Max HVAC power [W]

# Step size
dt = one_minute         # 60 seconds


# =============================================================================
# WHITEBOX MODEL DEFINITION
# =============================================================================

def create_whitebox_predictor(hvac_mode: str = 'cooling') -> WhiteBox:
    """
    Create a WhiteBox predictor for cabin temperature change.

    The energy balance:
        C * dT/dt = Q_hvac + Q_solar + Q_passengers + Q_transmission

    Rearranged for dT:
        dT = (dt/C) * (Q_hvac + Q_solar + Q_passengers + Q_transmission)

    Args:
        hvac_mode: 'cooling' (Q_hvac negative) or 'heating' (Q_hvac positive)
    """

    # Get symbolic variables from sources
    # These are CasADi symbolic variables used in the optimization
    T_cab = T_cabin.source[0]       # Cabin temperature [K]
    T_amb = T_ambient.source[0]     # Ambient temperature [K]
    u = u_hvac.source[0]            # HVAC modulation [0-1]
    I_solar = solar_radiation.source[0]  # Solar irradiance [W/m²]
    n_pass = n_passengers.source[0]      # Number of passengers
    v = v_vehicle.source[0]              # Vehicle velocity [m/s]
    head = heading.source[0]             # Vehicle heading [rad]

    # Radiator efficiency: eta = 0.5 + 0.5 * (1 - exp(-v/v_ref))
    # Approximated for CasADi (smooth function)
    import casadi as ca
    v_ref = 15.0
    eta_radiator = 0.5 + 0.5 * (1 - ca.exp(-v / v_ref))

    # Solar factor based on heading (simplified)
    solar_factor = 0.3 + 0.2 * ca.fabs(ca.sin(head))

    # Heat flows [W]
    Q_transmission = UA * (T_amb - T_cab)
    Q_solar = A_window * tau_window * I_solar * solar_factor
    Q_passengers = n_pass * 90.0

    if hvac_mode == 'cooling':
        Q_hvac = -u * Q_hvac_max * eta_radiator  # Cooling removes heat
    else:
        Q_hvac = u * Q_hvac_max * eta_radiator   # Heating adds heat

    # Total heat flow and temperature change
    Q_total = Q_hvac + Q_solar + Q_passengers + Q_transmission
    dT = (dt / C_cabin) * Q_total

    # Create WhiteBox predictor
    wb = WhiteBox(
        inputs=[
            T_cabin.source,
            T_ambient.source,
            u_hvac.source,
            solar_radiation.source,
            n_passengers.source,
            v_vehicle.source,
            heading.source,
        ],
        output=T_cabin_change,
        output_expression=dT,
        step_size=dt,
    )

    return wb


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    print("=" * 60)
    print("Robotaxi Cabin - WhiteBox Model for T_cabin")
    print("=" * 60)

    # Create predictor for cooling mode (summer scenario)
    wb_cooling = create_whitebox_predictor(hvac_mode='cooling')

    print("\nWhiteBox model created successfully!")
    print(f"  Inputs: {[str(inp.source) for inp in wb_cooling.inputs]}")
    print(f"  Output: {wb_cooling.output.source}")
    print(f"  Step size: {wb_cooling.step_size} s")

    # Test prediction with sample values
    print("\nTest prediction:")
    test_inputs = [
        298.15,     # T_cabin = 25°C
        308.15,     # T_ambient = 35°C
        0.5,        # u_hvac = 50%
        600.0,      # solar = 600 W/m²
        2.0,        # 2 passengers
        10.0,       # 10 m/s velocity
        1.57,       # heading = pi/2
    ]

    dT_pred = float(wb_cooling.predict(test_inputs))
    print(f"  Inputs: T_cab=25°C, T_amb=35°C, u=0.5, solar=600, n=2, v=10m/s")
    print(f"  Predicted dT: {dT_pred:.4f} K per {dt}s step")
    print(f"  Rate: {dT_pred * 3600 / dt:.2f} K/h")

    # Save model
    # wb_cooling.save('T_cabin_WB_cooling', override=True)

    print("\nModel ready for use in MPC.")