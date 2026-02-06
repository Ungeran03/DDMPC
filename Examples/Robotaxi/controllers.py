"""
Custom controllers for Robotaxi cabin climate control.
"""

import numpy as np
import pandas as pd


class FixedPID:
    """
    PID controller with corrected anti-windup logic.

    The standard DDMPC PID has a bug in anti-windup that causes
    the integrator to grow when output is saturated in the wrong direction.
    """

    def __init__(
            self,
            y,              # Controlled feature
            u,              # Control feature
            step_size: int,
            Kp: float = 0.1,
            Ti: float = 100.0,  # Integral time constant [s]
            Td: float = 0.0,    # Derivative time constant [s]
            reverse_act: bool = False,
    ):
        self.y = y
        self.u = u
        self.step_size = step_size
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.reverse_act = reverse_act

        # State
        self.integral = 0.0
        self.last_error = 0.0

    def __call__(self, df: pd.DataFrame) -> tuple[dict, dict]:
        """Calculate control action."""

        # Get error from controlled feature
        e = self.y.error if self.y.error is not None else 0.0

        # Reverse action if needed
        if self.reverse_act:
            e = -e

        # Proportional term
        P = self.Kp * e

        # Integral term with anti-windup
        if self.Ti > 0:
            # Only integrate if not saturated in the wrong direction
            potential_output = P + self.Kp * self.integral

            # Check if we're about to saturate
            if potential_output > self.u.ub and e > 0:
                # Would saturate high and error is positive -> don't integrate more
                pass
            elif potential_output < self.u.lb and e < 0:
                # Would saturate low and error is negative -> don't integrate more
                pass
            else:
                # Safe to integrate
                self.integral += (1 / self.Ti) * e * self.step_size

            # Clamp integral to prevent excessive windup
            max_integral = (self.u.ub - self.u.lb) / self.Kp if self.Kp > 0 else 100
            self.integral = np.clip(self.integral, -max_integral, max_integral)

        I = self.Kp * self.integral

        # Derivative term
        if self.Td > 0 and self.step_size > 0:
            D = self.Kp * self.Td * (e - self.last_error) / self.step_size
        else:
            D = 0.0

        self.last_error = e

        # Total output
        output = P + I + D

        # Clamp to control limits
        output = np.clip(output, self.u.lb, self.u.ub)

        return {self.u.source.col_name: output}, {}

    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.last_error = 0.0


class BlowerPI:
    """
    PI controller for blower fan speed using product error.

    Uses product error: e = |T_cabin - T_target| × |T_amb - T_target| / scale
    This keeps blower higher in extreme conditions even when T_cabin is near target.

    Physics motivation:
    - Heat load scales with |T_amb - T_target| (transmission, fresh air)
    - Required blower scales with heat load
    - Product error captures this dependency

    - Extreme conditions (summer/winter): high e_product → high blower
    - Mild conditions: low e_product → lower blower OK
    - Near target: small T_cabin error reduces blower appropriately
    """

    def __init__(
            self,
            y,                      # Controlled feature (T_cabin)
            u,                      # Control feature (u_blower)
            T_amb_feature,          # Disturbance feature (T_ambient)
            step_size: int,
            Kp: float = 0.15,       # Proportional gain (lower due to product scaling)
            Ti: float = 200.0,      # Integral time constant [s]
            deadband: float = 1.0,  # Deadband for integrator [product units]
            scale: float = 10.0,    # Scale factor for product error
            u_min: float = 0.15,    # Minimum blower (ventilation)
    ):
        self.y = y
        self.u = u
        self.T_amb_feature = T_amb_feature
        self.step_size = step_size
        self.Kp = Kp
        self.Ti = Ti
        self.deadband = deadband
        self.scale = scale
        self.u_min = u_min

        self.integral = 0.0

    def __call__(self, df: pd.DataFrame) -> tuple[dict, dict]:
        """Calculate blower control action using product error."""

        # Get temperatures
        T_cabin = self.y.value if self.y.value is not None else 293.15
        T_target = self.y.target if self.y.target is not None else 295.15
        T_amb_col = self.T_amb_feature.source.col_name
        T_amb = float(df[T_amb_col].iloc[-1])

        # Product error: scales with both cabin error AND environmental extremity
        e_cabin = abs(T_cabin - T_target)  # [K]
        e_env = abs(T_amb - T_target)      # [K]
        e_product = e_cabin * e_env / self.scale  # dimensionless after scaling

        # Proportional term
        P = self.Kp * e_product

        # Integral term with deadband
        if self.Ti > 0:
            e_I = e_product - self.deadband  # negative when product is small

            # Anti-windup: don't integrate up when output is saturated high
            potential = P + self.Kp * self.integral
            if potential >= self.u.ub and e_I > 0:
                pass  # saturated high, don't integrate further
            else:
                self.integral += (1 / self.Ti) * e_I * self.step_size

            # Clip integral to [0, max] — never negative
            max_integral = (self.u.ub - self.u.lb) / self.Kp if self.Kp > 0 else 100
            self.integral = np.clip(self.integral, 0, max_integral)

        I = self.Kp * self.integral

        output = P + I

        # Enforce minimum blower for ventilation
        output = max(self.u_min, output)
        output = np.clip(output, self.u.lb, self.u.ub)

        return {self.u.source.col_name: output}, {}

    def reset(self):
        self.integral = 0.0


class PTCRelay:
    """
    Relay controller for PTC heater with hysteresis.

    Reads T_ptc from the simulator (4-node model) and applies relay logic:
    1. If T_env >= T_setpoint: PTC OFF (no heating needed)
    2. If T_cabin >= T_setpoint - margin: PTC OFF (close enough, let HP fine-tune)
    3. Otherwise: relay based on T_ptc from simulator:
       - T_ptc > T_off (65°C): relay OFF
       - T_ptc < T_on  (50°C): relay ON
       - Between: maintain previous state (hysteresis)

    The simulator handles T_ptc dynamics:
        C_ptc * dT_ptc/dt = u_ptc * Q_ptc_max - h_ptc * (T_ptc - T_vent)
        T_ptc_ss ≈ T_vent + Q_ptc_max / h_ptc = T_vent + 30K

    Thresholds tuned for 4-node model with C_ptc=1500 J/K, h_ptc=200 W/K:
    - With these dynamics, T_ptc reaches ~50-65°C during operation
    - T_on=50°C, T_off=65°C ensures cycling during warmup
    - Too low thresholds (35-45°C) cause T_ptc to stabilize in hysteresis band

    For MPC: not used — MPC controls u_ptc continuously [0,1].
    """

    def __init__(
            self,
            y,                          # T_cabin (Controlled)
            u,                          # u_ptc (Control)
            T_amb_feature,              # T_ambient (Disturbance) — for reading from df
            step_size: int,
            T_on: float = 323.15,       # Relay ON threshold: 50°C (was 35°C)
            T_off: float = 338.15,      # Relay OFF threshold: 65°C (was 45°C)
            T_margin: float = 1.0,      # Stop PTC this many K below setpoint
    ):
        self.y = y
        self.u = u
        self.T_amb_feature = T_amb_feature
        self.step_size = step_size
        self.T_on = T_on
        self.T_off = T_off
        self.T_margin = T_margin

        self.relay_on = False

    def __call__(self, df: pd.DataFrame) -> tuple[dict, dict]:
        """Calculate PTC relay output."""

        T_cabin = self.y.value if self.y.value is not None else 293.15
        T_setpoint = self.y.target if self.y.target is not None else 295.15

        # Read T_ambient from dataframe
        T_amb_col = self.T_amb_feature.source.col_name
        T_amb = float(df[T_amb_col].iloc[-1])

        # Read T_ptc from simulator (via dataframe)
        T_ptc = float(df['ptc_temperature'].iloc[-1]) if 'ptc_temperature' in df.columns else 293.15

        # Switch: heating needed?
        if T_amb >= T_setpoint:
            # Environment warm enough — no PTC needed
            u_ptc = 0.0
            self.relay_on = False
        elif T_cabin >= T_setpoint - self.T_margin:
            # Close to target — let HP fine-tune
            u_ptc = 0.0
            self.relay_on = False
        else:
            # Relay hysteresis on T_ptc from simulator
            if T_ptc > self.T_off:
                self.relay_on = False
            elif T_ptc < self.T_on:
                self.relay_on = True
            # else: maintain previous state

            u_ptc = 1.0 if self.relay_on else 0.0

        return {self.u.source.col_name: u_ptc}, {}

    def reset(self):
        self.relay_on = False


class DualModePID:
    """
    PID that automatically handles both heating and cooling.

    Uses the sign of the error to determine the mode:
    - error > 0 (T < target): heating mode
    - error < 0 (T > target): cooling mode
    """

    def __init__(
            self,
            y,
            u,
            step_size: int,
            Kp: float = 0.1,
            Ti: float = 100.0,
    ):
        self.y = y
        self.u = u
        self.step_size = step_size
        self.Kp = Kp
        self.Ti = Ti

        self.integral = 0.0

    def __call__(self, df: pd.DataFrame) -> tuple[dict, dict]:
        """Calculate control action based on absolute error."""

        # error = target - value
        # error > 0 means T < target (need heating)
        # error < 0 means T > target (need cooling)
        e = self.y.error if self.y.error is not None else 0.0

        # Use absolute error for control intensity
        # The simulator will determine heat/cool based on T vs target
        abs_e = abs(e)

        # P term
        P = self.Kp * abs_e

        # I term with anti-windup
        if self.Ti > 0:
            self.integral += (1 / self.Ti) * abs_e * self.step_size
            # Clamp integral
            max_i = self.u.ub / self.Kp if self.Kp > 0 else 10
            self.integral = min(self.integral, max_i)

        I = self.Kp * self.integral

        # Reduce integral when close to target (prevents overshoot)
        if abs_e < 0.5:  # Within 0.5K of target
            self.integral *= 0.9  # Decay integral

        output = P + I
        output = np.clip(output, self.u.lb, self.u.ub)

        return {self.u.source.col_name: output}, {}

    def reset(self):
        self.integral = 0.0
