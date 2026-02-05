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
    Blower fan speed controller with exponential smoothing.

    Proportional response to abs(T_target - T_cabin) with a minimum
    ventilation floor. Exponential smoothing prevents abrupt changes.

    - Large error (transient): blower ramps to 1.0
    - Small error (steady state): blower settles at min_vent
    - Passenger disturbance: error rises → blower ramps up → more fresh air
    """

    def __init__(
            self,
            y,                      # Controlled feature (T_cabin)
            u,                      # Control feature (u_blower)
            step_size: int,
            Kp: float = 1.0,       # Proportional gain on abs(error)
            tau: float = 300.0,     # Smoothing time constant [s]
            min_vent: float = 0.4,  # Minimum ventilation floor [-]
    ):
        self.y = y
        self.u = u
        self.step_size = step_size
        self.Kp = Kp
        self.tau = tau
        self.min_vent = min_vent

        self.output = u.default if hasattr(u, 'default') else 0.5

    def __call__(self, df: pd.DataFrame) -> tuple[dict, dict]:
        """Calculate blower control action."""

        # Absolute temperature error — always positive
        e = abs(self.y.error) if self.y.error is not None else 0.0

        # Target: proportional + minimum ventilation floor
        target = self.Kp * e + self.min_vent
        target = np.clip(target, self.u.lb, self.u.ub)

        # Exponential smoothing (first-order lag)
        alpha = self.step_size / self.tau
        self.output = alpha * target + (1 - alpha) * self.output
        self.output = np.clip(self.output, self.u.lb, self.u.ub)

        return {self.u.source.col_name: self.output}, {}

    def reset(self):
        self.output = self.u.default if hasattr(self.u, 'default') else 0.5


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
