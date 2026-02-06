from abc import ABC, abstractmethod
from typing import Union

from casadi import MX, fmax, exp, log


class Cost(ABC):
    """ cost function for objective """

    def __init__(
            self,
            weight: float,
    ):
        """ scalar weight for the cost function """

        self.weight = weight

    def __str__(self):
        return f'CostOrder({self.__class__.__name__})'

    def __repr__(self):
        return f'CostOrder({self.__class__.__name__})'

    @abstractmethod
    def __call__(self, mx: Union[MX, float]) -> Union[MX, float]:
        """ the call function takes a casadi MX variable and applies the cost function to it """
        pass


class Linear(Cost):
    """ linear cost function """

    def __init__(
            self,
            weight: float = 1,
            offset: float = 0,
            norm:   float = 1,
    ):
        super(Linear, self).__init__(
            weight=weight,
        )

        # the offset can be used to shift the costs by a constant factor
        self.offset = offset
        self.norm = norm

    def __call__(self, mx: Union[MX, float]) -> Union[MX, float]:
        """ the call function takes a casadi MX variable and applies the cost function to it """

        return (mx - self.offset) / self.norm * self.weight

    def __str__(self):
        return f'{self.__class__.__name__}'


class AbsoluteLinear(Cost):
    """ absolute linear cost function """

    def __init__(
            self,
            weight: float = 1,
    ):
        super(AbsoluteLinear, self).__init__(
            weight=weight,
        )

    def __call__(self, mx: Union[MX, float]) -> Union[MX, float]:
        """ the call function takes a casadi MX variable and applies the cost function to it """

        return mx * self.weight

    def __str__(self):
        return f'{self.__class__.__name__}'


class Quadratic(Cost):

    def __init__(
            self,
            weight: float = 1,
            norm: float = 1,
            offset: float = 0,
    ):

        super(Quadratic, self).__init__(
            weight=weight,
        )

        # before squaring the feature is divided by the norm
        self.norm = norm
        self.offset = offset

    def __call__(self, mx: Union[MX, float]) -> Union[MX, float]:
        """ the call function takes a casadi MX variable and applies the cost function to it """

        return ((mx - self.offset) / self.norm) ** 2 * self.weight

    def __str__(self):
        return f'{self.__class__.__name__}'


def _softplus(x, sharpness: float = 10.0):
    """
    Smooth approximation to max(0, x): softplus(x) = log(1 + exp(sharpness * x)) / sharpness

    For large sharpness, approaches max(0, x).
    Sharpness=10 gives good balance between smoothness and accuracy.
    """
    return log(1 + exp(sharpness * x)) / sharpness


class BandViolation(Cost):
    """
    Quadratic penalty for values outside a band [lb, ub].
    No penalty inside the band, quadratic penalty outside.

    Uses smooth softplus approximation for IPOPT compatibility.

    J = weight * (softplus(lb - x)² + softplus(x - ub)²)
    """

    def __init__(
            self,
            weight: float = 1,
            lb: float = 0,
            ub: float = 1,
            norm: float = 1,
            sharpness: float = 10.0,
    ):
        super(BandViolation, self).__init__(weight=weight)
        self.lb = lb
        self.ub = ub
        self.norm = norm
        self.sharpness = sharpness

    def __call__(self, mx: Union[MX, float]) -> Union[MX, float]:
        """ Quadratic penalty for band violation (smooth) """
        # Violation below lower bound (positive when below)
        below = _softplus((self.lb - mx) / self.norm, self.sharpness)
        # Violation above upper bound (positive when above)
        above = _softplus((mx - self.ub) / self.norm, self.sharpness)
        return (below ** 2 + above ** 2) * self.weight

    def __str__(self):
        return f'{self.__class__.__name__}(lb={self.lb}, ub={self.ub})'


class SoftUpperBound(Cost):
    """
    Quadratic penalty for values above a threshold.
    No penalty below threshold, quadratic penalty above.

    Uses smooth softplus approximation for IPOPT compatibility.

    J = weight * softplus(x - threshold)²
    """

    def __init__(
            self,
            weight: float = 1,
            threshold: float = 0,
            norm: float = 1,
            sharpness: float = 10.0,
    ):
        super(SoftUpperBound, self).__init__(weight=weight)
        self.threshold = threshold
        self.norm = norm
        self.sharpness = sharpness

    def __call__(self, mx: Union[MX, float]) -> Union[MX, float]:
        """ Quadratic penalty for exceeding threshold (smooth) """
        violation = _softplus((mx - self.threshold) / self.norm, self.sharpness)
        return violation ** 2 * self.weight

    def __str__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'


class SoftLowerBound(Cost):
    """
    Quadratic penalty for values below a threshold.
    No penalty above threshold, quadratic penalty below.

    Uses smooth softplus approximation for IPOPT compatibility.

    J = weight * softplus(threshold - x)²
    """

    def __init__(
            self,
            weight: float = 1,
            threshold: float = 0,
            norm: float = 1,
            sharpness: float = 10.0,
    ):
        super(SoftLowerBound, self).__init__(weight=weight)
        self.threshold = threshold
        self.norm = norm
        self.sharpness = sharpness

    def __call__(self, mx: Union[MX, float]) -> Union[MX, float]:
        """ Quadratic penalty for going below threshold (smooth) """
        violation = _softplus((self.threshold - mx) / self.norm, self.sharpness)
        return violation ** 2 * self.weight

    def __str__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'

