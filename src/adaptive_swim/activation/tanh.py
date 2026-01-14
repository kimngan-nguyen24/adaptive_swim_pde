import numpy as np
import numpy.typing as npt

from .activations import Activation


class Tanh(Activation):
    name: str = "tanh"

    def __init__(self, max_abs: float | None = 10):
        self._max_abs = max_abs

    def _clip(self, x: npt.ArrayLike) -> npt.ArrayLike:
        if self._max_abs is not None:
            x = np.clip(x, -self._max_abs, self._max_abs)
        return x

    def _f(self, x: npt.ArrayLike, a_params: npt.ArrayLike) -> npt.ArrayLike:
        return np.tanh(x)

    def _dx(self, x: npt.ArrayLike, a_params: npt.ArrayLike, order: int = 0) -> npt.ArrayLike:
        x = self._clip(x)
        tanh = np.tanh(x)
        cosh = np.cosh(x)
        sinh = np.sinh(x)
        match order:
            case 1:
                return 1 / cosh**2
            case 2:
                return -2 * sinh / cosh**3
            case 3:
                return 4 * tanh**2 / cosh**2 - 2 * cosh**4
            case 4:
                return 16 * sinh / cosh**5 - 8 * sinh**3 / cosh**5
            case _:
                raise ValueError(
                    f"Derivative of order={order} " "is not implemented for 'tanh'."
                )
