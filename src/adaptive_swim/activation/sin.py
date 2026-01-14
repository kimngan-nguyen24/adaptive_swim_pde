import numpy as np
import numpy.typing as npt

from .activations import Activation


class Sin(Activation):
    name: str = "sin"

    def _f(self, x: npt.ArrayLike, a_params: npt.ArrayLike) -> npt.ArrayLike:
        return np.sin(x)

    def _dx(self, x: npt.ArrayLike, a_params: npt.ArrayLike, order: int = 0) -> npt.ArrayLike:
        match (order + 1) % 4:
            case 1:
                return np.cos(x)
            case 2:
                return -np.sin(x)
            case 3:
                return -np.cos(x)
            case 0:
                return np.sin(x)
