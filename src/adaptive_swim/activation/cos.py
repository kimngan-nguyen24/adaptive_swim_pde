import numpy as np
import numpy.typing as npt

from .activations import Activation


class Cos(Activation):
    name: str = "cos"

    def __init__(self):
        super().__init__()

    def _f(self, x: npt.ArrayLike, a_params: npt.ArrayLike) -> npt.ArrayLike:
        return np.cos(x)

    def _dx(self, x: npt.ArrayLike, a_params: npt.ArrayLike, order: int) -> npt.ArrayLike:
        match (order + 1) % 4:
            case 1:
                return -np.sin(x)
            case 2:
                return -np.cos(x)
            case 3:
                return np.sin(x)
            case 0:
                return np.cos(x)