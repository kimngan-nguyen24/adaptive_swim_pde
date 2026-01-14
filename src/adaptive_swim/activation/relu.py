import numpy as np
import numpy.typing as npt

from .activations import Activation


class ReLU(Activation):
    name: str = "relu"

    def _f(self, x: npt.ArrayLike, a_params: npt.ArrayLike) -> npt.ArrayLike:
        return np.maximum(x, 0)

    def _dx(self, x: npt.ArrayLike, a_params: npt.ArrayLike, order: int = 0) -> npt.ArrayLike:
        if order == 0:
            return np.where(x < 0.0, 0.0, 1.0)
        else:
            return np.zeros_like(x)
