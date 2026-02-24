import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from .activations import Activation
from .activations import TorchActivation


class Adaptive_Tanh(Activation):
    name: str = "a_tanh"

    def __init__(self, max_abs: float | None = 10):
        self._max_abs = max_abs

    def _clip(self, x: npt.ArrayLike) -> npt.ArrayLike:
        if self._max_abs is not None:
            x = np.clip(x, -self._max_abs, self._max_abs)
        return x

    def _f(self, x: npt.ArrayLike, a_params: npt.ArrayLike) -> npt.ArrayLike:
        return np.tanh(a_params * x)

    def _dx(self, x: npt.ArrayLike, a_params: npt.ArrayLike, order: int = 0) -> npt.ArrayLike:
        x = self._clip(a_params * x)
        tanh = np.tanh(x)
        cosh = np.cosh(x)
        sinh = np.sinh(x)
        match order:
            case 1:
                return 1 / cosh**2 * a_params
            case 2:
                return -2 * sinh / cosh**3 * a_params**2
            case 3:
                return (4 * tanh**2 / cosh**2 - 2 * cosh**4) * a_params**3
            case 4:
                return (16 * sinh / cosh**5 - 8 * sinh**3 / cosh**5) * a_params**4
            case _:
                raise ValueError(
                    f"Derivative of order={order} " "is not implemented for 'tanh'."
                )


class Torch_Adaptive_Tanh(TorchActivation):
    name = "a_tanh"

    def __init__(self, n_params):
        super().__init__()
        self.a_params = nn.Parameter(torch.ones(n_params, 1))

    def forward(self, x):
        #print(x.shape, self.a_params.shape)
        return torch.tanh(self.a_params * x)