from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import numpy.typing as npt
import torch
import torch.nn as nn


class Activation(ABC):
    """A base class for differentiable activation function.

    Attributes:
    -----------
    name: str
        Name of the activation function.


    Methods:
    --------
    get_f(self, order: int = 0) -> Callable[[npt.ArrayLike], npt.ArrayLike]:
        Returns a function for the derivative of order 'order'.
    __call__(self, x: npt.NDArray, order: int=0) -> npt.NDArray:
        Evaluates the activation's derivative of order 'order'.
    _f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        An abstract method for evaluating the activation function.
    _dx(self, x: npt.NDArray, order: int) -> npt.NDArray:
        An abstract methods for evaluating activation's derivative.
    """

    name: str = "base"

    def __call__(self, x: npt.ArrayLike, a_params: npt.ArrayLike, order: int = 0) -> npt.ArrayLike:
        """Evaluates the activation's derivative of order 'order'.

        Evalutes the activation function itself when order is set to 0.
        """
        f = self.get_f(order)
        return f(x, a_params)

    def get_f(self, order: int = 0) -> Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
        if order < 0 or not isinstance(order, int):
            raise ValueError(
                "The order must be a non-negative integer. " f"Got {order} instead."
            )

        if order == 0:
            return self._f

        return partial(self._dx, order=order)

    @abstractmethod
    def _f(self, x: npt.ArrayLike, a_params: npt.ArrayLike) -> npt.ArrayLike:
        """Evaluates the activation function."""
        pass

    @abstractmethod
    def _dx(self, x: npt.ArrayLike, a_params: npt.ArrayLike, order: int = 1) -> npt.ArrayLike:
        """Evaluates the derivative of order 'order'."""
        pass


class TorchActivation(nn.Module, ABC):
    """A base class for (adaptive) activation function in PyTorch.

    Attributes:
    -----------
    name: str
        Name of the activation function.


    Methods:
    --------
    forward(self, x: torch.Tensor) -> torch.Tensor:
        Evaluates the activation function.
    """

    name: str = "base"

    def __init__(self, n_params):
        super().__init__()
        self.a_params = nn.Parameter(torch.ones(n_params, 1))

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass