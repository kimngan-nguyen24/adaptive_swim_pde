from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from swimpde.abstract import Ansatz
from swimpde.domain import Domain

from adaptive_swim.activation import Activation, TorchActivation, get_activation, get_parameter_sampler, get_torch_activation
from adaptive_swim.swim_backbones import Dense

from .utils import get_order


@dataclass(kw_only=True)
class BasicAnsatz(Ansatz):
    """Ansatz representing a simple neural network.

    Attributes:
    -----------
    n_basis: int
        Number of outer basis functions.
    n_inner_basis: int
        Number of inner basis functions.
    activation: Union[str, Activation]
        Activation function to use for the inner basis.
    parameter_sampler: str
        Parameter sampler to use in the SWIM algorithm.
        See the SWIM package for possible options)
    random_seed: int
        Random seed to be used in all processes that require randomness.

    """

    n_basis: int
    activation: str | Activation
    parameter_sampler: str | Callable | None = None
    random_seed: int = 1
    torch_activation_cls: str | type[TorchActivation] | None = None

    _layer: Dense | None = None

    def __post_init__(self):
        if isinstance(self.activation, str):
            self.activation = get_activation(self.activation)
        elif not isinstance(self.activation, Activation):
            raise TypeError(
                "Activation function must be of type str or "
                f"Activation: got {type(self.activation)}"
            )
        if self.parameter_sampler is None:
            self.parameter_sampler = get_parameter_sampler(self.activation.name)

        if self.torch_activation_cls is None:
            self.torch_activation_cls = get_torch_activation(self.activation.name)
        elif isinstance(self.torch_activation_cls, str):
            self.torch_activation_cls = get_torch_activation(self.torch_activation_cls)

        self._layer = Dense(
            layer_width=self.n_basis,
            activation=self.activation.get_f(),
            torch_activation_cls=self.torch_activation_cls,
            parameter_sampler=self.parameter_sampler,
            random_seed=self.random_seed,
            prune_duplicates=False,
        )

        self.n_outputs = self.n_basis

    def _transform(
        self,
        x: npt.ArrayLike,
        operator: str | None = None,
        coordinate_scaling: npt.ArrayLike | None = None,
    ) -> npt.ArrayLike:
        order = get_order(operator)
        self._layer.activation = self.activation.get_f(order=order)
        layer_output = self._layer.transform(x)  # shape (n_points, n_basis)
        layer_weights = (
            self._layer.weights * coordinate_scaling[:, None]
        )  # shape (d, n_basis)

        match operator:
            case None:
                return layer_output
            case "gradient":
                return layer_output[..., None] * layer_weights.T
            case "laplace":
                weights_norm = np.linalg.norm(layer_weights, axis=0) ** 2
                return layer_output * weights_norm
            case "dxxxx":
                weights_norm = np.linalg.norm(layer_weights, axis=0) ** 4
                return layer_output * weights_norm
            case _:
                raise ValueError(f"Cannot evaluate basic ansatz for {operator=}.")

    def _fit(
        self,
        domain: Domain,
        target_fn: Callable[[npt.ArrayLike], npt.ArrayLike]
        | npt.ArrayLike
        | None = None,
        operator: str | None = None,
    ):
        """
        Fit the model to the data.

        Parameters:
        x: input values of shape (n_points, d)
        y: target values of shape (n_points,)
        order: order of the activation's derivative to apply before fitting.
        """
        if operator not in [None, "laplace"]:
            raise ValueError(
                "BasicAnsatz.fit() only supports opertors "
                f"in [None, 'laplace']. Got: {operator}."
            )

        if target_fn is None:
            target = np.zeros((domain.interior_points.shape[0], self.n_basis))
        elif callable(target_fn):
            target = target_fn(domain.interior_points)
        else:
            target = target_fn

        if target.shape[0] != domain.interior_points.shape[0]:
            raise ValueError(
                f"Target has shape {target.shape}, but "
                f"points have shape {domain.interior_points.shape}."
            )

        order = get_order(operator)
        self._layer.activation = self.activation.get_f(order=order)
        self._layer.fit(domain.interior_points, target)
