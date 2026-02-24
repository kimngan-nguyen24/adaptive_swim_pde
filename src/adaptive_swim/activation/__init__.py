from .activations import Activation
from .activations import TorchActivation
from .cos import Cos
from .relu import ReLU
from .sin import Sin
from .tanh import Tanh
from .a_tanh import Adaptive_Tanh
from .a_tanh import Torch_Adaptive_Tanh
from .rational import Rational
from .rational import Torch_Rational


def get_activation(activation_name: str) -> Activation:
    """Returns an object for the activation function."""
    activations = {
        "tanh": Tanh(),
        "relu": ReLU(),
        "cos": Cos(),
        "sin": Sin(),
        "a_tanh": Adaptive_Tanh(),
        "rational": Rational()}
    if activation_name not in activations:
        raise ValueError(f"Unknown activation {activation_name}.")
    return activations[activation_name]


def get_parameter_sampler(activation_name: str) -> str:
    """Returns a parameter sampler for the activation function."""
    parameter_samplers = {
        "tanh": "tanh",
        "relu": "relu",
        "cos": "tanh",
        "sin": "tanh",
        "a_tanh": "tanh",
        "rational": "rational",
    }
    if activation_name not in parameter_samplers:
        raise ValueError(f"Unknown activation {activation_name}.")

    return parameter_samplers[activation_name]

def get_torch_activation(activation_name: str) -> type[TorchActivation] | None:
    """Returns an object for the PyTorch activation function."""
    activations = {
        "a_tanh": Torch_Adaptive_Tanh,
        "rational": Torch_Rational
    }
    if activation_name not in activations:
        return None
    return activations[activation_name]


__all__ = [
    "Activation",
    "TorchActivation",
    "Cos",
    "Sin",
    "Tanh",
    "ReLU",
    "Adaptive_Tanh",
    "Torch_Adaptive_Tanh",
    "Rational",
    "Torch_Rational",
    "get_activation",
    "get_parameter_sampler",
]
