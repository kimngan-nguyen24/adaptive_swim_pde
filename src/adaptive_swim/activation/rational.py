import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from .activations import Activation
from .activations import TorchActivation


import numpy as np
import numpy.typing as npt

from .activations import Activation


def _poly_eval_derivs_inner(
    x: np.ndarray,          # (N, M)
    coeff: np.ndarray,      # (M, deg+1)
    max_order: int,
) -> list[np.ndarray]:
    """
    Evaluate polynomial and its x-derivatives up to max_order column-wise.

    For each inner dimension m:
        P_m(x) = sum_i coeff[m,i] * x^i

    Returns [P, P', ..., P^(max_order)], each of shape (N, M).
    """
    x = np.asarray(x)
    coeff = np.asarray(coeff)

    N, M = x.shape
    deg = coeff.shape[1] - 1

    derivs: list[np.ndarray] = []

    # Order 0
    exps = np.arange(deg + 1, dtype=x.dtype)                  # (deg+1,)
    x_pow = x[:, :, None] ** exps[None, None, :]              # (N, M, deg+1)
    d0 = np.sum(x_pow * coeff[None, :, :], axis=2)            # (N, M)
    derivs.append(d0)

    # Higher orders
    for k in range(1, max_order + 1):
        if deg < k:
            derivs.append(np.zeros_like(d0))
            continue

        i = np.arange(k, deg + 1, dtype=x.dtype)              # i = k..deg (len = deg+1-k)

        # falling factorial: i*(i-1)*...*(i-k+1)
        fall = np.ones_like(i)
        for t in range(k):
            fall *= (i - t)

        coeff_k = coeff[:, k:] * fall[None, :]                # (M, deg+1-k)
        x_pow_k = x[:, :, None] ** (i - k)[None, None, :]     # (N, M, deg+1-k)
        dk = np.sum(x_pow_k * coeff_k[None, :, :], axis=2)     # (N, M)
        derivs.append(dk)

    return derivs


def _reciprocal_derivs(u_derivs: list[np.ndarray], max_order: int) -> list[np.ndarray]:
    """
    Given u, u', ..., u^(max_order), compute v=1/u and v-derivatives via:

      v^(n) = -(1/u) * sum_{k=1..n} binom(n,k) * u^(k) * v^(n-k)

    All arrays are same shape (N, M).
    """
    u0 = u_derivs[0]
    v_derivs: list[np.ndarray] = [1.0 / u0]

    binom = {
        1: (1, 1),
        2: (1, 2, 1),
        3: (1, 3, 3, 1),
        4: (1, 4, 6, 4, 1),
    }

    for n in range(1, max_order + 1):
        s = np.zeros_like(u0)
        b = binom[n]
        for k in range(1, n + 1):
            s += b[k] * u_derivs[k] * v_derivs[n - k]
        v_n = -(v_derivs[0] * s)
        v_derivs.append(v_n)

    return v_derivs


class Rational(Activation):
    """
    Column-wise rational activation:

        f(X) = P(X) / (1 + Q(X)^2),

    where for each inner dimension m:
        P_m(x) = sum_{i=0..p-1} a_{m,i} x^i
        Q_m(x) = sum_{j=0..q-1} b_{m,j} x^j

    Shapes:
      X:       (N, n_inner)
      a_params:(n_inner, p+q) or (1, n_inner, p+q)
      output:  (N, n_inner)
    """
    name: str = "rational"

    def __init__(self, num_coeff_p: int = 4, num_coeff_q: int = 3):
        self.num_coeff_p = int(num_coeff_p)
        self.num_coeff_q = int(num_coeff_q)

    def _normalize_params(self, a_params: npt.ArrayLike) -> np.ndarray:
        """
        Accept:
          - (M, p+q)
          - (1, M, p+q)
        Return:
          - (M, p+q)
        """
        a_params = np.asarray(a_params)
        p, q = self.num_coeff_p, self.num_coeff_q
        pq = p + q

        if a_params.ndim == 3 and a_params.shape[0] == 1:
            a_params = a_params[0]

        if a_params.ndim != 2 or a_params.shape[1] != pq:
            raise ValueError(f"a_params must have shape (n_inner, {pq}) or (1, n_inner, {pq}), got {a_params.shape}")

        return a_params

    def _split(self, a_params: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        a_params = self._normalize_params(a_params)
        p = self.num_coeff_p
        return a_params[:, :p], a_params[:, p:]

    def _f(self, x: npt.ArrayLike, a_params: npt.ArrayLike) -> np.ndarray:
        x = np.asarray(x)  # (N, M)
        coeff_p, coeff_q = self._split(a_params)  # (M,p), (M,q)

        P = _poly_eval_derivs_inner(x, coeff_p, max_order=0)[0]  # (N,M)
        Q = _poly_eval_derivs_inner(x, coeff_q, max_order=0)[0]  # (N,M)
        D = 1.0 + Q**2
        return P / D  # (N,M)

    def _grad(self, x: npt.ArrayLike, a_params: npt.ArrayLike) -> np.ndarray:
        """
        Gradient w.r.t. a_params, returned as shape (N, M, p+q).

        For each inner dim m:
          f = P / D, D = 1 + Q^2

          ∂f/∂a_i = x^i / D
          ∂f/∂b_j = - P * (2 Q x^j) / D^2
        """
        x = np.asarray(x)  # (N,M)
        coeff_p, coeff_q = self._split(a_params)
        p, q = self.num_coeff_p, self.num_coeff_q

        P = _poly_eval_derivs_inner(x, coeff_p, max_order=0)[0]  # (N,M)
        Q = _poly_eval_derivs_inner(x, coeff_q, max_order=0)[0]  # (N,M)
        D = 1.0 + Q**2
        invD = 1.0 / D
        invD2 = invD**2

        # Powers of x for numerator params: (N,M,p)
        xp = np.stack([x**i for i in range(p)], axis=2)
        grad_p = xp * invD[:, :, None]  # (N,M,p)

        # Powers of x for denominator params: (N,M,q)
        xq = np.stack([x**j for j in range(q)], axis=2)
        grad_q = -(P[:, :, None] * (2.0 * Q[:, :, None] * xq) * invD2[:, :, None])  # (N,M,q)

        return np.concatenate([grad_p, grad_q], axis=2)  # (N,M,p+q)

    def _dx(self, x: npt.ArrayLike, a_params: npt.ArrayLike, order: int = 0) -> np.ndarray:
        """
        x-derivatives up to 4th order, shape (N, M).
        """
        if order not in (1, 2, 3, 4):
            raise ValueError(f"Derivative of order={order} is not implemented for 'rational'.")

        x = np.asarray(x)  # (N,M)
        coeff_p, coeff_q = self._split(a_params)

        # P^(k), Q^(k) for k=0..4 (each (N,M))
        P_der = _poly_eval_derivs_inner(x, coeff_p, max_order=4)
        Q_der = _poly_eval_derivs_inner(x, coeff_q, max_order=4)
        Q0, Q1, Q2, Q3, Q4 = Q_der

        # D = 1 + Q^2 and derivatives
        D0 = 1.0 + Q0**2
        D1 = 2.0 * Q0 * Q1
        D2 = 2.0 * (Q1**2 + Q0 * Q2)
        D3 = 2.0 * (3.0 * Q1 * Q2 + Q0 * Q3)
        D4 = 2.0 * (3.0 * Q2**2 + 4.0 * Q1 * Q3 + Q0 * Q4)
        D_der = [D0, D1, D2, D3, D4]

        # invD and derivatives
        invD_der = _reciprocal_derivs(D_der, max_order=4)

        # Product rule for f = P * invD
        binom = {
            0: (1,),
            1: (1, 1),
            2: (1, 2, 1),
            3: (1, 3, 3, 1),
            4: (1, 4, 6, 4, 1),
        }

        n = order
        b = binom[n]
        out = np.zeros_like(P_der[0])
        for k in range(0, n + 1):
            out += b[k] * P_der[k] * invD_der[n - k]
        return out


class Torch_Rational(TorchActivation):
    num_coeff_p: int
    num_coeff_q: int

    def __init__(self, n_params, num_coeff_p = 4, num_coeff_q = 3):
        # Automatically set the parent's num_a_params
        super().__init__()
        self.num_coeff_p = num_coeff_p
        self.num_coeff_q = num_coeff_q
        self.a_params = nn.Parameter(torch.ones(n_params, num_coeff_p + num_coeff_q, dtype=torch.float64)) # shape (N, p + q)

    def forward(self, x) -> torch.Tensor:
        # x: (N, d), self.a_params: (N, p + q), where N is the batch size
        # Split numerator / denominator coefficients
        coeff_p = self.a_params[:, :self.num_coeff_p]     # shape (N, p)
        coeff_q = self.a_params[:, self.num_coeff_p:]     # shape (N, q)

        # Construct numerator: power matrix of x multiplied linearly with coeff_p
        x_powers_p = torch.stack(
            [x ** i for i in range(self.num_coeff_p)],
            dim=2
        )  # shape (N, d, p)

        numerator = torch.bmm(
            x_powers_p, coeff_p.unsqueeze(2)
        ).squeeze(2)
        # (N, d, p) bmm (N, p, 1) = (N, d, 1)
        # squeeze -> (N, d)

        # Construct denominator: generate different denominator forms based on pole information
        x_powers_q = torch.stack(
            [x ** i for i in range(self.num_coeff_q)],
            dim=2
        )  # shape (N, d, q)

        denominator = 1 + (
            torch.bmm(x_powers_q, coeff_q.unsqueeze(2)).squeeze(2)
        ) ** 2  # (N, d)

        res = numerator / denominator

        if res.shape[0] == 1:
            res = res.squeeze(0)  # (d,)

        return res  # (N, d)
