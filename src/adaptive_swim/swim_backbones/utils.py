import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

def compute_interpolation_points(
    x_start: NDArray,
    x_end: NDArray,
    k: int,
) -> NDArray:
    """
    Computes k evenly spaced interpolation points between x_start and x_end.

    Args:
        x_start: ndarray of shape (m, d)
        x_end: ndarray of shape (m, d)
        k: number of interpolation points

    Returns:
        ndarray of shape (m, k, d)
    """
    # alpha: (1, k, 1)
    alpha = np.linspace(0.0, 1.0, num=k, dtype=x_start.dtype)[None, :, None]

    # (m, 1, d)
    x_start_exp = x_start[:, None, :]
    x_end_exp = x_end[:, None, :]

    return x_start_exp + alpha * (x_end_exp - x_start_exp)



def find_nearest_indices(
    x_inter: NDArray,
    x_real: NDArray,
) -> NDArray:
    """
    For each point in x_inter (m, k, d), find the index of the nearest point
    in x_real (N, d).

    Args:
        x_inter: ndarray of shape (m, k, d)
        x_real: ndarray of shape (N, d)

    Returns:
        ndarray of shape (m, k) with nearest-neighbor indices
    """
    m, k, d = x_inter.shape

    # Flatten to (m * k, d)
    inter_flat = x_inter.reshape(-1, d)

    tree = KDTree(x_real)
    _, indices = tree.query(inter_flat)

    return indices.reshape(m, k)
