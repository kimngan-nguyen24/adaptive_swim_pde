import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def compute_metrics(
    u_pred: npt.ArrayLike, u_exact: npt.ArrayLike
) -> tuple[npt.ArrayLike, float]:
    if u_pred.shape != u_exact.shape:
        raise ValueError(f"Cannot compute metrics: {u_pred.shape=}, {u_exact.shape=}.")

    abs = np.abs(u_pred - u_exact)
    rel_l2 = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    return abs, rel_l2


def plot_results(
    x_eval: npt.ArrayLike,
    t_eval: npt.ArrayLike,
    u_exact: npt.ArrayLike,
    u_pred: npt.ArrayLike,
    abs_error: npt.ArrayLike,
) -> plt.Figure:
    fig, (axes_full, axes_samples) = plt.subplots(2, 3, figsize=(18, 8))

    # Plot solutions for all t.
    data_dict = {
        "Ground truth": u_exact,
        "Predictions": u_pred,
        "Absolute error": abs_error,
    }
    extent = [x_eval.min(), x_eval.max(), t_eval.min(), t_eval.max()]
    for ax, (label, data) in zip(axes_full, data_dict.items()):
        im = ax.imshow(data.T, extent=extent, origin="lower")
        fig.colorbar(im, ax=ax, location="bottom")
        ax.set_title(label)

    # Plot solutions and for certain values of t.
    for i, ax in enumerate(axes_samples):
        idx = i * t_eval.shape[0] // 3
        ax.plot(x_eval, u_exact[:, idx], "b-", label="Ground truth")
        ax.plot(x_eval, u_pred[:, idx], "r--", label="Predictions")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u(t,x)$")
        ax.set_aspect("equal")
        ax.set_title(f"$t = {t_eval[idx]:.3f}$")

    return fig
