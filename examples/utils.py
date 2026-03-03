import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas
import matplotlib.ticker as ticker

from scipy.stats.qmc import LatinHypercube
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plots
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def compute_metrics(
    u_pred: npt.ArrayLike, u_exact: npt.ArrayLike
) -> tuple[npt.ArrayLike, float]:
    if u_pred.shape != u_exact.shape:
        raise ValueError(f"Cannot compute metrics: {u_pred.shape=}, {u_exact.shape=}.")

    abs = np.abs(u_pred - u_exact)
    rel_l2 = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    return abs, rel_l2


def filter_results(df: pandas.DataFrame, activation, n_inner_basis):
    filtered = df[df["activation"] == activation]
    rmse = []
    rel_l2 = []
    for n in n_inner_basis:
        subset = filtered[filtered["n_inner"] == n]
        # Minimum
        min_rmse = subset["rmse"].min()
        min_rel_l2 = subset["rel_l2"].min()
        rmse.append(min_rmse)
        rel_l2.append(min_rel_l2)
    return np.array(rmse), np.array(rel_l2)


def compute_relative_l2_error(u_true, u_pred):
    """
    Compute the relative L2 error between true and predicted solutions.

    Parameters:
        u_true (ndarray): Ground truth solution.
        u_pred (ndarray): Predicted solution (flattened).

    Returns:
        float: Relative L2 error.
    """
    return np.linalg.norm(u_true - u_pred.flatten()[:, None], 2) / np.linalg.norm(u_true, 2)


def plot_spatiotemporal_solutions(u_true, u_model, x_space, filename="burgers_1.pdf", title='Frozen-PINN-swim'):
    """
    Plot the ground truth, model prediction, and absolute error across space and time.

    Parameters:
        u_true (ndarray): Ground truth solution.
        u_model (ndarray): Model-predicted solution.
        x_space (ndarray): Spatial grid.
        filename (str): Output filename for the figure.
    """
    fontsize = 14
    fig, ax = plt.subplots(1, 3, figsize=(7, 3), constrained_layout=True)
    extent = [0, 1, np.min(x_space), np.max(x_space)]
    aspect = 0.3

    sol_img1 = ax[0].imshow(u_true.T, extent=extent, origin='lower', aspect=aspect)
    sol_img2 = ax[1].imshow(u_model.T, extent=extent, origin='lower', aspect=aspect)
    error_img = ax[2].imshow(np.abs(u_model - u_true).T, extent=extent, origin='lower', aspect=aspect)

    for a in ax:
        a.set_xlabel('t', fontsize=fontsize)
    ax[0].set_ylabel('x', fontsize=fontsize)

    for line_pos in [0.25, 0.5, 0.75]:
        ax[0].axvline(x=line_pos, color='k', linestyle='--', linewidth=2)

    sampling_times = np.linspace(0, 1, 10)[1:-1]
    for s_t in sampling_times:
        ax[1].axvline(x=s_t, color='gray', linestyle='dotted', linewidth=3)

    add_colorbar(fig, sol_img1, ax[0], label="Ground truth", scientific=True)
    add_colorbar(fig, sol_img2, ax[1], label="Prediction", scientific=True)
    add_colorbar(fig, error_img, ax[2], label="Error", scientific=False)

    ax[0].set_title('Ground truth', fontsize=fontsize)
    ax[1].set_title(title, fontsize=fontsize)
    ax[2].set_title('Absolute error', fontsize=fontsize)

    fig.savefig(filename)


def add_colorbar(fig, img, ax, label="", scientific=False):
    """
    Add a colorbar to a subplot with optional scientific notation formatting.

    Parameters:
        fig (Figure): Matplotlib figure object.
        img (AxesImage): The image to attach the colorbar to.
        ax (Axes): Axis to place the colorbar under.
        label (str): Optional label for the colorbar.
        scientific (bool): Whether to use scientific notation.
    """
    if scientific:
        formatter = ScalarFormatter()
        formatter.set_scientific(True)
        formatter.set_useMathText(False)
        formatter.set_powerlimits((-8, 8))
        cbar = fig.colorbar(img, ax=ax, location='bottom', format=formatter, fraction=0.049)
    else:
        cbar = fig.colorbar(img, ax=ax, location='bottom', format='%.0e', fraction=0.049)

    cbar.locator = ticker.MaxNLocator(nbins=2)
    cbar.update_ticks()


def plot_temporal_slices(u_true, u_model, x_eval, filename="burgers_2.pdf"):
    """
    Plot solution slices at t = 0.25, 0.50, and 0.75 comparing model and ground truth.

    Parameters:
        u_true (ndarray): Ground truth solution.
        u_model (ndarray): Predicted solution.
        x_eval (ndarray): Evaluation grid in space.
        filename (str): Output filename for the figure.
    """
    fontsize = 14
    fig, ax = plt.subplots(1, 3, figsize=(7, 3), constrained_layout=True)

    time_indices = [25, 50, 75]
    time_labels = [0.25, 0.50, 0.75]

    for i, (t_idx, t_label) in enumerate(zip(time_indices, time_labels)):
        ax[i].plot(x_eval, u_true[t_idx, :], 'b-', linewidth=2, label='Ground truth')
        ax[i].plot(x_eval, u_model[t_idx, :], 'r--', linewidth=2, label='Frozen-PINN-swim (resampling)')
        ax[i].set_xlabel('$x$', fontsize=fontsize)
        ax[i].set_ylabel('$u(t,x)$', fontsize=fontsize)
        ax[i].set_title(f'$t = {t_label}$', fontsize=fontsize)
        ax[i].axis('square')
        ax[i].set_xlim([-1.1, 1.1])
        ax[i].set_ylim([-1.1, 1.1])

    fig.legend(*ax[1].get_legend_handles_labels(), loc='upper center', ncol=2, fontsize=fontsize, frameon=False)
    fig.savefig(filename, bbox_inches='tight')