import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def dec_boundary_plotter(name, method, X, y, xx, yy, grid_probs):


    base_path = Path(__file__).parent
    plot_name_prob = "prob_boundary_" + name + "_" + method + ".png"
    plot_path_prob = (base_path / "../../plots/plots decision boundaries" / plot_name_prob).resolve()

    plot_name_prob_2 = "data_plotter" + name + "_" + ".png"
    plot_path_prob_2 = (base_path / "../../plots/plots decision boundaries" / plot_name_prob_2).resolve()

    grid_probs = grid_probs.reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))

    contour = ax.contourf(xx, yy, grid_probs, 25, cmap="coolwarm_r",
                          vmin=np.min(grid_probs), vmax=np.max(grid_probs))
    ax_c = f.colorbar(contour)
    ax_c.set_label("Output")
    ax_c.set_ticks(np.linspace(np.min(grid_probs), np.max(grid_probs), 4))

    ax.scatter(X[:, 0],  X[:, 1], c=y, s=50,
               cmap="coolwarm_r", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(np.min(xx), np.max(xx)), ylim=(np.min(yy), np.max(yy)),
           xlabel="$X_1$", ylabel="$X_2$")

    plt.savefig(plot_path_prob)
    plt.close()


    f, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(X[:, 0], X[:, 1], c=y, s=50,
               cmap="coolwarm_r", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(np.min(xx), np.max(xx)), ylim=(np.min(yy), np.max(yy)),
           xlabel="$X_1$", ylabel="$X_2$")

    plt.savefig(plot_path_prob_2)
    plt.close()




