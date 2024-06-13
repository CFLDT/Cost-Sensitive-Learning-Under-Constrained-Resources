import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def informativeness_plotter(target_metric, original_metric, actual_target_metric_value_array, inferred_target_metric_value_array,
                            tau):

    base_path = Path(__file__).parent
    plot_name = "Correlations_" + original_metric + "_" + target_metric + ".png"
    plot_path = (base_path / "../../plots/plots informativeness" / plot_name).resolve()

    fig, ax = plt.subplots()
    ax.scatter(actual_target_metric_value_array, inferred_target_metric_value_array)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    ax.set(aspect="equal", xlim=(0, 1), ylim=(0, 1),
           xlabel='Actual '+target_metric, ylabel='Inferred '+target_metric)
    ax.text(0.05, 0.95, str(tau))

    plt.savefig(plot_path)
    plt.close()
