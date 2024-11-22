# rmt.simulations.entropy_evolution.py
"""
This module contains the programs for performing the Monte Carlo
simulations of the evolution of entropy of random matrix physics.
It is grouped into the following sections:
    1. Imports
    2. Entropy Evolution Class
    3. Main Function
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
import os
from argparse import ArgumentParser
from importlib import import_module
from math import log10
from multiprocessing import Pool
from time import time

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Local application imports
from ._mc import MonteCarlo
from config.entropy_evolution_config import config


# =============================
# 2. Entropy Evolution Class
# =============================
class EntropyEvolution(MonteCarlo):
    @staticmethod
    def _worker_func(args: dict):
        # Unpack worker argument
        worker_id = args["id"]
        num_workers = args["num_workers"]
        ens_input = args["ens_input"]
        sim_input = args["sim_input"]

        # Copy ensemble input and pop name
        ens_args = ens_input.copy()
        ens_args.pop("name")

        # Initialize ensemble
        module = import_module(f"ensembles.{ens_input['name']}")
        ENS = getattr(module, module.class_name)
        ensemble = ENS(**ens_args)

        # Unpack simulation input
        realizs = sim_input["realizs"]

        # Initialize times array
        times = np.logspace(
            config.logtime_min, config.logtime_max, num=config.logtime_num
        )

        # Split times array and choose worker times based on worker ID
        worker_times = np.array_split(times, num_workers)[worker_id]

        # Declare initial state
        initial_state = np.zeros(ensemble.dim, dtype=ensemble.dtype)
        initial_state[0] = 1.0 + 0.0j

        # Calculate evolution of state
        evolved_states = ensemble.evolve_state(initial_state, worker_times, realizs)

        # Calculate entropy curve
        entropy, probabilities = ensemble.entropy(evolved_states)

        # Return entropy evolution
        return np.vstack((worker_times, entropy, probabilities))

    def _save_entropy_evolution(self, results: np.ndarray) -> None:
        # Point to times and entropy curve arrays
        times = results[0, :]
        entropy = results[1, :]
        probabilities = results[2:, :]

        # Retrieve Thouless time
        thouless_time = self._retrieve_thouless_time()

        # Create results directory
        data_dir = self._create_res_dir(res_type="data")

        # Save entropy evolution to file
        np.savez_compressed(
            os.path.join(data_dir, config.entropy_data_filename),
            times=times,
            entropy=entropy,
            probabilities=probabilities,
            thouless_time=thouless_time,
        )

    def _plot_entropy_evolution(self) -> None:
        # Load entropy evolution data
        data_dir = self._create_res_dir(res_type="data")
        data = np.load(os.path.join(data_dir, config.entropy_data_filename))

        # Unpack data
        times = data["times"]
        entropy = data["entropy"]
        probabilities = data["probabilities"]

        # Retrieve Thouless time if it exists
        try:
            thouless_time = data["thouless_time"]
        except ValueError:
            thouless_time = None

        # Initialize plot
        fig, ax = self._initialize_plot()

        # Set x- and y-axis scales to logarithmic
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Plot entropy evolution
        ax.plot(
            times,
            entropy,
            color=config.entropy_color,
            zorder=config.entropy_zorder,
            alpha=config.entropy_alpha,
            linewidth=config.entropy_width,
        )

        # Plot probabilities
        for i in range(self.ensemble.dim):
            # Plot initial state probability in red
            ax.plot(
                times,
                probabilities[i, :],
                color=config.initial_color if i == 0 else config.other_color,
                zorder=config.initial_zorder if i == 0 else config.other_zorder,
                alpha=(
                    config.initial_alpha
                    if i == 0
                    else config.other_alpha / (self.ensemble.dim - 1)
                ),
                linewidth=config.initial_width,
            )

        # Add horizontal line at probability 1/dim
        ax.axhline(
            y=1.0 / self.ensemble.dim,
            color="Black",
            zorder=0,
            alpha=1.0,
            linewidth=1.0,
            linestyle="--",
        )

        # Set legend elements
        legend_elements = [
            Line2D(
                [0],
                [0],
                color=config.entropy_color,
                lw=config.entropy_width,
                label=config.entropy_legend,
            ),
            Line2D(
                [0],
                [0],
                color=config.initial_color,
                lw=config.initial_width,
                label=config.initial_legend,
            ),
            Line2D(
                [0],
                [0],
                color=config.other_color,
                lw=config.other_width,
                alpha=config.other_legend_alpha,
                label=config.other_legend,
            ),
            Line2D(
                [0],
                [0],
                color="Black",
                linestyle="--",
                lw=1.0,
                alpha=1.0,
                label=r"$1 / D$",
            ),
        ]

        # Add legend to plot
        ax.legend(framealpha=1.0, handles=legend_elements)

        # Add Thouless time to plot if it exists
        if thouless_time is not None:
            # Add Thouless time to plot
            ax.axvline(
                x=thouless_time,
                color=config.thouless_color,
                zorder=config.thouless_zorder,
                alpha=config.thouless_alpha,
                linewidth=config.thouless_width,
                label=config.thouless_legend,
            )

            # Shade region after Thouless time
            ax.fill_between(
                times,
                config.entropy_ymin,
                config.entropy_ymax,
                where=times >= thouless_time,
                color=config.chaos_color,
                zorder=config.chaos_zorder,
                alpha=config.chaos_alpha,
            )

            # Add text box with Thouless time
            ax.text(
                config.x_position * thouless_time,
                config.y_position * config.entropy_ymin,
                rf"{config.thouless_legend} {log10(thouless_time):.2f}",
                fontsize=config.fontsize,
                color="Black",
                bbox=dict(
                    facecolor="white",
                    edgecolor=config.edgecolor,
                    boxstyle="round,pad=0.5",
                ),
                zorder=config.thouless_zorder,
            )

        # Add Heisenberg time to plot
        ax.axvline(
            x=1e0,
            color=config.heisenberg_color,
            zorder=config.heisenberg_zorder,
            alpha=config.heisenberg_alpha,
            linewidth=config.heisenberg_width,
        )

        # Shade region after Heisenberg time
        ax.fill_between(
            times,
            config.entropy_ymin,
            config.entropy_ymax,
            where=times >= 1e0,
            color=config.thermalization_color,
            zorder=config.thermalization_zorder,
            alpha=config.thermalization_alpha,
        )

        # Set titles, labels, and axis limits
        ax.set_title(f"{self.ensemble} {config.entropy_title}")
        ax.set_xlabel(config.entropy_xlabel)
        ax.set_ylabel(config.entropy_ylabel)
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(config.entropy_ymin, config.entropy_ymax)

        # Create results directory to store plot
        plot_dir = self._create_res_dir(res_type="plots")

        # Save plot to file
        fig.savefig(os.path.join(plot_dir, config.entropy_plot_filename), dpi=300)

        # Close plot
        plt.close(fig)

        # Print messages
        initial_state_prob = np.mean(probabilities[0, :][times > 1e0])
        other_state_prob = np.mean(probabilities[-1, :][times > 1e0])
        ratio = initial_state_prob / other_state_prob
        beta_estimate = 2.0 / (ratio - 1)
        print("Plotting Complete")
        print("Initial State Probability: ", np.mean(probabilities[0, :][times > 1e0]))
        print("Other State Probability: ", np.mean(probabilities[-1, :][times > 1e0]))
        print("Ratio: ", ratio)
        print("Beta Estimate: ", beta_estimate)

    def run(self):
        # Start timer
        start_time = time()

        # Create worker arguments
        worker_args = self._create_worker_args()

        # Run workers
        with Pool(processes=self.workers) as pool:
            data = pool.map(self._worker_func, worker_args)

        # Retrieve entropy and probabilities from results
        results = np.concatenate(data, axis=1)
        del data

        # Save entropy evolution to file
        self._save_entropy_evolution(results)

        # Plot entropy evolution
        self._plot_entropy_evolution()

        # Stop timer and calculate elapsed time
        sim_time = time() - start_time

        # Print simulation time
        print(f"Simulation time: {sim_time:.2f} seconds")


# =============================
# 3. Main Function
# =============================
def main():
    # Create argument parser
    parser = ArgumentParser(description="Entropy Evolution Monte Carlo")

    # Retrieve Monte Carlo arguments
    mc_args = EntropyEvolution._get_mc_args(parser)

    # Initialize spectral statistics simulation class
    mc = EntropyEvolution(**mc_args)

    # Run spectral statistics simulation
    mc.run()


# Run the main function
if __name__ == "__main__":
    main()
