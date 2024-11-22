# rmt.simulations.time_delay_statistics.py
"""
This module contains the programs for performing the Monte Carlo
simulations of random matrix ensemble time delay statistics.
It is grouped into the following sections:
    1. Imports
    2. Time Delay Statistics Class
    3. Main Function
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
import os
from argparse import ArgumentParser
from importlib import import_module
from multiprocessing import Pool
from time import time

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Local application imports
from ._mc import MonteCarlo
from config.time_delay_statistics_config import config


# =============================
# 2. Time Delay Statistics Class
# =============================
class TimeDelayStatistics(MonteCarlo):
    @staticmethod
    def _worker_func(args: dict):
        # Unpack arguments
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
        sim_name = sim_input["name"]
        couplings = sim_input["couplings"]

        # Check if couplings is dictionary
        if isinstance(couplings, dict):
            # Obtain coupling matrix attributes
            channels = couplings["channels"]
            strength = couplings["strength"]

            # Construct coupling matrix
            W = np.ones(channels, dtype=ensemble.dtype) * strength / np.sqrt(2 * np.pi)

        # Check if couplings is string
        else:
            # Retrieve coupling matrix
            try:
                # Load channel couplings array
                coupling_data = np.load(f"inputs/{sim_name}_inputs/{couplings}.npz")

                # Retrieve coupling matrix
                if "W" in coupling_data:
                    # Retrieve coupling matrix
                    W = coupling_data["W"]

                    # If W is 2D array and its shape isn't (ensemble.dim, channels)
                    if len(W.shape) == 2 and W.shape[0] != ensemble.dim:
                        # Raise error
                        raise ValueError("Coupling matrix shape mismatch")

                    # Determine number of channels
                    channels = W.shape[-1]
                else:
                    raise FileNotFoundError

            # Construct default coupling matrix if array not found
            except FileNotFoundError:
                # Retrieve default coupling matrix if module not found
                W = np.ones(1, dtype=ensemble.dtype) / np.sqrt(2 * np.pi)

                # Set number of channels
                channels = 1

        # Initialize energy array
        energies = np.linspace(config.energy_min, config.energy_max, config.energy_num)

        # Split energies array and choose worker energies based on worker ID
        worker_energies = np.array_split(energies, num_workers)[worker_id]

        # Return time delay sample
        return ensemble.time_delay_sample(worker_energies, W, realizs)

    def _save_proper_time_delays(
        self, energies: np.ndarray, proper_time_delays: np.ndarray
    ):
        # Create results directory
        data_dir = self._create_res_dir("data")

        # Save proper times to file
        np.savez_compressed(
            os.path.join(data_dir, config.proper_times_data_filename),
            energies=energies,
            proper_time_delays=proper_time_delays,
        )

    def _save_wigner_time_delays(
        self, energies: np.ndarray, wigner_time_delays: np.ndarray
    ):
        # Create results directory
        data_dir = self._create_res_dir("data")

        # Save Wigner times to file
        np.savez_compressed(
            os.path.join(data_dir, config.wigner_times_data_filename),
            energies=energies,
            wigner_time_delays=wigner_time_delays,
        )

    def _plot_proper_time_delays(self):
        # Load proper time delays data
        data_dir = self._create_res_dir("data")
        data = np.load(os.path.join(data_dir, config.proper_times_data_filename))

        # Unpack data
        energies = data["energies"]
        proper_times = data["proper_time_delays"]

        # Reshape proper times
        proper_times = proper_times.reshape((energies.size, -1))

        # Retrieve thouless time
        thouless_time = self._retrieve_thouless_time()

        # Initialize plot
        fig, ax = self._initialize_plot()

        # Plot proper time delays
        ax.plot(
            energies,
            np.max(proper_times, axis=1),
            color="RoyalBlue",
            linewidth=1.5,
        )
        ax.plot(
            energies,
            np.min(proper_times, axis=1),
            color="RoyalBlue",
            linewidth=1.5,
        )
        ax.fill_between(
            energies,
            np.min(proper_times, axis=1),
            np.max(proper_times, axis=1),
            color="LightBlue",
            alpha=0.7,
        )
        ax.plot(
            energies,
            np.mean(proper_times, axis=1),
            color="Black",
            linewidth=1.5,
        )

        # Set y-axis to log scale
        ax.set_yscale("log")

        # Plot horizontal line at Thouless time
        ax.axhline(
            y=thouless_time,
            color="Black",
            linestyle="--",
            linewidth=1.5,
        )

        # Plot horizontal line at Heisenberg time
        ax.axhline(
            y=1,
            color="Black",
            linestyle="--",
            linewidth=1.5,
        )

        # Set y-axis limits
        ax.set_ylim(10**config.logtime_min, 10**config.logtime_max)

        # Create results directory to store plot
        plot_dir = self._create_res_dir("plots")

        # Save plot to file
        plt.savefig(os.path.join(plot_dir, config.proper_times_plot_filename), dpi=300)

        # Close plot
        plt.close(fig)

    def _plot_wigner_time_delays(self):
        pass

    def run(self):
        # Start timer
        start_time = time()

        # Create worker arguments
        worker_args = self._create_worker_args()

        # Run workers
        with Pool(processes=self.workers) as pool:
            results = pool.map(self._worker_func, worker_args)

        # Unpack results
        proper_times = np.concatenate([result[0] for result in results], axis=0)
        wigner_times = np.concatenate([result[1] for result in results], axis=0)

        # Nomalize proper times with mean level spacing
        mean_spacing = self._retrieve_mean_spacing()
        if mean_spacing is not None:
            print(f"Mean level spacing: {mean_spacing:.2f}")
            proper_times *= mean_spacing
            wigner_times *= mean_spacing

        # End timer
        end_time = time()

        # Print simulation time
        print(f"Simulation time: {end_time - start_time:.2f} seconds")

        # Reinitialize energy array
        energies = np.linspace(config.energy_min, config.energy_max, config.energy_num)

        # Save proper and Wigner time delays
        self._save_proper_time_delays(energies, proper_times)
        self._save_wigner_time_delays(energies, wigner_times)

        # Plot proper and Wigner time delays
        self._plot_proper_time_delays()
        self._plot_wigner_time_delays()


# =============================
# 3. Main Function
# =============================
def main():
    # Create argument parser
    parser = ArgumentParser(description="Time delay statistics Monte Carlo")

    # Retrieve Monte Carlo arguments
    mc_args = TimeDelayStatistics._get_mc_args(parser)

    # Initialize time delay statistics simulations class
    mc = TimeDelayStatistics(**mc_args)

    # Run time delay statistics simulations
    mc.run()


# Run main function
if __name__ == "__main__":
    main()
