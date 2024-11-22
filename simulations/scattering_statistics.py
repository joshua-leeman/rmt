# rmt.simulations.scattering_statistics.py
"""
This module contains the programs for performing the Monte Carlo
simulations of random matrix ensemble scattering statistics.
It is grouped into the following sections:
    1. Imports
    2. Scattering Statistics Class
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

# Local application imports
from ._mc import MonteCarlo
from config.scattering_statistics_config import config


# =============================
# 2. Scattering Statistics Class
# =============================
class ScatteringStatistics(MonteCarlo):
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
        indices = sim_input["indices"]

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

        # Determine number of realizations per worker
        realizs_per_worker, remainder = divmod(realizs, num_workers)

        # Initialize realization array
        realizs_array = np.full(num_workers, realizs_per_worker, dtype=int)

        # Distribute remainder realizations
        realizs_array[:remainder] += 1

        # Determine worker's number of realizations
        realizs = realizs_array[worker_id]

        # Initialize energy array
        energies = np.linspace(config.energy_min, config.energy_max, config.energy_num)

        # Return scattering one- and two-point correlations
        return ensemble.S_elem_sample(energies, indices, W, realizs)

    def _save_scattering_statistics(self, energies, T_coeffs, indices, SS_conn):
        # Create results directory
        data_dir = self._create_res_dir(res_type="data")

        # Save results to file
        np.savez_compressed(
            os.path.join(data_dir, config.scattering_data_filename),
            energies=energies,
            T_coeffs=T_coeffs,
            indices=indices,
            SS_conn=SS_conn,
        )

    def _plot_SS_conn(self):
        # Load two-point correlation data
        data_dir = self._create_res_dir(res_type="data")
        data = np.load(os.path.join(data_dir, config.scattering_data_filename))

        # Unpack data
        energies = data["energies"]
        indices = data["indices"]
        SS_conn = data["SS_conn"]

        # Plot connected two-point correlation
        for i, index in enumerate(indices):
            # Initialize plot
            fig, ax = self._initialize_plot()

            ax.plot(
                energies,
                SS_conn[:, i].real,
                label="real",
                color="Red",
                linestyle="-",
                zorder=2,
            )
            ax.plot(
                energies,
                SS_conn[:, i].imag,
                label="imag",
                color="Blue",
                linestyle="-",
                zorder=1,
            )

            # Create two-point correlation expression
            plot_title = rf"$\langle S_{{{index[0][0]}{index[0][1]}}}(E_1)S^*_{{{index[1][0]}{index[1][1]}}}(E_2) \rangle{{}}_\textrm{{\small conn}}$"

            ax.set_title(rf"{self.ensemble} {plot_title} $\Lambda = {self.channels}$")
            ax.set_ylabel(plot_title)

            # Add legend to plot
            ax.legend(framealpha=1.0)

            # Set titles, labels, and axis limits
            ax.set_xlabel(config.SS_conn_xlabel)
            ax.set_xlim(0.0, energies[-1])

            # Create results directory to store plot
            plot_dir = self._create_res_dir(res_type="plots")

            # Save plot to file
            plt.savefig(
                os.path.join(
                    plot_dir,
                    f"SS{index[0][0]}{index[0][1]}{index[1][0]}{index[1][1]}_conn.png",
                ),
                dpi=300,
            )

            # Close plot
            plt.close(fig)

    def run(self):
        # Start timer
        start_time = time()

        # Unpack indices from simulation input
        indices = self._sim_input["indices"]

        # Create worker arguments
        worker_args = self._create_worker_args()

        # Run workers
        with Pool(processes=self.workers) as pool:
            results = pool.map(self._worker_func, worker_args)

        # Unpack results
        S_avgs = np.sum([result[0] for result in results], axis=0)
        SS_disc = np.sum([result[1] for result in results], axis=0)
        SS_conn = np.sum([result[2] for result in results], axis=0)
        del results

        # Stop timer and calculate elapsed time
        sim_time = time() - start_time

        # Print simulation time
        print(f"Simulation time: {sim_time:.2f} seconds")

        # Divide by number of realizations in-place
        S_avgs /= self.realizs
        SS_disc /= self.realizs
        SS_conn /= self.realizs

        # Finish calculation of transmission coefficients
        T_coeffs = 1 - np.abs(S_avgs) ** 2

        # Determine number of channels
        self._channels = T_coeffs.shape[-1]

        # Finish calculation of connected two-point correlation
        SS_conn -= SS_disc[:, ::2] * SS_disc[:, 1::2]

        # Copy ensemble input and pop name
        ens_input = self._ens_input.copy()
        ens_input.pop("name")

        # Initialize ensemble
        module = import_module(f"ensembles.{self._ens_input['name']}")
        ENS = getattr(module, module.class_name)
        ensemble = ENS(**ens_input)

        # Replicate energy array
        energies = np.linspace(config.energy_min, config.energy_max, config.energy_num)

        # Unfold energies
        energies = ensemble.unfold(energies)

        # Determine center of energy array
        c = energies.size // 2

        # Calculate energy differences
        energies = energies[c:] - energies[c::-1]

        # Save scattering statistics data
        self._save_scattering_statistics(energies, T_coeffs, indices, SS_conn)

        # Plot results
        self._plot_SS_conn()

    @property
    def channels(self):
        return self._channels


# =============================
# 3. Main Function
# =============================
def main():
    # Create argument parser
    parser = ArgumentParser(description="Scattering statistics Monte Carlo")

    # Retrieve Monte Carlo arguments
    mc_args = ScatteringStatistics._get_mc_args(parser)

    # Initialize scattering statistics simulations class
    mc = ScatteringStatistics(**mc_args)

    # Run scattering statistics simulations
    mc.run()


# Run main function
if __name__ == "__main__":
    main()
