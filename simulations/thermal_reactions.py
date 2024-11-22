# rmt.simulations.thermal_reactions.py
"""
This module contains the programs for performing the Monte Carlo
simulations of the thermodynamics of quantum-chaotic compound reactions.
It is grouped into the following sections:
    1. Imports
    2. Thermal Reactions Class
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
from config.thermal_reactions_config import config


# =============================
# 2. Thermal Reactions Class
# =============================
class ThermalReactions(MonteCarlo):
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
        ens_module = import_module(f"ensembles.{ens_input['name']}")
        ENS = getattr(ens_module, ens_module.class_name)
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
        energies = np.linspace(
            config.energy_min, config.energy_max, num=config.energy_num
        )

        # Split energies array and choose worker energies based on worker ID
        worker_energies = np.array_split(energies, num_workers)[worker_id]

        # Declare initial state
        initial_state = np.zeros(channels, dtype=ensemble.dtype)
        initial_state[0] = 1.0 + 0.0j

        # Calculate scattered states
        T_coeffs, scattered_states = ensemble.scatter_state(
            initial_state, worker_energies, W, realizs
        )

        # Calculate entropy curve
        entropy, probabilities = ensemble.entropy(scattered_states)

        # Return stacked results
        return np.vstack((worker_energies, T_coeffs, entropy, probabilities))

    def _save_transmission_coeffs(self, energies, T_coeffs):
        # Create results directory
        data_dir = self._create_res_dir(res_type="data")

        # Save transmission coefficients to file
        np.savez_compressed(
            os.path.join(data_dir, config.transmission_data_filename),
            energies=energies,
            T_coeffs=T_coeffs,
        )

    def _save_entropy_evolution(self, energies, entropy):
        # Create results directory
        data_dir = self._create_res_dir(res_type="data")

        # Save entropy evolution to file
        np.savez_compressed(
            os.path.join(data_dir, config.entropy_data_filename),
            energies=energies,
            entropy=entropy,
        )

    def _save_probabilities(self, energies, probabilities):
        # Create results directory
        data_dir = self._create_res_dir(res_type="data")

        # Save probabilities to file
        np.savez_compressed(
            os.path.join(data_dir, config.probabilities_data_filename),
            energies=energies,
            probabilities=probabilities,
        )

    def _save_temperature_curve(self, energies, temperature):
        # Create results directory
        data_dir = self._create_res_dir(res_type="data")

        # Save temperature curve to file
        np.savez_compressed(
            os.path.join(data_dir, config.temperature_data_filename),
            energies=energies,
            temperature=temperature,
        )

    def _save_free_energy_curve(self, energies, free_energy):
        # Create results directory
        data_dir = self._create_res_dir(res_type="data")

        # Save free energy curve to file
        np.savez_compressed(
            os.path.join(data_dir, config.free_energy_data_filename),
            energies=energies,
            free_energy=free_energy,
        )

    def _save_heat_capacity_curve(self, energies, heat_capacity):
        # Create results directory
        data_dir = self._create_res_dir(res_type="data")

        # Save heat capacity curve to file
        np.savez_compressed(
            os.path.join(data_dir, config.heat_capacity_data_filename),
            energies=energies,
            heat_capacity=heat_capacity,
        )

    def _plot_transmisson_coeffs(self):
        # Load transmission coefficients data
        data_dir = self._create_res_dir(res_type="data")
        data = np.load(os.path.join(data_dir, config.transmission_data_filename))

        # Unpack data
        energies = data["energies"]
        T_coeffs = data["T_coeffs"]

        # Initialize plot
        fig, ax = self._initialize_plot()

        # Plot transmission coefficients
        for i in range(self.channels):
            ax.plot(
                energies,
                T_coeffs[i, :],
                color=(
                    config.initial_transmission_color
                    if i == 0
                    else config.other_transmission_color
                ),
                zorder=(
                    config.initial_transmission_zorder
                    if i == 0
                    else config.other_transmission_zorder
                ),
                alpha=(
                    config.initial_transmission_alpha
                    if i == 0
                    else config.other_transmission_alpha / (self.channels - 1)
                ),
                linewidth=config.transmission_width,
            )

        # Set legend elements
        legend_elements = [
            Line2D(
                [0],
                [0],
                color=config.initial_transmission_color,
                lw=config.transmission_width,
                label=config.initial_transmission_legend,
            ),
            Line2D(
                [0],
                [0],
                color=config.other_transmission_color,
                lw=config.transmission_width,
                label=config.other_transmission_legend,
            ),
        ]

        # Add legend to plot
        ax.legend(framealpha=1.0, handles=legend_elements)

        # Set titles, labels, and axis limits
        ax.set_title(
            rf"{self.ensemble} {config.transmission_title} $\Lambda = {self.channels}$"
        )
        ax.set_xlabel(config.transmission_xlabel)
        ax.set_ylabel(config.transmission_ylabel)
        ax.set_xlim(config.energy_xlim_min, config.energy_xlim_max)
        ax.set_ylim(config.transmission_ymin, config.transmission_ymax)

        # Create results directory to store plot
        plot_dir = self._create_res_dir(res_type="plots")

        # Save plot to file
        plt.savefig(os.path.join(plot_dir, config.transmission_plot_filename), dpi=300)

        # Close plot
        plt.close(fig)

    def _plot_entropy_evolution(self):
        # Load entropy evolution data
        data_dir = self._create_res_dir(res_type="data")
        data = np.load(os.path.join(data_dir, config.entropy_data_filename))

        # Unpack data
        energies = data["energies"]
        entropy = data["entropy"]

        # Initialize plot
        fig, ax = self._initialize_plot()

        # Plot entropy evolution
        ax.plot(
            energies,
            entropy,
            color=config.entropy_color,
            zorder=config.entropy_zorder,
            alpha=config.entropy_alpha,
            linewidth=config.entropy_width,
        )

        # Set titles, labels, and axis limits
        ax.set_title(
            rf"{self.ensemble} {config.entropy_title} $\Lambda = {self.channels}$"
        )
        ax.set_xlabel(config.entropy_xlabel)
        ax.set_ylabel(config.entropy_ylabel)
        ax.set_xlim(config.energy_xlim_min, config.energy_xlim_max)
        ax.set_ylim(config.entropy_ymin, config.entropy_ymax)

        # Create results directory to store plot
        plot_dir = self._create_res_dir(res_type="plots")

        # Save plot to file
        plt.savefig(os.path.join(plot_dir, config.entropy_plot_filename), dpi=300)

        # Close plot
        plt.close(fig)

    def _plot_probabilities(self):
        # Load probabilities data
        data_dir = self._create_res_dir(res_type="data")
        data = np.load(os.path.join(data_dir, config.probabilities_data_filename))

        # Unpack data
        energies = data["energies"]
        probabilities = data["probabilities"]

        # Initialize plot
        fig, ax = self._initialize_plot()

        # Set logarithmic scale for y-axis
        ax.set_yscale("log")

        # Plot probabilities
        for i in range(self.channels):
            ax.plot(
                energies,
                probabilities[i, :],
                color=(
                    config.initial_probability_color
                    if i == 0
                    else config.other_probability_color
                ),
                zorder=(
                    config.initial_probability_zorder
                    if i == 0
                    else config.other_probability_zorder
                ),
                alpha=(
                    config.initial_probability_alpha
                    if i == 0
                    else config.other_probability_alpha / (self.channels - 1)
                ),
                linewidth=config.probability_width,
            )

        # Set legend elements
        legend_elements = [
            Line2D(
                [0],
                [0],
                color=config.initial_probability_color,
                lw=config.probability_width,
                label=config.initial_probability_legend,
            ),
            Line2D(
                [0],
                [0],
                color=config.other_probability_color,
                lw=config.probability_width,
                label=config.other_probability_legend,
            ),
            Line2D(
                [0],
                [0],
                color="Black",
                linestyle="--",
                lw=1.0,
                alpha=1.0,
                label=r"$1 / \Lambda$",
            ),
        ]

        # Add horizontal line at probability 1/channels
        ax.axhline(
            y=1.0 / self.channels,
            color="Black",
            zorder=0,
            alpha=1.0,
            linewidth=1.0,
            linestyle="--",
        )

        # Add legend to plot
        ax.legend(framealpha=1.0, handles=legend_elements)

        # Set titles, labels, and axis limits
        ax.set_title(
            rf"{self.ensemble} {config.probabilities_title} $\Lambda = {self.channels}$"
        )
        ax.set_xlabel(config.probabilities_xlabel)
        ax.set_ylabel(config.probabilities_ylabel)
        ax.set_xlim(config.energy_xlim_min, config.energy_xlim_max)
        ax.set_ylim(config.prob_ymin, config.prob_ymax)

        # Create results directory to store plot
        plot_dir = self._create_res_dir(res_type="plots")

        # Save plot to file
        plt.savefig(os.path.join(plot_dir, config.probabilities_plot_filename), dpi=300)

        # Close plot
        plt.close(fig)

    def _plot_temperature_curve(self):
        # Load temperature data
        data_dir = self._create_res_dir(res_type="data")
        data = np.load(os.path.join(data_dir, config.temperature_data_filename))

        # Unpack data
        energies = data["energies"]
        temperature = data["temperature"]

        # Initialize plot
        fig, ax = self._initialize_plot()

        # Plot temperature curve
        ax.plot(
            energies[(temperature > 0)],
            temperature[(temperature > 0)],
            color=config.temperature_color,
            zorder=config.temperature_zorder,
            alpha=config.temperature_alpha,
            linewidth=config.temperature_width,
        )
        ax.plot(
            energies[(temperature < 0)],
            temperature[(temperature < 0)],
            color=config.temperature_color,
            zorder=config.temperature_zorder,
            alpha=config.temperature_alpha,
            linewidth=config.temperature_width,
        )

        # Add lines for x-axis and y-axis
        ax.axhline(y=0, color="Black", linewidth=1.0, linestyle="--", zorder=0)
        ax.axvline(x=0, color="Black", linewidth=1.0, linestyle="--", zorder=0)

        # Set titles, labels, and axis limits
        ax.set_title(
            rf"{self.ensemble} {config.temperature_title} $\Lambda = {self.channels}$"
        )
        ax.set_xlabel(config.temperature_xlabel)
        ax.set_ylabel(config.temperature_ylabel)
        ax.set_xlim(config.energy_xlim_min, config.energy_xlim_max)
        ax.set_ylim(config.temperature_ymin, config.temperature_ymax)

        # Create results directory to store plot
        plot_dir = self._create_res_dir(res_type="plots")

        # Save plot to file
        plt.savefig(os.path.join(plot_dir, config.temperature_plot_filename), dpi=300)

        # Close plot
        plt.close(fig)

    def _plot_free_energy_curve(self):
        # Load free energy data
        data_dir = self._create_res_dir(res_type="data")
        data = np.load(os.path.join(data_dir, config.free_energy_data_filename))

        # Unpack data
        energies = data["energies"]
        free_energy = data["free_energy"]

        # Initialize plot
        fig, ax = self._initialize_plot()

        # Plot free energy curve
        ax.plot(
            energies[(free_energy < 0)],
            free_energy[(free_energy < 0)],
            color=config.free_energy_color,
            zorder=config.free_energy_zorder,
            alpha=config.free_energy_alpha,
            linewidth=config.free_energy_width,
        )
        ax.plot(
            energies[(free_energy > 0)],
            free_energy[(free_energy > 0)],
            color=config.free_energy_color,
            zorder=config.free_energy_zorder,
            alpha=config.free_energy_alpha,
            linewidth=config.free_energy_width,
        )

        # Set titles, labels, and axis limits
        ax.set_title(
            rf"{self.ensemble} {config.free_energy_title} $\Lambda = {self.channels}$"
        )
        ax.set_xlabel(config.free_energy_xlabel)
        ax.set_ylabel(config.free_energy_ylabel)
        ax.set_xlim(config.energy_xlim_min, config.energy_xlim_max)
        ax.set_ylim(config.free_energy_ymin, config.free_energy_ymax)

        # Add lines for x-axis and y-axis
        ax.axhline(y=0, color="Black", linewidth=1.0, linestyle="--", zorder=0)
        ax.axvline(x=0, color="Black", linewidth=1.0, linestyle="--", zorder=0)

        # Create results directory to store plot
        plot_dir = self._create_res_dir(res_type="plots")

        # Save plot to file
        plt.savefig(os.path.join(plot_dir, config.free_energy_plot_filename), dpi=300)

        # Close plot
        plt.close(fig)

    def _plot_heat_capacity_curve(self):
        # Load heat capacity data
        data_dir = self._create_res_dir(res_type="data")
        data = np.load(os.path.join(data_dir, config.heat_capacity_data_filename))

        # Unpack data
        energies = data["energies"]
        heat_capacity = data["heat_capacity"]

        # Initialize plot
        fig, ax = self._initialize_plot()

        # Plot heat capacity curve
        ax.plot(
            energies,
            heat_capacity,
            color=config.heat_capacity_color,
            zorder=config.heat_capacity_zorder,
            alpha=config.heat_capacity_alpha,
            linewidth=config.heat_capacity_width,
        )

        # Set titles, labels, and axis limits
        ax.set_title(
            rf"{self.ensemble} {config.heat_capacity_title} $\Lambda = {self.channels}$"
        )
        ax.set_xlabel(config.heat_capacity_xlabel)
        ax.set_ylabel(config.heat_capacity_ylabel)
        ax.set_xlim(config.energy_xlim_min, config.energy_xlim_max)
        ax.set_ylim(config.heat_capacity_ymin, config.heat_capacity_ymax)

        # Add lines for x-axis and y-axis
        ax.axhline(y=0, color="Black", linewidth=1.0, linestyle="--", zorder=0)
        ax.axvline(x=0, color="Black", linewidth=1.0, linestyle="--", zorder=0)

        # Create results directory to store plot
        plot_dir = self._create_res_dir(res_type="plots")

        # Save plot to file
        plt.savefig(os.path.join(plot_dir, config.heat_capacity_plot_filename), dpi=300)

        # Close plot
        plt.close(fig)

    def run(self):
        # Start timer
        start_time = time()

        # Create worker arguments
        worker_args = self._create_worker_args()

        # Run workers
        with Pool(processes=self.workers) as pool:
            results = pool.map(self._worker_func, worker_args)

        # Concatenate results
        result_curves = np.concatenate(results, axis=1)
        del results

        # Stop timer and calculate elapsed time
        sim_time = time() - start_time

        # Print simulation time
        print(f"Simulation time: {sim_time:.2f} seconds")

        # Determine number of channels
        self._channels = (result_curves.shape[0] - 2) // 2

        # Unpack results
        energies = result_curves[0]
        T_coeffs = result_curves[1 : self.channels + 1]
        entropy = result_curves[self.channels + 1]
        probabilities = result_curves[self.channels + 2 :]

        # Calculate energy delta for derivative calculations
        delta = energies[1] - energies[0]

        # Calculate temperature, free energy, and heat capacity
        temperature = self.ensemble.temperature(entropy, delta)
        free_energy = self.ensemble.free_energy(energies, entropy, temperature)
        heat_capacity = self.ensemble.heat_capacity(entropy, temperature, delta)

        # Save results to .npz files
        self._save_transmission_coeffs(energies, T_coeffs)
        self._save_entropy_evolution(energies, entropy)
        self._save_probabilities(energies, probabilities)
        self._save_temperature_curve(energies, temperature)
        self._save_free_energy_curve(energies, free_energy)
        self._save_heat_capacity_curve(energies, heat_capacity)

        # Plot results
        self._plot_transmisson_coeffs()
        self._plot_entropy_evolution()
        self._plot_probabilities()
        self._plot_temperature_curve()
        self._plot_free_energy_curve()
        self._plot_heat_capacity_curve()

    @property
    def channels(self):
        return self._channels


# =============================
# 3. Main Function
# =============================
def main():
    # Create argument parser
    parser = ArgumentParser(description="Thermal Reactions Monte Carlo")

    # Retrieve Monte Carlo arguments
    mc_args = ThermalReactions._get_mc_args(parser)

    # Initialize spectral statistics simulation class
    mc = ThermalReactions(**mc_args)

    # Run spectral statistics simulation
    mc.run()


# Run the main function
if __name__ == "__main__":
    main()
