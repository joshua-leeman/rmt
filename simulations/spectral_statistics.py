# rmt.simulations.spectral_statistics.py
"""
This module contains the programs for performing the Monte Carlo
simulations of random matrix ensemble spectral statistics.
It is grouped into the following sections:
    1. Imports
    2. Spectral Statistics Class
    3. Main Function
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
import os
from argparse import ArgumentParser
from importlib import import_module
from math import exp, log, log10
from multiprocessing import Pool
from time import time

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Local application imports
from ._mc import MonteCarlo
from config.spectral_statistics_config import config


# =============================
# 2. Spectral Statistics Class
# =============================
class SpectralStatistics(MonteCarlo):
    def _create_worker_args(self):
        # Calculate realizations per worker and remainder
        realizs_per_worker, remainder = divmod(self.realizs, self.workers)

        # Initialize realization array
        realizs_array = np.full(self.workers, realizs_per_worker, dtype=int)

        # Distribute remainder realizations
        realizs_array[:remainder] += 1

        # Initialize list of worker arguments
        worker_args = [None for _ in range(self.workers)]

        # Loop over worker argument list
        for i in range(self.workers):
            # Write worker dictionary argument
            worker_args[i] = {
                "ens_input": self._ens_input,
                "sim_input": {"realizs": realizs_array[i]},
            }

        # Return list of worker arguments
        return worker_args

    @staticmethod
    def _worker_func(args: dict):
        # Unpack worker arguments
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

        # Return eigenvalues
        return ensemble.eigval_sample(realizs)

    def _calc_spectral_histogram(self, eigenvalues):
        # Calculate bin edges and bin range
        min_edge, max_edge = np.min(eigenvalues), np.max(eigenvalues)

        # Arrange bin edges
        bins = np.arange(min_edge, max_edge + config.bin_width, config.bin_width)

        # Calculate normalized histogram of eigenvalues
        hist_counts, hist_edges = np.histogram(eigenvalues, bins=bins, density=True)

        # Calculate mean level spacing
        mean_spacing = np.mean(np.diff(eigenvalues, axis=1))

        # Create results directory
        data_dir = self._create_res_dir(res_type="data")

        # Save histogram data to file
        np.savez_compressed(
            os.path.join(data_dir, config.spectral_data_filename),
            hist_counts=hist_counts,
            hist_edges=hist_edges,
            mean_spacing=mean_spacing,
        )

    def _calc_spacing_distribution(self, unfolded_eigvals):
        # If dimension is one, return
        if self.ensemble.dim == 1:
            return

        # Calculate spacings array
        spacings = self.ensemble.level_spacings(unfolded_eigvals)

        # Calculate bin edges and bin range
        min_edge, max_edge = np.min(spacings), np.max(spacings)

        # Arrange bin edges
        bins = np.arange(min_edge, max_edge + config.bin_width, config.bin_width)

        # Calculate normalized histogram of eigenvalues
        hist_counts, hist_edges = np.histogram(spacings, bins=bins, density=True)

        # Create results directory
        data_dir = self._create_res_dir(res_type="data")

        # Save histogram data to file
        np.savez_compressed(
            os.path.join(data_dir, config.spacings_data_filename),
            hist_counts=hist_counts,
            hist_edges=hist_edges,
        )

    def _calc_spectral_form_factors(self, unfolded_eigvals):
        # Initialize logtimes array
        times = np.logspace(
            config.logtime_min, config.logtime_max, num=config.logtime_num
        )

        # Allocate memory for form factors
        sff = np.empty_like(times)
        csff = np.empty_like(times)

        # Batch logtimes for memory efficiency
        num_batches = max(
            config.num_batches,
            10 * times.size * self.realizs * self.ensemble.dim // self.max_memory,
        )
        batched_times = np.array_split(times, num_batches)

        # Loop over batched logtimes and fill form factors
        index = 0
        for batch in batched_times:
            # Calculate and store form factors
            sff[index : index + batch.size], csff[index : index + batch.size] = (
                self.ensemble.form_factors(
                    times=batch, unfolded_eigvals=unfolded_eigvals
                )
            )

            # Increment index
            index += batch.size

        # Initialize Thouless time
        thouless_time = times[-1]

        # Loop backwards over form factors to determine Thouless time
        start = len(np.where(times < 1)[0])
        for i in range(start - 2, -1, -1):
            if log10(sff[i] / csff[i]) > config.sffs_epsilon:
                thouless_time = times[i + 1]
                break

        # Create results directory
        data_dir = self._create_res_dir(res_type="data")

        # Save form factors data to file
        np.savez_compressed(
            os.path.join(data_dir, config.form_factors_data_filename),
            times=times,
            sff=sff,
            csff=csff,
            thouless_time=thouless_time,
        )

    def _plot_spectral_histogram(self):
        # Load histogram data from file
        data_dir = self._create_res_dir(res_type="data")
        hist_data = np.load(os.path.join(data_dir, config.spectral_data_filename))

        # Unpack histogram data
        hist_counts = hist_data["hist_counts"]
        hist_edges = hist_data["hist_edges"]

        # Initialize plot
        fig, ax = self._initialize_plot()

        # Plot histogram
        ax.hist(
            hist_edges[:-1],
            bins=hist_edges,
            weights=hist_counts,
            color=config.hist_color,
            zorder=config.hist_zorder,
            alpha=config.hist_alpha,
        )

        # Create array of energy values
        energies = np.linspace(
            -self.ensemble.scale, self.ensemble.scale, num=config.density_num
        )

        # Evaluate average spectral density at energy values
        density = np.vectorize(self.ensemble.mean_density)(energies)

        # Plot average spectral density
        density_line = ax.plot(
            energies,
            density,
            color=config.density_color,
            zorder=config.spectral_zorder,
            linewidth=config.density_width,
        )

        # Set title, labels, and limits
        ax.set_title(f"{self.ensemble} {config.spectrum_title}")
        ax.set_xlabel(config.spectrum_xlabel)
        ax.set_ylabel(config.spectrum_ylabel)
        ax.set_xlim(
            -config.spectral_bound * self.ensemble.scale,
            config.spectral_bound * self.ensemble.scale,
        )

        # Add legend
        ax.legend(
            handles=[Patch(color=config.hist_color), density_line[0]],
            labels=[config.hist_legend, config.spectral_legend],
            framealpha=1.0,
        )

        # Create results directory to store plot
        plot_dir = self._create_res_dir(res_type="plots")

        # Save plot to file
        fig.savefig(os.path.join(plot_dir, config.spectral_plot_filename), dpi=300)

        # Close plot
        plt.close(fig)

    def _plot_spacing_distribution(self):
        # Load histogram data from file
        data_dir = self._create_res_dir(res_type="data")
        hist_data = np.load(os.path.join(data_dir, config.spacings_data_filename))

        # Unpack histogram data
        hist_counts = hist_data["hist_counts"]
        hist_edges = hist_data["hist_edges"]

        # Initialize plot
        fig, ax = self._initialize_plot()

        # Plot histogram
        ax.hist(
            hist_edges[:-1],
            bins=hist_edges,
            weights=hist_counts,
            color=config.hist_color,
            zorder=config.hist_zorder,
            alpha=config.hist_alpha,
        )

        # Create array of spacing values
        spacings_array = np.linspace(0, config.spacings_max, num=config.density_num)

        # Calculate Wigner surmise at spacing values
        surmise = self.ensemble.wigner_surmise(s=spacings_array)

        # Plot Wigner surmise
        surmise_line = ax.plot(
            spacings_array,
            surmise,
            color=config.density_color,
            zorder=config.surmise_zorder,
            linewidth=config.density_width,
        )

        # Set title, labels, and limits
        ax.set_title(f"{self.ensemble} {config.spacings_title}")
        ax.set_xlabel(config.spacings_xlabel)
        ax.set_ylabel(config.spacings_ylabel)
        ax.set_xlim(0, config.spacings_max)

        # Add legend
        ax.legend(
            handles=[Patch(color=config.hist_color), surmise_line[0]],
            labels=[config.spacings_legend, config.surmise_legend],
            framealpha=1.0,
        )

        # Create results directory to store plot
        plot_dir = self._create_res_dir(res_type="plots")

        # Save plot to file
        fig.savefig(os.path.join(plot_dir, config.spacings_plot_filename), dpi=300)

        # Close plot
        plt.close(fig)

    def _plot_spectral_form_factors(self):
        # Load form factors data from file
        data_dir = self._create_res_dir(res_type="data")
        form_factors_data = np.load(
            os.path.join(data_dir, config.form_factors_data_filename)
        )

        # Unpack form factors data
        times = form_factors_data["times"]
        sff = form_factors_data["sff"]
        csff = form_factors_data["csff"]
        thouless_time = form_factors_data["thouless_time"]

        # Calculate universal connected spectral form factor
        csff_universal = np.vectorize(self.ensemble.csff_universal)(times)

        # Initialize plot
        fig, ax = self._initialize_plot()

        # Set x- and y-scales to logarithmic
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Plot spectral form factor
        ax.plot(
            times,
            sff,
            color=config.sff_color,
            zorder=config.sff_zorder,
            alpha=config.sff_alpha,
            linewidth=config.sff_width,
            label=config.sff_legend,
        )

        # Plot connected spectral form factor
        ax.plot(
            times,
            csff,
            color=config.csff_color,
            zorder=config.csff_zorder,
            alpha=config.csff_alpha,
            linewidth=config.density_width,
            label=config.csff_legend,
        )

        # Plot universal connected spectral form factor
        ax.plot(
            times,
            csff_universal,
            color=config.surmise_color,
            zorder=config.surmise_zorder,
            linewidth=config.density_width,
            label=config.csff_theory_legend,
        )

        # Add vertical line at Thouless time
        ax.axvline(
            x=thouless_time,
            color=config.thouless_color,
            linewidth=config.thouless_width,
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

        # Set title, labels, and x-axis limits
        ax.set_title(f"{self.ensemble} {config.form_factors_title}")
        ax.set_xlabel(config.form_factors_xlabel)
        ax.set_ylabel(config.form_factors_ylabel)
        ax.set_xlim(10**config.logtime_min, 10**config.logtime_max)

        # Calculate y-axis limits
        if csff_universal[0] != 1.0:
            logy_lower = 1.1 * log(csff_universal[0]) - 0.1 * log(self.ensemble.dim)
            logy_upper = -0.35 * log(csff_universal[0]) + 1.35 * log(self.ensemble.dim)
        else:
            logy_lower = -1.4 * log(self.ensemble.dim)
            logy_upper = 1.4 * log(self.ensemble.dim)

        # Set y-axis limits
        ax.set_ylim([exp(logy_lower), exp(logy_upper)])

        # Shade region after Thouless time
        ax.fill_between(
            times,
            exp(logy_lower),
            exp(logy_upper),
            where=times > thouless_time,
            color=config.chaos_color,
            alpha=config.chaos_alpha,
            zorder=config.chaos_zorder,
        )

        # Shade region after Heisenberg time
        ax.fill_between(
            times,
            exp(logy_lower),
            exp(logy_upper),
            where=times >= 1e0,
            color=config.thermalization_color,
            zorder=config.thermalization_zorder,
            alpha=config.thermalization_alpha,
        )

        # Add legend
        ax.legend(framealpha=1.0)

        # Add text box with Thouless time
        ax.text(
            config.x_position * thouless_time,
            config.y_position * exp(logy_lower),
            rf"{config.thouless_legend} {log10(thouless_time):.2f}",
            fontsize=config.fontsize,
            color="Black",
            bbox=dict(
                facecolor="white", edgecolor=config.edgecolor, boxstyle="round,pad=0.5"
            ),
            zorder=config.thouless_zorder,
        )

        # Create results directory to store plot
        plot_dir = self._create_res_dir(res_type="plots")

        # Save plot to file
        fig.savefig(os.path.join(plot_dir, config.form_factors_plot_filename), dpi=300)

        # Close plot
        plt.close(fig)

    def run(self):
        # Start timer
        start_time = time()

        # Create worker arguments
        worker_args = self._create_worker_args()

        # Run workers
        with Pool(processes=self.workers) as pool:
            eigenvalues = np.vstack(pool.map(self._worker_func, worker_args))

        # Calculate spectral histogram
        self._calc_spectral_histogram(eigenvalues)

        # Unfold eigenvalues
        for r in range(self.realizs):
            eigenvalues[r, :] = self.ensemble.unfold(eigenvalues[r, :])

        # Calculate spacing distribution and spectral form factors
        self._calc_spacing_distribution(eigenvalues)
        self._calc_spectral_form_factors(eigenvalues)

        # Plot results
        self._plot_spectral_histogram()
        self._plot_spacing_distribution()
        self._plot_spectral_form_factors()

        # Stop timer and calculate elapsed time
        sim_time = time() - start_time

        # Print simulation time
        print(f"Simulation time: {sim_time:.2f} seconds")


# =============================
# 3. Main Function
# =============================
def main():
    # Create argument parser
    parser = ArgumentParser(description="Spectral Statistics Monte Carlo")

    # Retrieve Monte Carlo arguments
    mc_args = SpectralStatistics._get_mc_args(parser)

    # Initialize spectral statistics simulation class
    mc = SpectralStatistics(**mc_args)

    # Run spectral statistics simulation
    mc.run()


# Run the main function
if __name__ == "__main__":
    main()
