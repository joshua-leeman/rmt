# rmt.config.spectral_statistics_config.py
"""
This module contains the data class related to the spectral statistics
simulation program.
It is grouped into the following sections:
    1. Imports
    2. Data Class
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
from dataclasses import dataclass
from typing import Tuple

# Third-party imports
from matplotlib.pyplot import rcParams


# =============================
# 2. Data Class
# =============================
@dataclass
class SpectralStatisticsConfig:
    # Simulation parameters
    density_num: int = 1000
    logtime_num: int = 5000
    num_batches: int = 500
    logtime_min: int = -5
    logtime_max: int = 2
    bin_width: float = 0.02
    spectral_bound: float = 1.2  # factor of scale
    spacings_max: float = 4.0
    sffs_epsilon: float = 0.05
    batch_jump: int = 10

    # Filenames for data and plots
    spectral_data_filename: str = "spectrum.npz"
    spectral_plot_filename: str = "spectrum.png"
    spacings_data_filename: str = "spacings.npz"
    spacings_plot_filename: str = "spacings.png"
    form_factors_data_filename: str = "form_factors.npz"
    form_factors_plot_filename: str = "form_factors.png"

    # Colors for plot lines and histograms
    hist_color: str = "RoyalBlue"
    density_color: str = "Red"
    sff_color: str = "Red"
    csff_color: str = "Blue"
    surmise_color: str = "Black"
    thouless_color: str = "Black"
    heisenberg_color: str = "Black"
    chaos_color: str = "Orange"
    thermalization_color: str = "Orangered"

    # Plot titles and labels
    spectrum_title: str = "Average Spectral Density"
    spectrum_xlabel: str = r"Energy $x = E / \lambda$"
    spectrum_ylabel: str = r"Density $\rho(x)$"
    spacings_title: str = "NNS Distribution"
    spacings_xlabel: str = r"Unfolded Spacing $s$"
    spacings_ylabel: str = r"Density $\rho(s)$"
    form_factors_title: str = "Spectral Form Factors"
    form_factors_xlabel: str = r"Time $\tau = t / t_{\textrm{\small H}}$"
    form_factors_ylabel: str = r"Form Factor $K(\tau)$"
    thouless_legend: str = rf"$\log_{{\small 10}} \tau_{{\textrm{{\small th}}}} = $"

    # Plot legends
    hist_legend: str = "simulation"
    spectral_legend: str = "theory"
    spacings_legend: str = "simulation"
    surmise_legend: str = "surmise"
    sff_legend: str = r"$\textrm{SFF}$"
    csff_legend: str = r"$\textrm{cSFF}$"
    csff_theory_legend: str = "theory"

    # Thouless time legend
    x_position: float = 1.8  # x-position of legend w.r.t. thouless time
    y_position: float = 50  # y-position of legend w.r.t. y_min
    edgecolor: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    fontsize: int = 14

    # Plot alphas, widths, and styles
    hist_alpha: float = 0.86
    sff_alpha: float = 0.8
    csff_alpha: float = 1.0
    axes_width: float = 1.2
    chaos_alpha: float = 0.2
    heisenberg_alpha: float = 1.0
    thermalization_alpha: float = 0.2
    density_width: float = 1.65
    sff_width: float = 1.2
    thouless_width: float = 1.2
    heisenberg_width: float = 1.2

    # Zorder for plot elements
    hist_zorder: int = 2
    spectral_zorder: int = 3
    surmise_zorder: int = 4
    chaos_zorder: int = 2
    thermalization_zorder: int = 2
    thouless_zorder: int = 5
    heisenberg_zorder: int = 5
    csff_zorder: int = 3
    sff_zorder: int = 4


# Instantiate spectral statistics configuration
config = SpectralStatisticsConfig()

# Set matplotlib rcParams for plots
rcParams["text.usetex"] = True
rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Latin Modern Roman"
rcParams["font.size"] = 14
