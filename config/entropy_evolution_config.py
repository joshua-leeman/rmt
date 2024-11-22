# rmt.config.entropy_evolution_config.py
"""
This module contains the data class related to the entropy evolution
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
class EntropyEvolutionConfig:
    # Simulation parameters
    logtime_num: int = 1000
    logtime_min: int = -5
    logtime_max: int = 2
    entropy_ymin: float = 1e-5
    entropy_ymax: float = 1e3
    prob_ymin: float = 1e-2
    prob_ymax: float = 1e6

    # Filenames for data and plots
    entropy_data_filename: str = "entropy.npz"
    entropy_plot_filename: str = "entropy.png"

    # Plot titles and labels
    entropy_title: str = "Entropy Evolution"
    entropy_xlabel: str = r"Time $\tau = t / t_{\textrm{\small H}}$"
    entropy_ylabel: str = r"Entropy $\mathcal{S}(\tau) / \ln D$"

    # Colors for plot lines
    entropy_color: str = "Black"
    initial_color: str = "Red"
    other_color: str = "Blue"
    thouless_color: str = "Black"
    heisenberg_color: str = "Black"
    chaos_color: str = "Orange"
    thermalization_color: str = "Orangered"

    # Plot legends
    entropy_legend: str = "entropy"
    initial_legend: str = r"$p_0$"
    other_legend: str = r"$p_{i \neq 0}$"
    thouless_legend: str = rf"$\log_{{\small 10}} \tau_{{\textrm{{\small th}}}} = $"

    # Thouless time legend
    x_position: float = 1.8
    y_position: float = 4
    edgecolor: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    fontsize: int = 14

    # Plot alphas, widths, and styles
    entropy_alpha: float = 1.0
    initial_alpha: float = 1.0
    other_alpha: float = 1.0
    other_legend_alpha: float = 1.0
    thouless_alpha: float = 1.0
    heisenberg_alpha: float = 1.0
    chaos_alpha: float = 0.2
    thermalization_alpha: float = 0.2
    entropy_width: float = 1.85
    initial_width: float = 1.2
    other_width: float = 1.2
    thouless_width: float = 1.0
    heisenberg_width: float = 1.0

    # Zorder for plot elements
    chaos_zorder: int = 0
    thermalization_zorder: int = 0
    other_zorder: int = 1
    initial_zorder: int = 2
    entropy_zorder: int = 3
    thouless_zorder: int = 5
    heisenberg_zorder: int = 4


# Instantiate spectral statistics configuration
config = EntropyEvolutionConfig()

# Set matplotlib rcParams for plots
rcParams["text.usetex"] = True
rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Latin Modern Roman"
rcParams["font.size"] = 14
