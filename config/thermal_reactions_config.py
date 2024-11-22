# rmt.config.thermal_reactions_config.py
"""
This module contains the data class related to the thermal reactions
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
class ThermalReactionsConfig:
    # Simulation parameters
    energy_num: int = 10000
    energy_min: float = -2.5
    energy_max: float = 2.5
    energy_xlim_min: float = -2.0
    energy_xlim_max: float = 2.0
    transmission_ymin: float = 0.0
    transmission_ymax: float = 1.05
    entropy_ymin: float = 0.0
    entropy_ymax: float = 1.05
    prob_ymin: float = 1e-3
    prob_ymax: float = 1e1
    temperature_ymin: float = -1e3
    temperature_ymax: float = 1e3
    free_energy_ymin: float = -1e2
    free_energy_ymax: float = 1e2
    heat_capacity_ymin: float = -1e2
    heat_capacity_ymax: float = 1e2

    # Filenames for data and plots
    transmission_data_filename: str = "transmission.npz"
    transmission_plot_filename: str = "transmission.png"
    entropy_data_filename: str = "entropy.npz"
    entropy_plot_filename: str = "entropy.png"
    probabilities_data_filename: str = "probabilities.npz"
    probabilities_plot_filename: str = "probabilities.png"
    temperature_data_filename: str = "temperature.npz"
    temperature_plot_filename: str = "temperature.png"
    free_energy_data_filename: str = "free_energy.npz"
    free_energy_plot_filename: str = "free_energy.png"
    heat_capacity_data_filename: str = "heat_capacity.npz"
    heat_capacity_plot_filename: str = "heat_capacity.png"

    # Plot titles and labels
    transmission_title: str = "Transmission Coefficients"
    transmission_xlabel: str = r"Energy $x = E / \lambda$"
    transmission_ylabel: str = r"Transmission $T(x)$"
    entropy_title: str = "Entropy"
    entropy_xlabel: str = r"Energy $x = E / \lambda$"
    entropy_ylabel: str = r"Entropy $\mathcal{S}(x) / \ln \Lambda$"
    probabilities_title: str = "Channel Probabilities"
    probabilities_xlabel: str = r"Energy $x = E / \lambda$"
    probabilities_ylabel: str = r"Probability $p(x)$"
    temperature_title: str = "Temperature"
    temperature_xlabel: str = r"Energy $x = E / \lambda$"
    temperature_ylabel: str = r"Temperature $\mathcal{T}(x)$"
    free_energy_title: str = "Free Energy"
    free_energy_xlabel: str = r"Energy $x = E / \lambda$"
    free_energy_ylabel: str = r"Free Energy $\mathcal{F}(x)$"
    heat_capacity_title: str = "Heat Capacity"
    heat_capacity_xlabel: str = r"Energy $x = E / \lambda$"
    heat_capacity_ylabel: str = r"Heat Capacity $\mathcal{C}(x)$"
    hist_ylabel: str = r"Probability Density $\rho(x)$"

    # Colors for plot lines
    initial_transmission_color: str = "Red"
    other_transmission_color: str = "Blue"
    entropy_color: str = "Red"
    initial_probability_color: str = "Red"
    other_probability_color: str = "Blue"
    temperature_color: str = "Red"
    free_energy_color: str = "Red"
    heat_capacity_color: str = "Red"
    spectral_color: str = "Orange"
    hist_color: str = "RoyalBlue"

    # Plot legends
    initial_transmission_legend: str = r"$T_0$"
    other_transmission_legend: str = r"$T_{i \neq 0}$"
    initial_probability_legend: str = r"$p_0$"
    other_probability_legend: str = r"$p_{i \neq 0}$"
    hist_legend: str = "simulation"
    spectral_legend: str = r"$\rho(x)$"

    # Plot alphas, widths, and styles
    initial_transmission_alpha: float = 1.0
    other_transmission_alpha: float = 1.0
    entropy_alpha: float = 1.0
    initial_probability_alpha: float = 1.0
    other_probability_alpha: float = 1.0
    temperature_alpha: float = 1.0
    free_energy_alpha: float = 1.0
    heat_capacity_alpha: float = 1.0
    spectral_alpha: float = 0.8
    hist_alpha: float = 0.50
    axes_width: float = 1.2
    transmission_width: float = 1.85
    entropy_width: float = 1.85
    probability_width: float = 1.85
    temperature_width: float = 1.85
    free_energy_width: float = 1.85
    heat_capacity_width: float = 1.85
    spectral_width: float = 1.25

    # Zorder for plot elements
    hist_zorder: int = 0
    spectral_zorder: int = 1
    other_transmission_zorder: int = 2
    initial_transmission_zorder: int = 3
    entropy_zorder: int = 2
    other_probability_zorder: int = 2
    initial_probability_zorder: int = 3
    temperature_zorder: int = 2
    free_energy_zorder: int = 2
    heat_capacity_zorder: int = 2


# Instantiate thermal reactions configuration
config = ThermalReactionsConfig()

# Set matplotlib rcParams for plots
rcParams["text.usetex"] = True
rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Latin Modern Roman"
rcParams["font.size"] = 14
