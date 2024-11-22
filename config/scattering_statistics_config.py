# rmt.config.scattering_statistics_config.py
"""
This module contains the data class related to the scattering statistics
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
class ScatteringStatisticsConfig:
    # Simulation parameters
    energy_num: int = 1001
    energy_min: float = -1e-1
    energy_max: float = 1e-1
    SS_conn_min: float = -1e-2
    SS_conn_max: float = 1e-2

    # Filenames for data and plots
    scattering_data_filename: str = "scattering.npz"
    transmission_plot_filename: str = "transmission.png"
    SS_conn_plot_filename: str = "SS_conn.png"

    # Plot titles and labels
    SS_conn_xlabel: str = r"Unfolded Energy Difference $\varepsilon$"


# Instantiate scattering statistics configuration
config = ScatteringStatisticsConfig()

# Set matplotlib rcParams for plots
rcParams["text.usetex"] = True
rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Latin Modern Roman"
rcParams["font.size"] = 14
