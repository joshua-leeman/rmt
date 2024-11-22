# rmt.config.time_delay_statistics_config.py
"""
This module contains the data class related to the time delay statistics
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
class TimeDelayStatisticsConfig:
    # Simulation parameters
    energy_num: int = 1000
    energy_min: float = -2.0
    energy_max: float = 2.0
    logtime_min: int = -5
    logtime_max: int = 2

    # Filenames for data and plots
    proper_times_data_filename: str = "proper_times.npz"
    wigner_times_data_filename: str = "wigner_times.npz"
    proper_times_plot_filename: str = "proper_times.png"
    wigner_times_plot_filename: str = "wigner_times.png"


# Instantiate time delay statistics configuration
config = TimeDelayStatisticsConfig()

# Set matplotlib rcParams for plots
rcParams["text.usetex"] = True
rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Latin Modern Roman"
rcParams["font.size"] = 14
