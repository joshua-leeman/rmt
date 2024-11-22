# rmt.simulations._mc.py
"""
This module contains classes and functions related to the default Monte Carlo
simulation program.
It is grouped into the following sections:
    1. Imports
    2. Monte Carlo Class
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
import os
import re
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from ast import literal_eval
from importlib import import_module
from textwrap import dedent
from time import strftime

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from psutil import cpu_count, virtual_memory


# =============================
# 2. Monte Carlo Class
# =============================
class MonteCarlo(ABC):
    def __init__(self, ens_input: dict, sim_input: dict = {}, spec: dict = {}):
        # Clean ensemble name
        ens_input["name"] = re.sub(r"\W+", "", ens_input["name"]).strip().lower()

        # Copy ensemble input and pop name
        ens_args = ens_input.copy()
        ens_args.pop("name")

        # Store system parameters
        self._max_workers = cpu_count()
        self._max_memory = virtual_memory().total  # in bytes

        # Set job specifications
        self._workers = int(spec.get("workers", 1))
        self._memory = int(
            spec.get("memory", self.max_memory / 2**30) * 2**30
        )  # in bytes

        # Initialize ensemble
        module = import_module(f"ensembles.{ens_input['name']}")
        ENS = getattr(module, module.class_name)
        self._ensemble = ENS(**ens_args)

        # Reorder ensemble input and store
        self._ens_input = {
            key: ens_input[key] for key in self.ensemble._order if key in ens_input
        }

        # Set default ensemble name
        self._name = self.__class__.__name__

        # Store simulation input and update with name
        self._sim_input = sim_input
        self._sim_input["name"] = str(self)

        # Store realizations
        self._realizs = int(sim_input.get("realizs", 1))

        # Calculate number of usable workers
        self._workers = min(self.workers, self.memory // self.mem_per_calc)

        # Set project path
        self._project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Check if Monte Carlo parameters are valid
        self._check_mc()

        # Store date and time of simulation
        self._date_time = strftime("%Y-%m-%d %H:%M:%S")
        self._dir_time = self.date_time.replace(" ", "_").replace(":", "-")

    def __repr__(self):
        # Return string representation
        return f"{self._name}(ensemble={self.ensemble}, realizs={self.realizs})"

    def __str__(self):
        # Replace underscores with spaces
        string = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", self._name)
        string = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", string)

        # Convert to lowercase and return
        return string.lower()

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def _get_mc_args(parser: ArgumentParser) -> dict:
        # Add ensemble argument
        parser.add_argument(
            "-ens",
            "--ensemble",
            type=str,
            required=True,
            help="random matrix ensemble to simulate in JSON (required)",
        )

        # Add simulation argument(s)
        parser.add_argument(
            "-args",
            "--arguments",
            type=str,
            default="{}",
            help="simulation arguments in JSON (default: {'realizs': 1})",
        )

        # Add job configuration arguments
        parser.add_argument(
            "-spec",
            "--specification",
            type=str,
            help=f"job specification in JSON (default: {{'workers': 1, 'memory': {virtual_memory().total // 2**30} [GB]}})",
        )

        # Parse arguments into dictionary
        dic = vars(parser.parse_args())

        # Initialize output dictionary
        mc_args = {}

        # Convert ensemble argument to dictionary
        mc_args["ens_input"] = literal_eval(dic["ensemble"])

        # Convert simulation argument to dictionary
        mc_args["sim_input"] = literal_eval(dic["arguments"])

        # Convert configuration argument to dictionary
        mc_args["spec"] = literal_eval(dic["specification"])

        # Return dictionary of Monte Carlo arguments
        return mc_args

    def _check_mc(self):
        # Retrieve list of valid ensembles
        ens_list = [
            file.rstrip(".py")
            for file in os.listdir(f"{self.project_path}/ensembles")
            if file.endswith(".py") and not file.startswith("_")
        ]

        # Check if ensemble is valid
        if self._ens_input["name"] not in ens_list:
            raise ValueError(f"Ensemble must be one of the following:\n", *ens_list)

        # Check if number of workers is valid
        if (
            not isinstance(self.workers, (int, float))
            or self.workers < 1
            or self.workers > self.max_workers
            or self.workers != int(self.workers)
        ):
            raise ValueError(
                "Number of workers must be a positive integer less than or equal to the number of CPUs."
            )

        # Check if memory is valid
        if (
            not isinstance(self.memory, (int, float))
            or self.memory < 1
            or self.memory > self.max_memory
        ):
            raise ValueError(
                "Memory must be a positive number less than or equal to the total system virtual memory."
            )

        # Check if number of realizations is valid
        if (
            not isinstance(self.realizs, (int, float))
            or self.realizs < 1
            or self.realizs != int(self.realizs)
        ):
            raise ValueError("Number of realizations must be a positive integer.")

        # Check if provided memory is sufficient
        if self._memory < self.mem_per_calc:
            raise ValueError(
                dedent(
                    f"""
                    Provided memory is insufficient for calculations:
                    Memory Required: {self.mem_per_calc // 2**30} GB
                    Memory Provided: {self.memory // 2**30} GB
                    """
                )
            )

    def _create_worker_args(self):
        # Initialize list of worker arguments
        worker_args = [None for _ in range(self._workers)]

        # Loop over workers
        for i in range(self._workers):
            # Initialize worker arguments
            worker_args[i] = {
                "id": i,
                "num_workers": self.workers,
                "ens_input": self._ens_input,
                "sim_input": self._sim_input,
            }

        # Return list of worker arguments
        return worker_args

    def _create_res_dir(self, res_type: str = "") -> str:
        # Construct results directory path
        res_dir = f"{self.project_path}/res/{str(self)}/{self._ens_input['name']}/"
        res_dir += "/".join(
            f"{key}_{val}" for key, val in self._ens_input.items() if key != "name"
        )
        res_dir += f"/realizs={self.realizs}/{self._dir_time}/{res_type}"

        # Create directory if it does not exist
        os.makedirs(res_dir, exist_ok=True)

        # Return results directory path
        return res_dir

    def _retrieve_mean_spacing(self):
        # Check if corresponding spectral statistics simulation is present
        spe_dir = f"/res/spectral_statistics/{self._ens_input['name']}/"
        spe_dir += "/".join(
            f"{key}_{val}" for key, val in self._ens_input.items() if key != "name"
        )

        # If present, recursively search all subdirectories and retrieve Thouless time
        if os.path.exists(f"{self.project_path}{spe_dir}"):
            for root, _, files in os.walk(f"{self.project_path}{spe_dir}"):
                for file in files:
                    if file == "spectrum.npz":
                        return np.load(os.path.join(root, file))["mean_spacing"]

    def _retrieve_thouless_time(self):
        # Check if corresponding spectral statistics simulation is present
        spe_dir = f"/res/spectral_statistics/{self._ens_input['name']}/"
        spe_dir += "/".join(
            f"{key}_{val}" for key, val in self._ens_input.items() if key != "name"
        )

        # If present, recursively search all subdirectories and retrieve Thouless time
        if os.path.exists(f"{self.project_path}{spe_dir}"):
            for root, _, files in os.walk(f"{self.project_path}{spe_dir}"):
                for file in files:
                    if file == "form_factors.npz":
                        return np.load(os.path.join(root, file))["thouless_time"]

    def _initialize_plot(self):
        # Initialize plt.subplots() figure and axis
        fig, ax = plt.subplots()

        # Set line widths
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        # Add grid to plot
        ax.grid(True, color="black", zorder=1, linewidth=0.3, alpha=0.225)
        ax.minorticks_on()
        ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.5)

        # Return figure and axis
        return fig, ax

    @property
    def ensemble(self):
        return self._ensemble

    @property
    def realizs(self):
        return self._realizs

    @property
    def workers(self):
        return self._workers

    @property
    def max_workers(self):
        return self._max_workers

    @property
    def memory(self):
        return self._memory

    @property
    def max_memory(self):
        return self._max_memory

    @property
    def mem_per_calc(self):
        return self.ensemble._mem_per_calc

    @property
    def project_path(self):
        return self._project_path

    @property
    def date_time(self):
        return self._date_time
