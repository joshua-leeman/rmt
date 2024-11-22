# rmt.inputs.channel_couplings.py
"""
This module contains code to create and store input coupling
matrices for the simulation quantum-chaotic compound reactions.
It is grouped into the following sections:
    1. Imports
    2. Coupling Matrix
    3. Utility Functions
    4. Main Function
"""


# =============================
# 1. Imports
# =============================
# Standard library Imports
import os
from argparse import ArgumentParser

# Third-party imports
import numpy as np


# =============================
# 2. Coupling Matrix
# =============================
W = np.ones(10, dtype=np.complex64) / np.sqrt(2 * np.pi)


# =============================
# 3. Utility Functions
# =============================
def _get_args():
    # Create argument parser
    parser = ArgumentParser(
        description="Create input coupling matrices for the simulation of quantum-chaotic compound reactions."
    )

    # Add simulation input
    parser.add_argument(
        "-s",
        "--simulation",
        type=str,
        default="W.npz",
        help="Name of simulation to run.",
    )

    # Add filename input
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default="W",
        help="Name of .npz to store the coupling matrix.",
    )

    # Return arguments as dictionary
    return vars(parser.parse_args())


def _check_simulation(simulation: str) -> None:
    # Check if simulation is found in inputs directory
    if not os.path.isdir(f"./inputs/{simulation}_inputs"):
        raise FileNotFoundError(
            f"Simulation directory './inputs/{simulation}' not found."
        )


def _store_coupling_matrix(simulation: str, filename: str) -> None:
    # Store coupling matrix to npz file
    np.savez(f"./inputs/{simulation}_inputs/{filename}.npz", W=W)


# =============================
# 4. Main Function
# =============================
def main():
    # Get arguments
    args = _get_args()

    # Check simulation
    _check_simulation(args["simulation"])

    # Store coupling matrix
    _store_coupling_matrix(args["simulation"], args["filename"])

    # Print success message
    print(f"Successfully stored coupling matrix for {args['simulation']} simulation.")


# Run main function
if __name__ == "__main__":
    main()
