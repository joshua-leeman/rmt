# rmt.cluster.jobs.create.__main__.py
"""
This module contains the main function to create jobs for the cluster.
It is grouped into the following sections:
    1. Imports
    2. Attributes & Functions
    3. Main Function
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
import os
import re
from argparse import ArgumentParser
from ast import literal_eval
from itertools import product
from textwrap import dedent

# Third-party imports
from dotenv import load_dotenv


# =============================
# 2. Attributes & Functions
# =============================
# Define dictonary of valid queues
queues = {
    # > 16 nodes, time <= 8 hours
    "large-40core": {"ntasks": "40", "memory": "192", "time": "8:00:00", "nodes": "50"},
    "hbm-large-96core": {
        "ntasks": "96",
        "memory": "384",
        "time": "8:00:00",
        "nodes": "38",
    },
    "large-96core": {"ntasks": "96", "memory": "256", "time": "8:00:00", "nodes": "38"},
    # 2 days < time <= 7 days
    "extended-40core": {
        "ntasks": "40",
        "memory": "192",
        "time": "7-00:00:00",
        "nodes": "2",
    },
    "extended-40core-shared": {
        "ntasks": "40",
        "memory": "192",
        "time": "3-12:00:00",
        "nodes": "1",
    },
    "hbm-extended-96core": {
        "ntasks": "96",
        "memory": "384",
        "time": "7-00:00:00",
        "nodes": "2",
    },
    "extended-96core": {
        "ntasks": "96",
        "memory": "256",
        "time": "7-00:00:00",
        "nodes": "2",
    },
    "extended-96core-shared": {
        "ntasks": "96",
        "memory": "256",
        "time": "3-12:00:00",
        "nodes": "1",
    },
    # 4 hours < time <= 2 days
    "long-40core": {
        "ntasks": "40",
        "memory": "256",
        "time": "2-00:00:00",
        "nodes": "6",
    },
    "long-40core-shared": {
        "ntasks": "40",
        "memory": "256",
        "time": "1-00:00:00",
        "nodes": "3",
    },
    "hbm-long-96core": {
        "ntasks": "96",
        "memory": "384",
        "time": "2-00:00:00",
        "nodes": "6",
    },
    "hbm-1tb-long-96core": {
        "ntasks": "96",
        "memory": "1000",
        "time": "2-00:00:00",
        "nodes": "1",
    },
    "long-96core": {
        "ntasks": "96",
        "memory": "256",
        "time": "2-00:00:00",
        "nodes": "6",
    },
    "long-96core-shared": {
        "ntasks": "96",
        "memory": "256",
        "time": "1-00:00:00",
        "nodes": "3",
    },
    # time <= 4 hours
    "short-40core": {"ntasks": "40", "memory": "192", "time": "4:00:00", "nodes": "8"},
    "short-40core-shared": {
        "ntasks": "40",
        "memory": "192",
        "time": "4:00:00",
        "nodes": "4",
    },
    "hbm-short-96core": {
        "ntasks": "96",
        "memory": "384",
        "time": "4:00:00",
        "nodes": "8",
    },
    "short-96core": {"ntasks": "96", "memory": "256", "time": "4:00:00", "nodes": "8"},
    "short-96core-shared": {
        "ntasks": "96",
        "memory": "256",
        "time": "4:00:00",
        "nodes": "4",
    },
}


def get_args() -> dict:
    """
    Get the arguments from the command line and return them.

    Returns:
        dict: arguments
    """
    # Create argument parser
    parser = ArgumentParser(description="Automate creation of jobs for slurm cluster")

    # Add simulation argument
    parser.add_argument(
        "-sim",
        "--simulation",
        type=str,
        nargs="+",
        required=True,
        help="simulations (required)",
    )

    # Add ensemble argument
    parser.add_argument(
        "-ens",
        "--ensemble",
        type=str,
        nargs="+",
        required=True,
        help="ensemble (required)",
    )

    # Add queue argument
    parser.add_argument("--queue", type=str, required=True, help="queue (required)")

    # Parse arguments and return them as a dictionary
    return vars(parser.parse_args())


def main():
    # Retrieve keyword arguments from command line
    kwargs = get_args()

    # Extract keys and values from kwargs
    keys = list(kwargs.keys())
    values = list(kwargs.values())

    # Convert all values to lists
    values_lists = [v if isinstance(v, list) else [v] for v in values]

    # Create list of job configurations
    jobs = [dict(zip(keys, v)) for v in product(*values_lists)]

    # Load environment variables
    load_dotenv()

    # Create jobs
    for job in jobs:
        # Unpack job-specific keyword arguments
        sim = literal_eval(job["simulation"])
        ens = literal_eval(job["ensemble"])
        queue = job["queue"]

        # Check if queue is valid
        if queue not in queues:
            raise ValueError(f"Queue '{queue}' is not valid.")

        # Retrieve simulation and ensemble names
        sim_name = sim.pop("name").strip().lower()

        # Create job configuration dictionary
        config = {"workers": queues[queue]["ntasks"]}

        # Create job file content
        file_content = f"""\
        #!/bin/bash
        #
        #SBATCH --job-name={sim_name[:4]}{re.sub(r'[^a-zA-Z0-9]+', '_', str(ens))}r{sim["realizations"]}_{queue[:-4]}
        #SBATCH --output=errs/{sim_name[:4]}{re.sub(r'[^a-zA-Z0-9]+', '_', str(ens))}r{sim["realizations"]}_{queue[:-4]}.txt
        #SBATCH --cpus-per-task={queues[queue]["ntasks"]}
        #SBATCH --nodes=1
        #SBATCH --ntasks-per-node=1
        #SBATCH --time={queues[queue]["time"]}
        #SBATCH --partition={queue}
        #SBATCH --mail-type=BEGIN,END
        #SBATCH --mail-user={os.environ.get("EMAIL_ADDRESS")}
        
        # Load modules
        module load anaconda
        module load texlive
        
        # Load conda environment
        conda activate rmt-env
        
        # Run simulation
        python -m simulations.{sim_name} --ensemble \"{ens}\" --arguments \"{sim}\" --configuration \"{config}\"
        
        # Deactivate conda environment
        conda deactivate
        
        """

        # Dedent file content and remove leading/trailing whitespace
        file_content = dedent(file_content)

        # Save job to file
        with open(
            f"cluster/jobs/{sim_name[:4]}{re.sub(r'[^a-zA-Z0-9]+', '_', str(ens))}r{sim["realizations"]}_{queue[:-4]}.sh",
            "w",
        ) as file:
            file.write(file_content)


if __name__ == "__main__":
    # Run main function
    main()
