# rmt.cluster.jobs.run.__main__.py
"""
This module contains the main function to send job scripts to the SLURM cluster.
It is grouped into the following sections:
    1. Imports
    2. Main Function
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
import os
import sys


# =============================
# 2. Main Function
# =============================
def main():
    # Print program message
    print("\nSending job scripts to SLURM...\n")

    # Retrieve project path
    project_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # Add project path to system path
    sys.path.append(project_path)

    # Retrieve list of jobs from jobs directory
    jobs = [job for job in os.listdir(f"{project_path}/jobs") if job.endswith(".sh")]

    # Loop through jobs submitting each to SLURM
    for job in jobs:
        # Submit job to SLURM
        os.system(f"sbatch {project_path}/jobs/{job}")

    # Print completion message
    print("\nJob scripts have been sent to SLURM.\n")


if __name__ == "__main__":
    # Run main function
    main()
