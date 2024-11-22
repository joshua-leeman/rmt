# rmt.cluster.jobs.delete.__main__.py
"""
This module contains the main function to delete job scripts.
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
    # Retrieve project path
    project_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # Add project path to system path
    sys.path.append(project_path)

    # Retrieve list of jobs from jobs directory
    jobs = [job for job in os.listdir(f"{project_path}/jobs") if ".sh" in job]

    # Loop through jobs deleting each
    for job in jobs:
        # Delete job script
        os.system(f"rm {project_path}/jobs/{job}")

    # Print completion message
    print("\nJob scripts have been deleted.\n")


if __name__ == "__main__":
    # Run main function
    main()
