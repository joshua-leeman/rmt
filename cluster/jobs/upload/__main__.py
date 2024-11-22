# rmt.cluster.upload.__main__.py
"""
This module contains the main function to upload SLURM scripts to the cluster.
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

# Third-party imports
from dotenv import load_dotenv


# =============================
# 2. Main Function
# =============================
def main():
    # Print program message
    print("\nUploading jobs to cluster...\n")

    # Retrieve project path
    project_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # Add project path to system path
    sys.path.append(project_path)

    # Load environment variables
    load_dotenv()

    # Store cluster address
    cluster_address = os.environ.get("CLUSTER_ADDRESS")

    # Create command to upload repository to cluster
    command = f"scp -r cluster/jobs/*.sh {cluster_address}:rmt/cluster/jobs"

    # Print validation message
    print("Checking credentials...\n")

    # Run command to upload repository to cluster
    os.system(command)

    # Print completion message
    print("\nSLURM jobs have been uploaded to cluster.")


if __name__ == "__main__":
    # Run main function
    main()
