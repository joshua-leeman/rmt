# rmt.cluster.__main__.py
"""
This module contains the main function to login to the cluster.
It is grouped into the following sections:
    1. Imports
    2. Main Function
"""


# =============================
# 1. Imports
# =============================
import os
import sys

# Third-party imports
from dotenv import load_dotenv


# =============================
# 2. Main Function
# =============================
def main():
    # Retrieve project path
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Add project path to system path
    sys.path.append(project_path)

    # Load environment variables
    load_dotenv()

    # Store cluster address
    cluster_address = os.environ.get("CLUSTER_ADDRESS")

    # Create command to login to cluster
    command = f"ssh -X {cluster_address}"

    # Print extra line for formatting
    print()

    # Run command to login to cluster
    os.system(command)


if __name__ == "__main__":
    # Run main function
    main()
