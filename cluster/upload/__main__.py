# rmt.cluster.upload.__main__.py
"""
This module contains the main function to upload repository to the cluster.
It is grouped into the following sections:
    1. Imports
    2. Main Function
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
from os import environ, system
from os.path import abspath, dirname
from sys import path

# Third-party imports
from dotenv import load_dotenv


# =============================
# 2. Main Function
# =============================
def main():
    # Print program message
    print("\nUploading repository to cluster...\n")

    # Retrieve project path
    project_path = dirname(dirname(dirname(abspath(__file__))))

    # Add project path to system path
    path.append(project_path)

    # Load environment variables
    load_dotenv()

    # Store cluster address
    cluster_address = environ.get("CLUSTER_ADDRESS")

    # Create command to upload repository to cluster
    command = (
        f"rsync -av --filter=':- .gitignore' --exclude='.git' ./ {cluster_address}:rmt"
    )

    # Print validation message
    print("Checking credentials...\n")

    # Run command to upload repository to cluster
    system(command)

    # Print completion message
    print("\nRepository has been uploaded to cluster.")


if __name__ == "__main__":
    # Run main function
    main()
