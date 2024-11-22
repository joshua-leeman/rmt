# rmt.simulations.__init__.py
"""
This __init__ file dynamically imports all simulations from the simulation package.
It is grouped into the following section:
    1. Imports
    2. Set Environment Variables
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
from os import environ


# =============================
# 2. Set Environment Variables
# =============================
# Set number of threads for OpenBLAS
environ["OPENBLAS_NUM_THREADS"] = "1"
