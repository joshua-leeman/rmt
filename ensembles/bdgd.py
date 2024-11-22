# rmt.ensembles.bdgd.py
"""
This module contains the programs defining the BdG(D) ensemble.
It is grouped into the following sections:
    1. Imports
    2. Ensemble Class
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
from math import sqrt

# Third-party imports
import numpy as np

# Local application imports
from ._rmt import Tenfold


# =============================
# 2. Ensemble Class
# =============================
# Class name for dynamic imports
class_name = "BdGD"

# Dyson index
beta = 2


# The Bogoliubov-de Gennes (D) Ensemble
class BdGD(Tenfold):
    def __init__(
        self,
        N: int = None,
        dim: int = None,
        scale: float = 1.0,
        dtype: type = np.complex64,
    ) -> None:
        # Initialize tenfold ensemble
        super().__init__(beta=beta, N=N, dim=dim, scale=scale, dtype=dtype)

    def matrix(self) -> np.ndarray:
        # Preallocate memory for BdG(D) matrix
        H = np.empty((self.dim, self.dim), dtype=self.dtype)

        # Generate zeros for real parts and standard normal entries for imaginary parts
        H.real = np.zeros((self.dim, self.dim), dtype=self.real_dtype)
        H.imag = self._rng.standard_normal((self.dim, self.dim), dtype=self.real_dtype)

        # Anti-symmetrize matrix in-place
        np.subtract(H, H.T.conj(), out=H)

        # Scale matrix in-place
        H *= self.sigma / sqrt(2)

        # Return BdG(D) matrix
        return H
