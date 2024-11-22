# rmt.ensembles.goe.py
"""
This module contains the programs defining the GOE ensemble.
It is grouped into the following sections:
    1. Imports
    2. Ensemble Class
"""


# =============================
# 1. Imports
# =============================
# Third-party imports
import numpy as np

# Local application imports
from ._rmt import Tenfold


# =============================
# 2. Ensemble Class
# =============================
# Class name for dynamic imports
class_name = "GOE"

# Dyson index
beta = 1


# The Gaussian Orthogonal Ensemble
class GOE(Tenfold):
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
        # Allocate memory for GOE matrix
        H = np.empty((self.dim, self.dim), dtype=self.dtype)

        # Generate standard normal numbers for real parts and zeros for imaginary parts
        H.real = self._rng.standard_normal((self.dim, self.dim), dtype=self.real_dtype)
        H.imag = np.zeros((self.dim, self.dim), dtype=self.real_dtype)

        # Symmetrize matrix in-place
        np.add(H, H.T, out=H)

        # Scale matrix in-place
        H *= self.sigma / 2

        # Return GOE matrix
        return H
