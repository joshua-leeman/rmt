# rmt.ensembles.gue.py
"""
This module contains the programs defining the GUE ensemble.
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
class_name = "GUE"

# Dyson index
beta = 2


# The Gaussian Unitary Ensemble
class GUE(Tenfold):
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
        # Allocate memory for GUE matrix
        H = np.empty((self.dim, self.dim), dtype=self.dtype)

        # Fill matrix with real and imaginary random Gaussian entries
        H.real = self._rng.standard_normal(H.shape, dtype=self.real_dtype)
        H.imag = self._rng.standard_normal(H.shape, dtype=self.real_dtype)

        # Adjoint matrix in-place
        np.add(H, H.T.conj(), out=H)

        # Scale matrix in-place
        H *= self.sigma / 2

        # Return GUE matrix
        return H
