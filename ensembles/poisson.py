# rmt.ensembles.poisson.py
"""
This module contains the programs defining the Poisson ensemble.
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
from ._rmt import RMT


# =============================
# 2. Ensemble Class
# =============================
# Class name for dynamic imports
class_name = "Poisson"

# Dyson index
beta = 0


# The Poisson Ensemble
class Poisson(RMT):
    def __init__(
        self,
        N: int = None,
        dim: int = None,
        scale: float = 1.0,
        dtype: type = np.complex64,
    ) -> None:
        # Initialize RMT ensemble
        super().__init__(N=N, dim=dim, scale=scale, dtype=dtype)

        # Set Dyson index
        self._beta = beta

    def matrix(self) -> np.ndarray:
        # Generate random eigenvalues for Poisson matrix
        H = self._rng.random(self.dim, dtype=np.float32)

        # Scale and shift eigenvalues
        H *= 2 * self.scale
        H -= self.scale

        # Return Poisson matrix
        return np.diag(H).astype(self.dtype)

    def mean_density(self, E: float) -> float:
        # Return uniform density within support
        if np.abs(E) <= self.scale:
            return 1 / (2 * self.scale)
        else:
            return 0.0
