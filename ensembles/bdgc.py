# rmt.ensembles.bdgc.py
"""
This module contains the programs defining the BdG(C) ensemble.
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
class_name = "BdGC"

# Dyson index
beta = 2


# The Bogoliubov-de Gennes (C) Ensemble
class BdGC(Tenfold):
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
        # Compute block dimension
        dim = self.dim // 2

        # Preallocate memory for BdG(C) matrix
        H = np.empty((self.dim, self.dim), dtype=self.dtype)

        # Generate GUE in top-left block
        H[:dim, :dim].real = self._rng.standard_normal(
            (dim, dim), dtype=self.real_dtype
        )
        H[:dim, :dim].imag = self._rng.standard_normal(
            (dim, dim), dtype=self.real_dtype
        )
        np.add(
            H[:dim, :dim],
            H[:dim, :dim].T.conj(),
            out=H[:dim, :dim],
        )

        # Generate complex symmetric top-right block
        H[:dim, dim:].real = self._rng.standard_normal(
            (dim, dim), dtype=self.real_dtype
        )
        H[:dim, dim:].imag = self._rng.standard_normal(
            (dim, dim), dtype=self.real_dtype
        )
        np.add(
            H[:dim, dim:],
            H[:dim, dim:].T,
            out=H[:dim, dim:],
        )

        # Write bottom-left block with complex conjugate of top-right block
        np.conjugate(H[:dim, dim:], out=H[dim:, :dim])

        # Write bottom-right block with negative complex conjugate of top-left block
        np.conjugate(H[:dim, :dim], out=H[dim:, dim:])
        np.negative(H[dim:, dim:], out=H[dim:, dim:])

        # Scale matrix in-place
        H *= self.sigma / 2

        # Return BdG(C) matrix
        return H
