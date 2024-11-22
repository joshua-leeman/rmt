# rmt.ensembles.syk.py
"""
This module contains the programs defining the SYK ensemble.
It is grouped into the following sections:
    1. Imports
    2. Ensemble Class
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
from math import pi, comb, prod, sqrt

# Third-party imports
import numpy as np
from scipy.sparse import csr_matrix, eye_array, kron

# Local application imports
from ._rmt import SparseRMT


# =============================
# 2. Ensemble Class
# =============================
# Class name for dynamic imports
class_name = "SYK"


# The Sachdev-Ye-Kitaev Ensemble
class SYK(SparseRMT):
    def __init__(
        self,
        q: int,
        N: int,
        scale: float = 1.0,
        dtype: type = np.complex64,
    ):
        # Set SYK arguments
        self._q = int(q)
        self._N = int(N)

        # Calculate suppression factor
        self._eta = sum(
            (-1) ** (self.q - k)
            * comb(self.q, k)
            * comb(self.N - self.q, self.q - k)
            / comb(self.N, self.q)
            for k in range(self.q + 1)
        )

        # Initialize RMT ensemble
        super().__init__(N=N, scale=scale, dtype=dtype)

        # Set order of SYK arguments
        self._order = ["name", "q", "N", "scale"]

        # Determine Dyson index
        self._beta = (
            {(0, 0): 1, (0, 4): 4}.get((self.q % 4, self.N % 8), 2) if q > 2 else 0
        )

        # Calculate standard deviation
        self._sigma = self.scale * sqrt((1 - self.eta) / comb(self.N, self.q)) / 2

    def __repr__(self):
        return f"SYK(q={self.q}, N={self.N}, scale={self.scale})"

    def __str__(self):
        return rf"$\textrm{{SYK}}_{{{self.q}}}\ N={self.N}$"

    def matrix(self):
        # Create Majorana operators if not already created
        if not hasattr(self, "_majorana"):
            self._majorana = self._create_majoranas()

        # Initialize indices and products for Majorana operators
        indices = list(range(self.q))
        products = [None for _ in range(self.q)]

        # Fill products with initial products of Majorana operators
        products[0] = self.majorana[indices[0]]
        for i in range(1, self.q):
            products[i] = products[i - 1].dot(self.majorana[indices[i]])

        # Initialize Hamiltonian matrix
        H = csr_matrix((self.dim, self.dim), dtype=np.complex64)

        while True:
            # Add current product to SYK Hamiltonian
            H += (
                self._rng.standard_normal(dtype=np.float32)
                * products[-1][: self.dim, : self.dim]
            )

            # Generate next combination of indices
            for i in reversed(range(self.q)):
                # If index is less than maximum index, increment it, and break
                if indices[i] < self.N - self.q + i:
                    indices[i] += 1
                    for j in range(i + 1, self.q):
                        indices[j] = indices[j - 1] + 1
                    break
            else:
                # All combinations have been processed, break while loop
                break

            # Update product at changed index
            if i == 0:
                products[0] = self.majorana[indices[0]]
            else:
                products[i] = products[i - 1].dot(self.majorana[indices[i]])

            # Update products at indices greater than changed index
            for j in range(i + 1, self.q):
                products[j] = products[j - 1].dot(self.majorana[indices[j]])

        # Scale and return random SYK Hamiltonian matrix
        H *= 1j ** (self.q * (self.q - 1) // 2) * self.sigma
        return H.toarray()

    def mean_density(self, E, num_terms=1000):
        # Check if energy is within support
        if abs(E) < self.scale:
            # Approximate spectral density's infinite product
            product = prod(
                (
                    1
                    - (E / self.scale * 2) ** 2
                    * self.eta ** (k + 1)
                    / (1 + self.eta ** (k + 1)) ** 2
                )
                * ((1 - self.eta ** (2 * k + 2)) / (1 - self.eta ** (2 * k + 1)))
                for k in range(num_terms)
            )

            # Return SYK mean spectral density at energy E
            return product * sqrt(1 - (E / self.scale) ** 2) / (pi * self.scale / 2)
        else:
            return 0.0

    def _check_ens(self) -> None:
        # Check if default ensemble arguments are valid
        super()._check_ens()

        # Check if SYK parameters are valid
        if self.q < 2 or self.q % 2 != 0:
            raise ValueError(
                "SYK q-parameter must be an even integer greater than or equal to 2."
            )
        elif self.N <= self.q:
            raise ValueError(
                "Number of Majorana fermions must be greater than SYK q-parameter."
            )

    def _create_majoranas(self):
        # Create Pauli matrices
        pauli = [
            csr_matrix([[0, 1], [1, 0]], dtype=np.complex64),  # sigma_x
            csr_matrix([[0, -1j], [1j, 0]], dtype=np.complex64),  # sigma_y
            csr_matrix([[1, 0], [0, -1]], dtype=np.complex64),  # sigma_z
        ]

        # Create initial Majorana operators
        majorana_0 = pauli[:2]
        majorana_c0 = pauli[2]

        # With loop, build Majorana operators from the initials
        for i in range(self.N // 2 - 1):
            # Create identity matrix
            eye_mat = eye_array(2 ** (i + 1), format="csr", dtype=np.complex64)

            # Initialize new Majorana operators array
            majorana = [None for _ in range(len(majorana_0) + 2)]

            # Create new Majorana operators
            for j in range(len(majorana_0)):
                majorana[j] = kron(pauli[0], majorana_0[j], format="csr")
            majorana[-2] = kron(pauli[0], majorana_c0, format="csr")
            majorana[-1] = kron(pauli[1], eye_mat, format="csr")

            if i < self.N // 2 - 2:
                # Update Majorana operators
                majorana_0 = majorana
                majorana_c0 = kron(pauli[2], eye_mat, format="csr")
            else:
                # Return Majorana operators
                return majorana

    @property
    def q(self):
        return self._q

    @property
    def eta(self):
        return self._eta

    @property
    def sigma(self):
        return self._sigma

    @property
    def majorana(self):
        return self._majorana
