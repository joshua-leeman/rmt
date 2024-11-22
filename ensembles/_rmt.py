# rmt.ensembles._rmt.py
"""
This module contains classes and functions related to the default tenfold
gaussian random matrix ensembles.
It is grouped into the following sections:
    1. Imports
    2. RMT Class
    3. Tenfold Class
    4. Sparse RMT Class
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
from abc import ABC, abstractmethod
from math import pi, log, sqrt

# Third-party imports
import numpy as np
from scipy.integrate import cumtrapz, nquad
from scipy.interpolate import interp1d
from scipy.linalg import eig, eigh, eigvalsh
from scipy.signal import savgol_filter
from scipy.special import gamma


# =============================
# 2. RMT Class
# =============================
class RMT(ABC):
    def __init__(
        self,
        N: int = None,
        dim: int = None,
        scale: float = 1.0,
        dtype: type = np.complex64,
    ) -> None:
        # Set default Dyson index
        self._beta = None

        # Set input parameters while checking dimension
        self._N = int(N) if N is not None else None
        self._dim = int(dim) if dim is not None else 2 ** (self.N // 2 - 1)
        self._scale = scale

        # Set complex data type
        self._dtype = dtype

        # Check if ensemble is valid
        self._check_ens()

        # Set real data type
        self._real_dtype = dtype().real.dtype

        # Set random number generator
        self._rng = np.random.default_rng()

        # Calculate amount of memory per matrix
        self._mem_per_mat, self._mem_per_calc = self._measure_memory()

        # Set default ensemble name
        self._name = self.__class__.__name__

        # Set default order of arguments
        self._order = ["name", "N", "dim", "scale"]

        # Prepare cumulative density for unfolding
        self._prepare_cumulative_density()

    def __repr__(self) -> str:
        if self.N is None:
            return f"{self._name}(dim={self.dim}, scale={self.scale})"
        else:
            return f"{self._name}(N={self.N}, scale={self.scale})"

    def __str__(self) -> str:
        if self.N is None:
            return rf"$\textrm{{{self._name}}}\ D={self.dim}$"
        else:
            return rf"$\textrm{{{self._name}}}\ N={self.N}$"

    @abstractmethod
    def matrix(self, out=None) -> np.ndarray:
        pass

    @abstractmethod
    def mean_density(self, E: float) -> float:
        pass

    def eigval_sample(self, realizs: int = 1):
        # Allocate memory for eigenvalues
        eigenvalues = np.empty((realizs, self.dim), dtype=self.real_dtype)

        # Loop over number of samples
        for r in range(realizs):
            # Compute eigenvalues of random matrix
            eigenvalues[r, :] = eigvalsh(
                self.matrix(), overwrite_a=True, check_finite=False, driver="evr"
            )

        # Return eigenvalues
        return eigenvalues

    def S_matrix(self, energies: np.ndarray, couplings: np.ndarray) -> np.ndarray:
        # Determine number of channels
        channels = couplings.shape[-1]

        # Allocate memory for S-matrix
        S = np.empty((energies.size, channels, channels), dtype=self.dtype)

        # Compute D-matrix first with Hamiltonian
        D = self.matrix()

        # Check if couplings are diagonal
        if len(couplings.shape) == 1:
            # Set coupling matrix
            W = np.zeros((self.dim, channels), dtype=self.dtype)
            np.fill_diagonal(W, couplings)

            # Fill diagonal of D with couplings
            D[np.arange(channels), np.arange(channels)] -= 1j * pi * couplings**2
        else:
            # Compute F = -i \pi WW^\dagger
            F = np.matmul(couplings, couplings.T.conj())
            F *= -1j * pi

            # Point to coupling matrix
            W = couplings

            # Add F to H in-place
            np.add(D, F, out=D)

        # Perform eigendeomposition of D
        eigenvals, eigenvecs = eig(D, overwrite_a=True, check_finite=False)

        # Compute rotated coupling matrix
        rotated_couplings = np.linalg.solve(eigenvecs, W)

        # Compute inverses of eiegnvalues - energies for all energies
        Lambda_inv = 1.0 / (eigenvals[None, :] - energies[:, None])

        # Compute temporary product
        temp = Lambda_inv[:, :, None] * rotated_couplings[None, :, :]

        # Compute X by multiplying rotated couplings with temporary product
        X = np.matmul(eigenvecs, temp)

        # Compute S-matrix
        S = np.matmul(W.T.conj(), X)
        S *= 1j * 2 * pi
        S[:, np.arange(channels), np.arange(channels)] += 1

        # Return S-matrix
        return S

    def S_elem_sample(
        self,
        energies: np.ndarray,
        indices: list,
        couplings: np.ndarray,
        realizs: int = 1,
    ) -> np.ndarray:
        # Unpack number of channels
        channels = couplings.shape[-1]

        # Determine center index of energies
        c = energies.size // 2

        # Allocate memory to store one-point S-matrix elements
        S_avgs = np.zeros(channels, dtype=self.dtype)

        # Allocate memory for two-point S-matrix elements
        SS_disc = np.zeros((c + 1, 2 * len(indices)), dtype=self.dtype)
        SS_conn = np.zeros((c + 1, len(indices)), dtype=self.dtype)

        # Retwrite indices to be tuples of numpy arrays
        arr_inds = [tuple(np.array(i) for i in zip(*index)) for index in indices]

        # Sample elements from random S-matrices
        for _ in range(realizs):
            # Compute S-matrix
            S = self.S_matrix(energies, couplings)

            # Store diagonal elements of S-matrix
            S_avgs += np.diagonal(S[c])

            # Loop through indices and store one-point elements
            for i, a in enumerate(arr_inds):
                # Store one-point elements
                SS_disc[:, 2 * i] += S[c:, *a][:, 0]
                SS_disc[:, 2 * i + 1] += S[c::-1, *a][:, 1].conj()

                # Store two-point elements
                SS_conn[:, i] += S[c:, *a][:, 0] * S[c::-1, *a][:, 1].conj()

        # Return one-point and two-point S-matrix elements
        return S_avgs, SS_disc, SS_conn

    def Wigner_delay(self, energies: np.ndarray, couplings: np.ndarray) -> np.ndarray:
        # Determine number of channels
        channels = couplings.shape[-1]

        # Allocate memory for Wigner time delay matrix
        Q = np.empty((energies.size, channels, channels), dtype=self.dtype)

        # Compute D-matrix first with Hamiltonian
        D = self.matrix()

        # Check if couplings are diagonal
        if len(couplings.shape) == 1:
            # Set coupling matrix
            W = np.zeros((self.dim, channels), dtype=self.dtype)
            np.fill_diagonal(W, couplings)

            # Fill diagonal of D with couplings
            D[np.arange(channels), np.arange(channels)] -= 1j * pi * couplings**2
        else:
            # Compute F = -i \pi WW^\dagger
            F = np.matmul(couplings, couplings.T.conj())
            F *= -1j * pi

            # Point to coupling matrix
            W = couplings

            # Add F to H in-place
            np.add(D, F, out=D)

        # Perform eigendeomposition of D
        eigenvals, eigenvecs = eig(D, overwrite_a=True, check_finite=False)

        # Compute rotated coupling matrix
        rotated_couplings = np.linalg.solve(eigenvecs, W)

        # Compute inverses of eiegnvalues - energies for all energies
        Lambda_inv = 1.0 / (eigenvals[None, :] - energies[:, None])

        # Compute temporary product
        temp = Lambda_inv[:, :, None] * rotated_couplings[None, :, :]

        # Compute X by multiplying rotated couplings with temporary product
        X = np.matmul(eigenvecs, temp)

        # Compute S-matrix adjoint
        S = np.matmul(W.T.conj(), X)
        S *= 1j * 2 * pi
        S[:, np.arange(channels), np.arange(channels)] += 1
        np.conjugate(np.transpose(S, (0, 2, 1)), out=S)

        # Calculate derivative of S-matrix
        Y = np.matmul(eigenvecs, Lambda_inv[:, :, None] * temp)
        np.matmul(W.T.conj(), Y, out=Q)
        Q *= 1j * 2 * pi

        # Left-multiply by S-matrix adjoint and store in Q
        np.matmul(S, Q, out=Q)
        Q *= -1j

        # Return Wigner time delay matrix
        return Q

    def time_delay_sample(
        self, energies: np.ndarray, couplings: np.ndarray, realizs: int = 1
    ):
        # Determine number of channels
        channels = couplings.shape[-1]

        # Allocate memory for proper time delays
        propers = np.empty((energies.size, realizs, channels), dtype=self.real_dtype)

        # Allocate memory for Wigner time delays
        wigners = np.empty((energies.size, realizs), dtype=self.real_dtype)

        # Sample time delay matrices
        for r in range(realizs):
            # Compute Wigner time delay matrix
            Q = self.Wigner_delay(energies, couplings)

            # Compute proper time delays
            propers[:, r, :] = np.linalg.eigvalsh(Q)

            # Compute Wigner time delays
            wigners[:, r] = np.trace(Q, axis1=1, axis2=2).real / channels

        # Return time delay matrices
        return propers, wigners

    def unfold(self, eigenvalue: float) -> float:
        # Use linear interpolation to unfold eigenvalues
        interpolater = interp1d(
            self._eigen_grid, self._cumulative_density, kind="linear"
        )

        # If Kramer's degeneracy is present, return halved unfolded eigenvalue
        # This is because all eigenvalues have multiplicity 2
        if self.beta == 4:
            return 0.5 * interpolater(eigenvalue)
        else:
            return interpolater(eigenvalue)

    def level_spacings(self, unfolded_eigvals: np.ndarray) -> np.ndarray:
        # If dimension is one, return
        if self.dim == 1:
            return

        # Compute spacings between unfolded eigenvalues
        spacings = np.diff(unfolded_eigvals, axis=1)

        # If Kramer's degeneracy is present, clean spacings
        if self.beta == 4:
            spacings = spacings[:, 1::2]
            spacings = np.concatenate((spacings, spacings), axis=1)

        # Return level spacings
        return spacings

    def form_factors(self, times: np.ndarray, unfolded_eigvals: np.ndarray):
        # Calculate unfolded exponentials
        exponentials = np.multiply(times[:, None, None], unfolded_eigvals)
        exponentials = (-1j * 2 * pi) * exponentials

        np.exp(exponentials, out=exponentials)

        # Calculate partition function realizations and delete exponentials
        Z = np.sum(exponentials, axis=2)

        # Calculate mean of partition functions
        Z_mean = np.mean(Z, axis=1)

        # Calculate spectral and connected spectral form factors
        sff = np.abs(Z_mean) ** 2  # disconnected part
        csff = np.var(Z, axis=1)  # connected part
        sff += csff  # total

        # Nomalize form factors
        sff /= self.dim
        csff /= self.dim

        # Return spectral and connected spectral form factors
        return sff, csff

    def evolve_state(
        self, state: np.ndarray, times: np.ndarray, realizs: int = 1
    ) -> np.ndarray:
        # Initialize memory to store evolved states
        evolved_states = np.empty((realizs, times.size, state.size), dtype=self.dtype)

        # Loop over realizations of random eigensystems
        for r in range(realizs):
            # Diagonalize random Hamiltonian
            eigenvals, eigenvecs = eigh(
                self.matrix(), overwrite_a=True, check_finite=False, driver="evr"
            )

            # Unfold eigenvalues
            eigenvals = self.unfold(eigenvals)

            # Rotate initial state to eigenbasis
            rotated_state = np.matmul(eigenvecs.T.conj(), state)

            # Outer-multiply eigenvalues and times, exponentiate, then broadcast multiply
            np.outer(times, eigenvals, out=evolved_states[r, :, :])
            evolved_states[r, :, :] *= -1j * 2 * pi
            np.exp(evolved_states[r, :, :], out=evolved_states[r, :, :])
            np.multiply(
                evolved_states[r, :, :], rotated_state, out=evolved_states[r, :, :]
            )

            # Rotate back to original basis
            np.matmul(evolved_states[r, :, :], eigenvecs.T, out=evolved_states[r, :, :])

        # Return evolved states
        return evolved_states.transpose(1, 0, 2)

    def scatter_state(
        self,
        state: np.ndarray,
        energies: np.ndarray,
        couplings: np.ndarray = None,
        realizs: int = 1,
    ):
        # Unpack number of channels
        channels = state.size

        # Allocate memory for transmission curves
        T_coeffs = np.zeros((energies.size, channels), dtype=self.dtype)

        # Initialize memory to store scattered states
        scattered_states = np.empty(
            (realizs, energies.size, channels), dtype=self.dtype
        )

        # Compute scattering states
        for r in range(realizs):
            # Compute S-matrix
            S = self.S_matrix(energies, couplings)

            # Store diagonal elements of S-matrix
            T_coeffs += np.diagonal(S, axis1=1, axis2=2)

            # Compute scattered state
            scattered_states[r, :, :] = np.dot(S, state)

        # Divide by number of realizations in-place
        T_coeffs /= realizs

        # Return transmisson coefficients and scattered states
        return (1 - np.abs(T_coeffs) ** 2).T, scattered_states.transpose(1, 0, 2)

    def entropy(self, final_states: np.ndarray) -> np.ndarray:
        # Unpack number of realizations and Hilbert space dimension
        realizs = final_states.shape[1]
        dimension = final_states.shape[2]

        # Compute density matrices
        density_matrices = np.matmul(
            final_states.conj().transpose(0, 2, 1), final_states
        )

        # Divide by number of realizations in-place
        density_matrices /= realizs

        # Calculate probabilities at each time
        probabilities = np.diagonal(density_matrices, axis1=1, axis2=2).real.T

        # Compute eigenvalues of density matrices
        eigenvalues = np.linalg.eigvalsh(density_matrices)

        # Compute entropies and normalize w.r.t. maximum entropy
        entropy = -np.sum(eigenvalues * np.log(eigenvalues), axis=1)
        entropy /= np.log(dimension)

        # Return entropies
        return entropy, probabilities

    def temperature(self, entropies: np.ndarray, delta: float) -> np.ndarray:
        # Return temperature curve
        return 1.0 / self._smooth_data(entropies, 1, delta)

    def free_energy(
        self, energies: np.ndarray, entropies: np.ndarray, temperatures: np.ndarray
    ) -> np.ndarray:
        # Return free energy curve
        return energies - temperatures * entropies

    def heat_capacity(
        self, entropies: np.ndarray, temperatures: np.ndarray, delta: float
    ) -> np.ndarray:
        # Return heat capacity curve
        return -self._smooth_data(entropies, 2, delta) / (temperatures**2)

    def wigner_surmise(self, s: float) -> float:
        if self.beta == 0:
            return np.exp(-s)
        else:
            a = (
                2
                * gamma((self.beta + 2) / 2) ** (self.beta + 1)
                / gamma((self.beta + 1) / 2) ** (self.beta + 2)
            )
            b = (gamma((self.beta + 2) / 2) / gamma((self.beta + 1) / 2)) ** 2
            return a * s**self.beta * np.exp(-b * s**2)

    def csff_universal(self, t: float) -> float:
        # Return GOE connected spectral form factor if beta = 1
        if self.beta == 1:
            if t <= 1:
                return 2 * t - t * log(2 * t + 1)
            else:
                return 2 - t * log((2 * t + 1) / (2 * t - 1))

        # Return GUE connected spectral form factor if beta = 2
        elif self.beta == 2:
            if t <= 1:
                return t
            else:
                return 1.0

        # Return GSE connected spectral form factor if beta = 4
        elif self.beta == 4:
            if t <= 2:
                return t - t / 2 * log(abs(t - 1))
            else:
                return 2.0

        # Return unity for other Dyson indices
        else:
            return 1.0

    def SS_conn_universal(self, i: tuple, T: np.ndarray, s: float) -> float:
        # Define integrand as a function of three variables
        def integrand(x1, x2, x0):
            # Compute measure
            mu = ((1 - x0) * x0 * abs(x1 - x2)) / (
                sqrt(x1 * (1 + x1))
                * sqrt(x2 * (1 + x2))
                * (x0 + x1) ** 2
                * (x0 + x2) ** 2
            )

            # Compute general factor
            factor = np.prod((1 - T * x0) / np.sqrt((1 + T * x1) * (1 + T * x2)))

            # Compute complex exponential
            exponential = np.exp(-1j * pi * s * (x1 + x2 + 2 * x0)) / 8

            # Construct index-dependent factor
            J = (
                ((i[0][0] == i[1][0]) * (i[0][1] == i[1][1]))
                + ((i[0][1] == i[1][0]) * (i[0][0] == i[1][1]))
                * (T[i[0][0]] * T[i[0][1]])
                * (
                    2 * x0 * (1 - x0) / (1 - T[i[0][0]] * x0) / (1 - T[i[1][0]] * x0)
                    + x1 * (1 + x1) / (1 + T[i[0][0]] * x1) / (1 + T[i[1][0]] * x1)
                    + x2 * (1 + x2) / (1 + T[i[0][0]] * x2) / (1 + T[i[1][0]] * x2)
                )
                + ((i[0][0] == i[0][1]) * (i[1][0] == i[1][1]))
                * (
                    T[i[0][0]]
                    * T[i[1][1]]
                    * sqrt(1 - T[i[0][0]])
                    * sqrt(1 - T[i[1][0]])
                )
                * (
                    2 * x0 / (1 - T[i[0][0]] * x0)
                    + 2 * x1 / (1 + T[i[0][0]] * x1)
                    + 2 * x2 / (1 + T[i[0][0]] * x2)
                )
                * (
                    2 * x0 / (1 - T[i[1][0]] * x0)
                    + 2 * x1 / (1 + T[i[1][0]] * x1)
                    + 2 * x2 / (1 + T[i[1][0]] * x2)
                )
            )

            # Return integrand
            return mu * factor * exponential * J

        # Perform triple integral
        return nquad(
            integrand, ranges=[(0, 1), (0, np.inf), (0, np.inf)], opts={"limit": 100}
        )[0]

    def _check_ens(self) -> None:
        # Check if dimension parameters are valid
        if self.N is not None:
            if self.N < 1 or self.N % 2 != 0:
                raise ValueError("Number of Majoranas must be a positive even integer")
            if self.dim is not None and self.dim != 2 ** (self.N // 2 - 1):
                raise ValueError("N and dim must be consistent.")
        elif self.dim is not None:
            if self.dim < 1:
                raise ValueError("Dimension must be a positive integer.")
        else:
            raise ValueError("N or dim must be provided.")

        # Check if ensemble scale is valid
        if not isinstance(self.scale, (int, float)) or self.scale <= 0:
            raise ValueError("Scale parameter must be a positive number.")

        # Check if data type is valid
        if not isinstance(self.dtype, type):
            raise ValueError("Data type must be a valid numpy data type.")

    def _measure_memory(self) -> tuple:
        # Calculate memory per matrix
        memory = self.dim**2 * np.dtype(self.dtype).itemsize

        # Return memory per matrix and memory per calculation
        return memory, 4 * memory

    def _prepare_cumulative_density(self) -> None:
        # Create grid for cumulative trapezoidal integration
        self._eigen_grid = np.linspace(-3 * self.scale, 3 * self.scale, num=2**16)
        density_values = self.dim * np.vectorize(self.mean_density)(self._eigen_grid)

        # Compute cumulative trapezoidal integration
        self._cumulative_density = cumtrapz(density_values, self._eigen_grid, initial=0)

    def _smooth_data(
        self, data: np.ndarray, deriv: int = 0, delta: float = 1.0
    ) -> np.ndarray:
        # Use Savitzky-Golay filter to smooth data
        return savgol_filter(
            data,
            window_length=(data.size // 10) | 1,
            polyorder=2,
            deriv=deriv,
            delta=delta,
        )

    @property
    def N(self) -> int:
        return self._N

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def beta(self) -> int:
        return self._beta

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def kramer(self) -> bool:
        return self._kramer

    @property
    def dtype(self) -> type:
        return self._dtype

    @property
    def real_dtype(self) -> type:
        return self._real_dtype


# =============================
# 3. Tenfold Class
# =============================
class Tenfold(RMT):
    def __init__(
        self,
        beta: int,
        N: int = None,
        dim: int = None,
        scale: float = 1.0,
        dtype: type = np.float64,
    ) -> None:
        # Initialize RMT ensemble
        super().__init__(N=N, dim=dim, scale=scale, dtype=dtype)

        # Set Dyson index
        self._beta = beta

        # Calculate standard deviation
        self._sigma = self.scale / sqrt(2 * self.beta * self.dim)

    def mean_density(self, E: float) -> float:
        # Calculate semicircular spectral density
        if abs(E) < self.scale:
            return sqrt(1 - (E / self.scale) ** 2) / (pi * self.scale / 2)
        else:
            return 0.0

    @property
    def beta(self) -> int:
        return self._beta

    @property
    def sigma(self) -> float:
        return self._sigma


# =============================
# 4. Sparse RMT Class
# =============================
class SparseRMT(RMT):
    # def S_matrix(self, energies: np.ndarray, couplings: np.ndarray = None):
    #     pass

    # def scatter_state(
    #     self,
    #     state,
    #     energies: np.ndarray,
    #     couplings: np.ndarray = None,
    #     realizs: int = 1,
    # ):
    #     pass

    # def _measure_memory(self):
    #     pass
    pass
