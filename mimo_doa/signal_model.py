"""Math model of MIMO virtual array, signal generation and sample covariance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ArrayGeometry:
    """Linear MIMO array geometry. Coordinates are in units of wavelength λ."""

    d_tx: np.ndarray
    d_rx: np.ndarray

    @property
    def n_tx(self) -> int:
        return int(self.d_tx.size)

    @property
    def n_rx(self) -> int:
        return int(self.d_rx.size)

    @property
    def n_virtual(self) -> int:
        return self.n_tx * self.n_rx

    @property
    def d_virtual(self) -> np.ndarray:
        # virtual array = sum of tx and rx phase centers (Kronecker geometry).
        return (self.d_tx[:, None] + self.d_rx[None, :]).ravel()


def ti_awr1843_geometry() -> ArrayGeometry:
    """TI AWR1843: 2 Tx, 4 Rx; rx spacing λ/2, tx spacing 2λ → 8 virtual elements."""
    d_rx = np.arange(4) * 0.5
    d_tx = np.array([0.0, 2.0])
    return ArrayGeometry(d_tx=d_tx, d_rx=d_rx)


def steering_vector_mimo(
    theta_tx_rad: float,
    theta_rx_rad: float,
    geometry: ArrayGeometry,
) -> np.ndarray:
    """MIMO steering vector for tx/rx directions θ_tx, θ_rx (Eq. 3 of the article).

    Returns vector of size n_tx*n_rx where element (i,j) corresponds to
    phase shift exp(i k (d_tx[i] sin θ_tx + d_rx[j] sin θ_rx)).
    """
    k = 2 * np.pi
    phase_tx = np.exp(1j * k * geometry.d_tx * np.sin(theta_tx_rad))
    phase_rx = np.exp(1j * k * geometry.d_rx * np.sin(theta_rx_rad))
    return np.kron(phase_tx, phase_rx)


def steering_matrix(
    angles_rad: Sequence[float], geometry: ArrayGeometry
) -> np.ndarray:
    """Manifold matrix A(θ) for direct path (θ_tx = θ_rx = θ)."""
    cols = [steering_vector_mimo(a, a, geometry) for a in angles_rad]
    return np.column_stack(cols) if cols else np.zeros((geometry.n_virtual, 0), dtype=complex)


def _draw_complex_amplitudes(
    n: int, rng: np.random.Generator, amp_min: float = 0.5, amp_max: float = 1.5
) -> np.ndarray:
    mag = rng.uniform(amp_min, amp_max, size=n)
    phase = rng.uniform(-np.pi, np.pi, size=n)
    return mag * np.exp(1j * phase)


def generate_snapshots(
    *,
    geometry: ArrayGeometry,
    scenario: str,
    angles_deg: Sequence[float] | None = None,
    snr_db: float = 10.0,
    n_snapshots: int = 16,
    rng: np.random.Generator | None = None,
    multipath_pair_deg: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate signal snapshots for a chosen scenario.

    scenario ∈ {"single", "double", "multipath"}.
    Returns x of shape (n_virtual, n_snapshots) and metadata dict.
    """
    if rng is None:
        rng = np.random.default_rng()
    M = geometry.n_virtual

    if scenario == "single":
        if angles_deg is None:
            angles_deg = [float(rng.uniform(-60, 60))]
        angles_deg = list(angles_deg)[:1]
    elif scenario == "double":
        if angles_deg is None:
            t1 = float(rng.uniform(-60, 50))
            sep = float(rng.uniform(5, 25))
            angles_deg = [t1, t1 + sep]
        angles_deg = list(angles_deg)[:2]
    elif scenario == "multipath":
        if multipath_pair_deg is None:
            theta_tx = float(rng.uniform(-60, 60))
            delta = float(rng.choice([-1, 1])) * float(rng.uniform(10, 35))
            theta_rx = float(np.clip(theta_tx + delta, -75, 75))
            multipath_pair_deg = (theta_tx, theta_rx)
        angles_deg = list(multipath_pair_deg)
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    if scenario in ("single", "double"):
        A = steering_matrix(np.deg2rad(angles_deg), geometry)
        amps = _draw_complex_amplitudes(len(angles_deg), rng)
        s = amps[:, None] * (rng.standard_normal((len(angles_deg), n_snapshots))
                              + 1j * rng.standard_normal((len(angles_deg), n_snapshots))) / np.sqrt(2)
        signal = A @ s
    else:
        theta_tx, theta_rx = multipath_pair_deg
        a = steering_vector_mimo(np.deg2rad(theta_tx), np.deg2rad(theta_rx), geometry)
        amp = _draw_complex_amplitudes(1, rng)[0]
        s = amp * (rng.standard_normal(n_snapshots) + 1j * rng.standard_normal(n_snapshots)) / np.sqrt(2)
        signal = a[:, None] * s[None, :]

    sig_power = float(np.mean(np.abs(signal) ** 2))
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = sig_power / max(snr_lin, 1e-12)
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal((M, n_snapshots)) + 1j * rng.standard_normal((M, n_snapshots))
    )
    x = signal + noise

    meta = {
        "scenario": scenario,
        "angles_deg": list(angles_deg),
        "snr_db": float(snr_db),
        "n_snapshots": int(n_snapshots),
        "multipath_pair_deg": multipath_pair_deg,
    }
    return x, meta


def sample_covariance(x: np.ndarray) -> np.ndarray:
    """Sample covariance R̃ = 1/T Σ x(n) x^H(n)."""
    n_snap = x.shape[1]
    return (x @ x.conj().T) / max(n_snap, 1)


def true_covariance_direct_path(
    angles_deg: Sequence[float],
    geometry: ArrayGeometry,
    source_powers: Sequence[float] | None = None,
    noise_power: float = 0.0,
) -> np.ndarray:
    """Build noise-free R = A R_s A^H + σ² I for known direct-path sources."""
    A = steering_matrix(np.deg2rad(angles_deg), geometry)
    n = A.shape[1]
    if source_powers is None:
        source_powers = np.ones(n)
    R_s = np.diag(np.asarray(source_powers, dtype=complex))
    return A @ R_s @ A.conj().T + noise_power * np.eye(geometry.n_virtual)
