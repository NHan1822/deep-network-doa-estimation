"""Classical subspace DOA estimators: MUSIC and ESPRIT."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import find_peaks

from .signal_model import ArrayGeometry, steering_vector_mimo


def _eig_descending(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hermitian eigen-decomposition sorted in descending order."""
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx]


def music_spectrum(
    R: np.ndarray,
    geometry: ArrayGeometry,
    n_sources: int,
    theta_grid_deg: np.ndarray,
) -> np.ndarray:
    """Compute pseudo-spectrum P_MUSIC(θ) for direct-path assumption."""
    _, U = _eig_descending(R)
    n_total = U.shape[1]
    Un = U[:, n_sources:n_total]
    spectrum = np.empty(theta_grid_deg.size, dtype=float)
    for i, theta_deg in enumerate(theta_grid_deg):
        a = steering_vector_mimo(np.deg2rad(theta_deg), np.deg2rad(theta_deg), geometry)
        proj = Un.conj().T @ a
        denom = float(np.real(np.vdot(proj, proj)))
        spectrum[i] = 1.0 / max(denom, 1e-12)
    return spectrum


def music_doa(
    R: np.ndarray,
    geometry: ArrayGeometry,
    n_sources: int,
    theta_grid_deg: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate DOA via MUSIC. Returns (angles_deg, spectrum)."""
    if theta_grid_deg is None:
        theta_grid_deg = np.linspace(-90, 90, 1801)
    spectrum = music_spectrum(R, geometry, n_sources, theta_grid_deg)
    peaks, _ = find_peaks(spectrum)
    if peaks.size == 0:
        # Fallback: top values
        order = np.argsort(spectrum)[::-1][:n_sources]
    else:
        order = peaks[np.argsort(spectrum[peaks])[::-1]][:n_sources]
    angles = np.sort(theta_grid_deg[order])
    return angles, spectrum


def esprit_doa(
    R: np.ndarray,
    geometry: ArrayGeometry,
    n_sources: int,
) -> np.ndarray:
    """ESPRIT for the virtual ULA-like MIMO geometry.

    NOTE: works correctly when the virtual array is uniformly spaced with d_step.
    We extract subarrays U_s1 (rows 0..M-2) and U_s2 (rows 1..M-1) and solve
    Ψ via TLS, then arcsin from the phase.
    """
    _, U = _eig_descending(R)
    Us = U[:, :n_sources]
    M = Us.shape[0]
    Us1, Us2 = Us[: M - 1, :], Us[1:, :]

    C = np.hstack([Us1, Us2])
    _, _, Vh = np.linalg.svd(C, full_matrices=True)
    V = Vh.conj().T
    half = n_sources
    V12 = V[:half, half:]
    V22 = V[half:, half:]
    try:
        Psi = -V12 @ np.linalg.inv(V22)
    except np.linalg.LinAlgError:
        Psi = -V12 @ np.linalg.pinv(V22)
    eigvals = np.linalg.eigvals(Psi)
    d_step = float(np.mean(np.diff(np.sort(geometry.d_virtual))))
    if d_step <= 0:
        d_step = 0.5
    sin_theta = np.angle(eigvals) / (2 * np.pi * d_step)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    angles = np.sort(np.rad2deg(np.arcsin(sin_theta)))
    return angles


def forward_backward_smoothing(R: np.ndarray) -> np.ndarray:
    """Forward/backward smoothing to improve rank conditioning on a single snapshot."""
    M = R.shape[0]
    J = np.fliplr(np.eye(M))
    R_fb = 0.5 * (R + J @ R.conj() @ J)
    return R_fb
