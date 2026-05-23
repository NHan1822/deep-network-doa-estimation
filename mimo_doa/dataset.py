"""Synthetic dataset for joint training of scenario classifier and cov reconstructor."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .signal_model import (
    ArrayGeometry,
    generate_snapshots,
    sample_covariance,
    true_covariance_direct_path,
)


SCENARIO_LABELS = {"single": 0, "double": 1, "multipath": 2}


def _vector_to_iq_channels(v: np.ndarray) -> np.ndarray:
    """Return shape (2, M) of [real; imag]."""
    return np.stack([v.real, v.imag], axis=0).astype(np.float32)


def _matrix_to_iq_channels(M: np.ndarray) -> np.ndarray:
    """Return shape (2, m, m) of [real; imag]."""
    return np.stack([M.real, M.imag], axis=0).astype(np.float32)


class SyntheticDOADataset(Dataset):
    """Generates (x_vec_iq, cov_iq, cov_clean_iq, label, angles) tuples on the fly."""

    def __init__(
        self,
        geometry: ArrayGeometry,
        length: int = 8000,
        snr_db_range: tuple[float, float] = (-5.0, 25.0),
        n_snapshots: int = 16,
        scenario_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        seed: int | None = None,
    ):
        self.geometry = geometry
        self.length = length
        self.snr_db_range = snr_db_range
        self.n_snapshots = n_snapshots
        weights = np.asarray(scenario_weights, dtype=float)
        self.scenario_probs = weights / weights.sum()
        self._rng_seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        seed = None if self._rng_seed is None else int(self._rng_seed) + idx
        rng = np.random.default_rng(seed)
        scenario = rng.choice(list(SCENARIO_LABELS.keys()), p=self.scenario_probs)
        snr_db = float(rng.uniform(*self.snr_db_range))

        x, meta = generate_snapshots(
            geometry=self.geometry,
            scenario=str(scenario),
            snr_db=snr_db,
            n_snapshots=self.n_snapshots,
            rng=rng,
        )

        R_sample = sample_covariance(x)

        # Vector representation: use first snapshot
        x_vec_iq = _vector_to_iq_channels(x[:, 0])
        cov_iq = _matrix_to_iq_channels(R_sample)

        if scenario == "multipath":
            cov_clean = np.zeros_like(R_sample)
        else:
            cov_clean = true_covariance_direct_path(meta["angles_deg"], self.geometry)
        cov_clean_iq = _matrix_to_iq_channels(cov_clean)

        label = SCENARIO_LABELS[str(scenario)]
        angles_padded = np.full(2, np.nan, dtype=np.float32)
        for i, a in enumerate(meta["angles_deg"][:2]):
            angles_padded[i] = float(a)

        return (
            torch.from_numpy(x_vec_iq),
            torch.from_numpy(cov_iq),
            torch.from_numpy(cov_clean_iq),
            int(label),
            torch.from_numpy(angles_padded),
            float(snr_db),
        )
