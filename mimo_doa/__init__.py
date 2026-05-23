"""MIMO DOA estimation toolkit: classical algorithms + deep learning approach."""
from .signal_model import (
    ArrayGeometry,
    ti_awr1843_geometry,
    steering_vector_mimo,
    generate_snapshots,
    sample_covariance,
)
from .classic_doa import music_spectrum, esprit_doa, music_doa
from .models import ScenarioClassifier, CovReconstructor
from .dataset import SyntheticDOADataset

__all__ = [
    "ArrayGeometry",
    "ti_awr1843_geometry",
    "steering_vector_mimo",
    "generate_snapshots",
    "sample_covariance",
    "music_spectrum",
    "music_doa",
    "esprit_doa",
    "ScenarioClassifier",
    "CovReconstructor",
    "SyntheticDOADataset",
]
