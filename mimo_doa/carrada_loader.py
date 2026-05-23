"""Загрузчик и адаптер для датасета CARRADA (Valeo / Telecom ParisTech).

Датасет CARRADA содержит:
  - angle_doppler_raw/<frame>.npy : магнитудная карта (256 angle × 64 doppler);
  - annotations/dense/<frame>/{range_angle,range_doppler}.npy : 0/1 маски
    по 4 классам (background, pedestrian, cyclist, car);
  - annotations/box/{range_angle,range_doppler}_light.json : bounding boxes
    с метками класса в плоскостях RA и RD.

CARRADA-радар отличается от TI AWR1843 по физике (Synodi FLEX, 79 ГГц,
2 Tx + 4 Rx, виртуальная решётка 8 элементов с шагом λ/2), но имеет ту же
геометрию виртуальной решётки, что позволяет применять разработанные в
работе алгоритмы.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


CARRADA_CLASS_NAMES = {0: "background", 1: "pedestrian", 2: "cyclist", 3: "car"}

# CARRADA radar config (Synodi FLEX, see CARRADA paper)
CARRADA_CONFIG = {
    "frequency": 79e9,
    "wavelength": 3e8 / 79e9,
    "max_range_m": 50.0,
    "range_bins": 256,
    "angle_bins": 256,
    "doppler_bins": 64,
    "n_tx": 2,
    "n_rx": 4,
}


@dataclass
class CarradaFrame:
    """Один кадр CARRADA: AD heatmap + bbox annotations."""
    scene: str
    frame_id: str
    ad_raw: np.ndarray            # (angle=256, doppler=64), магнитуда
    range_angle_boxes: list       # list of (cls, [r0, a0, r1, a1])
    range_doppler_boxes: list     # list of (cls, [r0, d0, r1, d1])
    range_angle_dense: np.ndarray | None = None  # (4, 256, 256) маски
    range_doppler_dense: np.ndarray | None = None  # (4, 256, 64)


def scan_scenes(root: Path) -> list[str]:
    """Возвращает список идентификаторов сцен в корневом каталоге CARRADA."""
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir()
                   if d.is_dir() and d.name.startswith("20"))


def list_frames(scene_dir: Path) -> list[str]:
    """Возвращает идентификаторы кадров, для которых есть AD raw данные."""
    ad_dir = scene_dir / "angle_doppler_raw"
    if not ad_dir.exists():
        return []
    return sorted(p.stem for p in ad_dir.glob("*.npy"))


def load_frame(scene_dir: Path, frame_id: str) -> CarradaFrame:
    """Загрузка одного кадра CARRADA."""
    ad_path = scene_dir / "angle_doppler_raw" / f"{frame_id}.npy"
    ad_raw = np.load(ad_path)

    ra_json = scene_dir / "annotations" / "box" / "range_angle_light.json"
    rd_json = scene_dir / "annotations" / "box" / "range_doppler_light.json"
    ra_boxes, rd_boxes = [], []
    if ra_json.exists():
        with open(ra_json) as f:
            data = json.load(f)
        entry = data.get(frame_id, {})
        for box, label in zip(entry.get("boxes", []), entry.get("labels", [])):
            ra_boxes.append((int(label), [int(x) for x in box]))
    if rd_json.exists():
        with open(rd_json) as f:
            data = json.load(f)
        entry = data.get(frame_id, {})
        for box, label in zip(entry.get("boxes", []), entry.get("labels", [])):
            rd_boxes.append((int(label), [int(x) for x in box]))

    ra_dense_p = scene_dir / "annotations" / "dense" / frame_id / "range_angle.npy"
    rd_dense_p = scene_dir / "annotations" / "dense" / frame_id / "range_doppler.npy"
    ra_dense = np.load(ra_dense_p) if ra_dense_p.exists() else None
    rd_dense = np.load(rd_dense_p) if rd_dense_p.exists() else None

    return CarradaFrame(
        scene=scene_dir.name,
        frame_id=frame_id,
        ad_raw=ad_raw,
        range_angle_boxes=ra_boxes,
        range_doppler_boxes=rd_boxes,
        range_angle_dense=ra_dense,
        range_doppler_dense=rd_dense,
    )


def angle_axis_deg(n_bins: int = 256, fov_deg: float = 180.0) -> np.ndarray:
    """Возвращает ось углов для CARRADA AD heatmap (центрированно)."""
    return np.linspace(-fov_deg / 2, fov_deg / 2, n_bins)


def doppler_axis_ms(n_bins: int = 64, max_vel_ms: float = 13.5) -> np.ndarray:
    """Возвращает ось доплеровских частот для CARRADA."""
    return np.linspace(-max_vel_ms, max_vel_ms, n_bins)


def range_axis_m(n_bins: int = 256, max_range_m: float = 50.0) -> np.ndarray:
    """Возвращает ось дальности для CARRADA."""
    return np.linspace(0.0, max_range_m, n_bins)


def load_range_angle(scene_dir: Path, frame_id: str) -> np.ndarray | None:
    """Загрузка магнитудной range-angle карты (256×256) если есть."""
    p = scene_dir / "range_angle_numpy" / f"{frame_id}.npy"
    if not p.exists():
        return None
    return np.load(p)


def find_frames_with_targets(scene_dir: Path, target_classes: tuple[int, ...] = (1, 2, 3),
                              max_frames: int = 50) -> list[str]:
    """Возвращает идентификаторы кадров, содержащих цели нужных классов."""
    ra_json = scene_dir / "annotations" / "box" / "range_angle_light.json"
    if not ra_json.exists():
        return []
    with open(ra_json) as f:
        data = json.load(f)
    out = []
    raw_frames = set(list_frames(scene_dir))
    for fid, entry in data.items():
        if fid not in raw_frames:
            continue
        labels = entry.get("labels", [])
        if any(l in target_classes for l in labels):
            out.append(fid)
            if len(out) >= max_frames:
                break
    return sorted(out)


def signal_vector_from_ad(ad_raw: np.ndarray, doppler_bin: int,
                           n_virtual: int = 8) -> np.ndarray:
    """Извлекает сигнальный вектор виртуальной 8-элементной решётки из CARRADA-AD heatmap.

    Идея: angle_doppler_raw — это магнитуда после Angle-FFT. Возвращаем
    обратное FFT по оси углов (сужая до N_virtual элементов), что эквивалентно
    наблюдению на 8-элементной виртуальной ULA при beamforming-режиме.
    """
    angle_profile = ad_raw[:, doppler_bin].astype(np.complex64)
    # Inverse FFT angular profile → spatial samples
    full_signal = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(angle_profile)))
    n = full_signal.size
    start = (n - n_virtual) // 2
    return full_signal[start:start + n_virtual]


def correlation_matrix(x: np.ndarray) -> np.ndarray:
    """R = x x^H для одного снапшота."""
    x = np.asarray(x, dtype=np.complex64).reshape(-1, 1)
    return (x @ x.conj().T).astype(np.complex64)
