from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mimo_doa.classic_doa import (
    esprit_doa,
    forward_backward_smoothing,
    music_doa,
    music_spectrum,
)
from mimo_doa.dataset import SCENARIO_LABELS, SyntheticDOADataset
from mimo_doa.models import CovReconstructor, ScenarioClassifier
from mimo_doa.signal_model import (
    generate_snapshots,
    sample_covariance,
    ti_awr1843_geometry,
)
from mimo_doa.fmcw import (
    FMCWConfig,
    Target,
    doppler_axis_ms,
    range_axis_m,
    range_doppler_map,
    synthesize_adc_frame,
)
from mimo_doa.carrada_loader import (
    CARRADA_CLASS_NAMES,
    angle_axis_deg,
    correlation_matrix,
    find_frames_with_targets,
    list_frames,
    load_frame,
    load_range_angle,
    range_axis_m as carrada_range_axis_m,
    scan_scenes,
    signal_vector_from_ad,
)
from mimo_doa.carrada_loader import doppler_axis_ms as carrada_doppler_axis_ms

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 130,
})


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def fig_array_geometry(out_dir: Path):
    geometry = ti_awr1843_geometry()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

    ax = axes[0]
    ax.scatter(geometry.d_tx, [0.4] * geometry.n_tx, marker="^", s=180, color="crimson",
               label=f"Tx ({geometry.n_tx})", zorder=3)
    ax.scatter(geometry.d_rx, [0.0] * geometry.n_rx, marker="v", s=180, color="navy",
               label=f"Rx ({geometry.n_rx})", zorder=3)
    ax.annotate("d = λ/2", xy=(0.5, 0.0), xytext=(0.5, -0.25),
                ha="center", arrowprops=dict(arrowstyle="-"))
    ax.set_title("а) Физическая решётка (2 Tx + 4 Rx)")
    ax.set_xlabel("Координата, λ")
    ax.set_yticks([])
    ax.set_ylim(-0.5, 1.0)
    ax.set_xlim(-0.5, 3.0)
    ax.legend(loc="upper right")
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1]
    dv = np.sort(geometry.d_virtual)
    ax.scatter(dv, [0.0] * dv.size, marker="o", s=140, color="darkgreen",
               label=f"Виртуальная решётка ({dv.size})", zorder=3)
    for i, d in enumerate(dv):
        ax.text(d, 0.18, str(i), ha="center", fontsize=9)
    ax.set_title("б) Виртуальная решётка (8 элементов)")
    ax.set_xlabel("Координата, λ")
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.6)
    ax.set_xlim(-0.5, 4)
    ax.legend(loc="upper right")
    ax.grid(True, axis="x", alpha=0.3)

    return _save(fig, out_dir, "fig01_array_geometry")


def fig_geometry_direct_vs_multipath(out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    for ax, title, multipath in [
        (axes[0], "а) Однократное отражение: θ_tx = θ_rx", False),
        (axes[1], "б) Многократное отражение: θ_tx ≠ θ_rx", True),
    ]:
        radar = np.array([0.0, 0.0])
        target = np.array([6.0, 4.0])
        ax.plot(*radar, marker="s", markersize=14, color="black")
        ax.text(0.0, -0.6, "Радар", ha="center")
        ax.plot(*target, marker="o", markersize=12, color="darkorange")
        ax.text(target[0] + 0.3, target[1], "Цель")

        if not multipath:
            ax.annotate("", xy=target, xytext=radar,
                        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.6))
            ax.annotate("", xy=radar + (0.15, 0.0), xytext=target,
                        arrowprops=dict(arrowstyle="->", color="navy", lw=1.6))
            ax.text(2.8, 2.6, "θ_tx = θ_rx", color="black")
        else:
            mid = np.array([3.0, 6.0])
            ax.plot(*mid, marker="^", markersize=12, color="gray")
            ax.text(mid[0], mid[1] + 0.4, "Здание", ha="center")
            ax.annotate("", xy=mid, xytext=radar,
                        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.6))
            ax.annotate("", xy=target, xytext=mid,
                        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.4, linestyle="--"))
            ax.annotate("", xy=radar + (0.15, 0.0), xytext=target,
                        arrowprops=dict(arrowstyle="->", color="navy", lw=1.6))
            ax.text(1.0, 3.5, "θ_tx", color="crimson")
            ax.text(3.6, 2.0, "θ_rx", color="navy")

        ax.set_xlim(-1, 8)
        ax.set_ylim(-1, 8)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_xlabel("x, м"); ax.set_ylabel("y, м")

    return _save(fig, out_dir, "fig02_geometry_direct_vs_multipath")


def fig_classifier_arch(out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    blocks = [
        ("Сигнальный\nвектор\n(2×8)", "#fef3c7"),
        ("Conv1d×3,\ntanh, BN", "#bfdbfe"),
        ("|·|\nмодуль", "#fde68a"),
        ("FC, ReLU\n(256→64)", "#bbf7d0"),
        ("FC, softmax\n(64→3)", "#fecaca"),
    ]
    x = 0
    for label, color in blocks:
        rect = plt.Rectangle((x, 0.2), 1.6, 0.9, facecolor=color, edgecolor="black")
        ax.add_patch(rect)
        ax.text(x + 0.8, 0.65, label, ha="center", va="center", fontsize=10)
        x += 2
    for i in range(len(blocks) - 1):
        ax.annotate("", xy=(i * 2 + 1.6 + 0.2, 0.65), xytext=(i * 2 + 1.6, 0.65),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(x + 0.0, 0.65, "→ {1, 2, mp}", va="center", fontsize=10)
    ax.set_xlim(-0.4, x + 2.5)
    ax.set_ylim(-0.3, 1.6)
    ax.axis("off")
    ax.set_title("Архитектура нейросети-классификатора сценария распространения")
    return _save(fig, out_dir, "fig03_classifier_architecture")


def fig_reconstructor_arch(out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    blocks = [
        ("R̃ (выборочная)\n2×M×M", "#fef3c7"),
        ("Conv2d 1×1,\ntanh, BN", "#bfdbfe"),
        ("Conv2d 2×2,\ntanh, BN", "#bfdbfe"),
        ("FC → R̂\n2×M×M", "#bbf7d0"),
        ("MUSIC по R̂", "#fecaca"),
    ]
    x = 0
    for label, color in blocks:
        rect = plt.Rectangle((x, 0.2), 1.8, 1.0, facecolor=color, edgecolor="black")
        ax.add_patch(rect)
        ax.text(x + 0.9, 0.72, label, ha="center", va="center", fontsize=10)
        x += 2.1
    for i in range(len(blocks) - 1):
        ax.annotate("", xy=(i * 2.1 + 1.8 + 0.3, 0.72),
                    xytext=(i * 2.1 + 1.8, 0.72),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_xlim(-0.4, x + 0.5)
    ax.set_ylim(-0.3, 1.8)
    ax.axis("off")
    ax.set_title("Архитектура нейросети для реконструкции корреляционной матрицы")
    return _save(fig, out_dir, "fig04_reconstructor_architecture")


def fig_pipeline_overview(out_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    blocks = [
        ("RF-сигнал\nMIMO-радара", "#fef3c7"),
        ("Range–Doppler\nFFT + CFAR", "#fde68a"),
        ("Классификатор\nсценария", "#bfdbfe"),
        ("Реконструктор\nковариационной\nматрицы", "#bbf7d0"),
        ("MUSIC →\nоценка DOA", "#fecaca"),
        ("Трекер /\nADAS", "#e9d5ff"),
    ]
    x = 0
    for label, color in blocks:
        rect = plt.Rectangle((x, 0.2), 1.9, 1.0, facecolor=color, edgecolor="black")
        ax.add_patch(rect)
        ax.text(x + 0.95, 0.72, label, ha="center", va="center", fontsize=10)
        x += 2.1
    for i in range(len(blocks) - 1):
        ax.annotate("", xy=(i * 2.1 + 1.9 + 0.2, 0.72),
                    xytext=(i * 2.1 + 1.9, 0.72),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_xlim(-0.4, x + 0.5)
    ax.set_ylim(-0.3, 1.8)
    ax.axis("off")
    ax.set_title("Сквозной конвейер обработки радиолокационных данных")
    return _save(fig, out_dir, "fig05_pipeline_overview")


def fig_training_curves(history_path: Path, out_dir: Path):
    with open(history_path) as f:
        hist = json.load(f)
    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(epochs, hist["train_loss"], label="train", marker="o")
    axes[0].plot(epochs, hist["val_loss"], label="val", marker="s")
    axes[0].set_title("Потери обучения")
    axes[0].set_xlabel("Эпоха"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, hist["val_acc"], color="darkgreen", marker="o")
    axes[1].set_title("Точность классификатора (val)")
    axes[1].set_xlabel("Эпоха"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, hist["val_mse"], color="crimson", marker="o")
    axes[2].set_title("MSE реконструкции R̂ (val)")
    axes[2].set_xlabel("Эпоха"); axes[2].set_ylabel("MSE")
    axes[2].grid(alpha=0.3)
    return _save(fig, out_dir, "fig06_training_curves")


def _confusion(pred: np.ndarray, target: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(pred, target):
        cm[t, p] += 1
    return cm


def fig_confusion_matrix(classifier, geometry, out_dir: Path, device: str):
    rng = np.random.default_rng(0)
    targets, preds = [], []
    for _ in range(900):
        scenario = rng.choice(list(SCENARIO_LABELS.keys()))
        x, _ = generate_snapshots(geometry=geometry, scenario=scenario,
                                  snr_db=float(rng.uniform(0, 20)),
                                  n_snapshots=16, rng=rng)
        R = sample_covariance(x)
        inp = torch.tensor(np.stack([R.real, R.imag])[None].astype(np.float32)).to(device)
        with torch.no_grad():
            logit = classifier(inp).cpu().numpy()[0]
        preds.append(int(np.argmax(logit)))
        targets.append(SCENARIO_LABELS[scenario])

    cm = _confusion(np.array(preds), np.array(targets), 3)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    classes = ["single", "double", "multipath"]
    ax.set_xticks(range(3), classes); ax.set_yticks(range(3), classes)
    ax.set_xlabel("Прогноз"); ax.set_ylabel("Истина")
    ax.set_title("Матрица ошибок классификатора сценария")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.045)
    return _save(fig, out_dir, "fig07_confusion_matrix")


def fig_pseudospectrum(reconstructor, geometry, out_dir: Path, device: str):
    rng = np.random.default_rng(7)
    M = geometry.n_virtual

    def music_with_reconstruct(R_sample, n_sources):
        inp = torch.tensor(np.stack([R_sample.real, R_sample.imag])[None]
                            .astype(np.float32)).to(device)
        with torch.no_grad():
            R_hat_iq = reconstructor(inp).cpu().numpy()[0]
        R_hat = R_hat_iq[0] + 1j * R_hat_iq[1]
        R_hat = 0.5 * (R_hat + R_hat.conj().T)
        grid = np.linspace(-90, 90, 1801)
        spec_rec = music_spectrum(R_hat, geometry, n_sources, grid)
        return grid, spec_rec

    grid = np.linspace(-90, 90, 1801)

    # Single target
    x1, meta1 = generate_snapshots(geometry=geometry, scenario="single",
                                   angles_deg=[12.0], snr_db=5.0, n_snapshots=4, rng=rng)
    R1 = sample_covariance(x1)
    R1_fb = forward_backward_smoothing(R1)
    spec_music = music_spectrum(R1_fb, geometry, 1, grid)
    _, spec_rec = music_with_reconstruct(R1, 1)

    # Two targets
    x2, meta2 = generate_snapshots(geometry=geometry, scenario="double",
                                   angles_deg=[-8.0, 9.0], snr_db=5.0,
                                   n_snapshots=4, rng=rng)
    R2 = sample_covariance(x2)
    R2_fb = forward_backward_smoothing(R2)
    spec_music2 = music_spectrum(R2_fb, geometry, 2, grid)
    _, spec_rec2 = music_with_reconstruct(R2, 2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, spec_classic, spec_nn, true_angles, title in [
        (axes[0], spec_music, spec_rec, meta1["angles_deg"], "а) Одна цель"),
        (axes[1], spec_music2, spec_rec2, meta2["angles_deg"], "б) Две цели"),
    ]:
        ax.plot(grid, 10 * np.log10(spec_classic / spec_classic.max() + 1e-12),
                color="navy", label="MUSIC (forward/backward)")
        ax.plot(grid, 10 * np.log10(spec_nn / spec_nn.max() + 1e-12),
                color="black", label="MUSIC по реконструкции R̂")
        for a in true_angles:
            ax.axvline(a, color="crimson", linestyle="--", alpha=0.6)
        ax.set_xlim(-60, 60)
        ax.set_ylim(-40, 2)
        ax.set_xlabel("Угол θ, град")
        ax.set_ylabel("P(θ), дБ")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
    return _save(fig, out_dir, "fig08_pseudospectrum")


def fig_rmse_vs_snr(reconstructor, geometry, out_dir: Path, device: str):
    snrs = np.arange(-5, 25, 3)
    n_trials = 80
    rng = np.random.default_rng(11)
    grid = np.linspace(-90, 90, 1801)

    rmse_music, rmse_esprit, rmse_nn = [], [], []
    for snr in snrs:
        e_m, e_e, e_n = [], [], []
        for _ in range(n_trials):
            true_a = float(rng.uniform(-50, 50))
            x, _ = generate_snapshots(geometry=geometry, scenario="single",
                                      angles_deg=[true_a], snr_db=float(snr),
                                      n_snapshots=8, rng=rng)
            R = sample_covariance(x)
            R_fb = forward_backward_smoothing(R)

            est_m, _ = music_doa(R_fb, geometry, 1, grid)
            e_m.append((est_m[0] - true_a) ** 2)

            try:
                est_e = esprit_doa(R_fb, geometry, 1)
                e_e.append((est_e[0] - true_a) ** 2)
            except Exception:
                e_e.append(np.nan)

            inp = torch.tensor(np.stack([R.real, R.imag])[None]
                                .astype(np.float32)).to(device)
            with torch.no_grad():
                R_hat_iq = reconstructor(inp).cpu().numpy()[0]
            R_hat = R_hat_iq[0] + 1j * R_hat_iq[1]
            R_hat = 0.5 * (R_hat + R_hat.conj().T)
            est_nn, _ = music_doa(R_hat, geometry, 1, grid)
            e_n.append((est_nn[0] - true_a) ** 2)

        rmse_music.append(float(np.sqrt(np.nanmean(e_m))))
        rmse_esprit.append(float(np.sqrt(np.nanmean(e_e))))
        rmse_nn.append(float(np.sqrt(np.nanmean(e_n))))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(snrs, rmse_music, marker="o", color="navy", label="MUSIC")
    ax.plot(snrs, rmse_esprit, marker="s", color="darkgreen", label="ESPRIT")
    ax.plot(snrs, rmse_nn, marker="^", color="crimson", label="MUSIC + реконструкция R̂")
    ax.set_xlabel("SNR, дБ"); ax.set_ylabel("RMSE угла, град")
    ax.set_yscale("log")
    ax.set_title("RMSE оценки DOA от SNR (single target, T = 8 snapshots)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    return _save(fig, out_dir, "fig09_rmse_vs_snr")


def fig_resolution_probability(reconstructor, geometry, out_dir: Path, device: str):
    seps = np.arange(2, 22, 2, dtype=float)
    n_trials = 80
    rng = np.random.default_rng(17)
    grid = np.linspace(-90, 90, 3601)

    prob_music, prob_nn = [], []
    for sep in seps:
        ok_music, ok_nn = 0, 0
        for _ in range(n_trials):
            t1 = float(rng.uniform(-40, 40 - sep))
            t2 = t1 + float(sep)
            x, _ = generate_snapshots(geometry=geometry, scenario="double",
                                      angles_deg=[t1, t2], snr_db=10.0,
                                      n_snapshots=4, rng=rng)
            R = sample_covariance(x)
            R_fb = forward_backward_smoothing(R)
            est_m, _ = music_doa(R_fb, geometry, 2, grid)
            inp = torch.tensor(np.stack([R.real, R.imag])[None]
                                .astype(np.float32)).to(device)
            with torch.no_grad():
                R_hat_iq = reconstructor(inp).cpu().numpy()[0]
            R_hat = R_hat_iq[0] + 1j * R_hat_iq[1]
            R_hat = 0.5 * (R_hat + R_hat.conj().T)
            est_nn, _ = music_doa(R_hat, geometry, 2, grid)

            true = np.sort([t1, t2])
            if abs(est_m[0] - true[0]) < 0.5 * sep and abs(est_m[1] - true[1]) < 0.5 * sep:
                ok_music += 1
            if abs(est_nn[0] - true[0]) < 0.5 * sep and abs(est_nn[1] - true[1]) < 0.5 * sep:
                ok_nn += 1
        prob_music.append(ok_music / n_trials)
        prob_nn.append(ok_nn / n_trials)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(seps, prob_music, marker="o", color="navy", label="MUSIC")
    ax.plot(seps, prob_nn, marker="^", color="crimson", label="MUSIC + реконструкция R̂")
    ax.set_xlabel("Угловая разность Δθ, град")
    ax.set_ylabel("Вероятность правильного разрешения")
    ax.set_title("Разрешающая способность двух близко расположенных целей")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, out_dir, "fig10_resolution_probability")


def fig_range_doppler_map(out_dir: Path):
    """Карта дальность/скорость из физически корректной FMCW-симуляции."""
    rng = np.random.default_rng(23)
    cfg = FMCWConfig()
    # Два пешехода на 7,7 и 8,0 м + фантомная отметка на 9,5 м (от переотражения)
    targets = [
        Target(range_m=8.0, velocity_ms=-1.0, azimuth_deg=-11.0, rcs=0.4),
        Target(range_m=7.7, velocity_ms=-1.3, azimuth_deg=+11.0, rcs=0.4),
        Target(range_m=9.5, velocity_ms=-1.1, azimuth_deg=0.0, rcs=0.05),
    ]
    adc = synthesize_adc_frame(targets, cfg, snr_db_per_target=18, rng=rng)
    rd_full = range_doppler_map(adc)
    rd_db = 20 * np.log10(rd_full / rd_full.max() + 1e-6)

    # Ограничим диапазон визуализации
    R_max = 14.0
    n_range_show = int(R_max / cfg.range_resolution)
    rd_show = rd_db[:n_range_show, :]
    r_axis = range_axis_m(cfg)[:n_range_show]
    v_axis = doppler_axis_ms(cfg)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    img = ax.imshow(rd_show, origin="lower", cmap="viridis", aspect="auto",
                     extent=[v_axis.min(), v_axis.max(),
                             r_axis.min(), r_axis.max()],
                     vmin=-40, vmax=0)
    for tx, ty, lbl in [(-1.0, 8.0, "Пешеход 1"),
                         (-1.3, 7.7, "Пешеход 2"),
                         (-1.1, 9.5, "Фантомная отметка")]:
        marker = "x" if "Фантом" in lbl else "o"
        color = "red" if "Фантом" in lbl else "white"
        ax.scatter(tx, ty, marker=marker, color=color, edgecolor="black",
                   s=80, label=lbl)
    ax.set_xlabel("Радиальная скорость, м/с")
    ax.set_ylabel("Дальность, м")
    ax.set_title("а) Карта дальность/скорость (FMCW-симуляция, дБ)")
    ax.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Мощность, дБ")

    ax = axes[1]
    radar = np.array([0.0, 0.0])
    left = np.array([-1.5, 8.0])
    right = np.array([1.5, 7.7])
    ax.plot(*radar, "ks", markersize=10); ax.text(0.0, -0.4, "Радар", ha="center")
    ax.scatter(*left, marker="o", color="blue", s=100, label="Левый пешеход")
    ax.scatter(*right, marker="o", color="green", s=100, label="Правый пешеход")
    ax.add_patch(plt.Rectangle((-1.2, 8.8), 2.4, 1.4,
                               fill=False, edgecolor="red", linestyle="--",
                               label="Зона ghost-целей"))
    ax.set_xlim(-4, 4); ax.set_ylim(-1, 12)
    ax.set_xlabel("x, м"); ax.set_ylabel("y, м")
    ax.set_title("б) Геометрия сцены")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out_dir, "fig11_range_doppler_map")


def fig_carrada_real_sample(out_dir: Path, classifier=None, reconstructor=None,
                              geometry=None, device: str = "cpu"):
    """Визуализация реального кадра CARRADA с аннотациями и оценками."""
    carrada_root = Path(__file__).parent / "data" / "carrada" / "Carrada"
    scenes = scan_scenes(carrada_root)
    if not scenes:
        print("  CARRADA not found, skipping fig_carrada_real_sample")
        return None
    scene_dir = carrada_root / scenes[0]
    frames = find_frames_with_targets(scene_dir, max_frames=20)
    if not frames:
        print("  CARRADA no annotated frames, skipping")
        return None
    frame = load_frame(scene_dir, frames[0])
    print(f"  CARRADA: scene {scenes[0]}, frame {frame.frame_id}, "
           f"{len(frame.range_angle_boxes)} target(s)")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    angle_ax = angle_axis_deg(256)
    doppler_ax = carrada_doppler_axis_ms(64)

    # Левый: angle-doppler heatmap из реального радара CARRADA
    ax = axes[0]
    ad = frame.ad_raw
    ad_db = 20 * np.log10(ad / ad.max() + 1e-6)
    img = ax.imshow(ad_db, origin="lower", aspect="auto", cmap="viridis",
                     extent=[doppler_ax.min(), doppler_ax.max(),
                             angle_ax.min(), angle_ax.max()],
                     vmin=-40, vmax=0)
    # Оверлей бокса
    for cls, (r_lo, a_lo, r_hi, a_hi) in frame.range_doppler_boxes:
        d_lo = doppler_ax[a_lo]; d_hi = doppler_ax[a_hi]
        # Use a wide angle range to highlight the doppler bin
        ax.axvspan(d_lo, d_hi, alpha=0.15, color="red",
                   label=f"{CARRADA_CLASS_NAMES.get(cls, cls)} (RD-box)")
    ax.set_xlabel("Радиальная скорость, м/с")
    ax.set_ylabel("Угол, град")
    ax.set_title(f"а) Angle-Doppler карта (CARRADA, кадр {frame.frame_id})")
    ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(img, ax=ax, fraction=0.046, label="Мощность, дБ")

    # Правый: профиль по углу для активного доплеровского бина
    ax = axes[1]
    if frame.range_doppler_boxes:
        cls, (r_lo, a_lo, r_hi, a_hi) = frame.range_doppler_boxes[0]
        active_doppler = (a_lo + a_hi) // 2
    else:
        active_doppler = 32
    angle_profile = ad[:, active_doppler]
    angle_profile_db = 20 * np.log10(angle_profile / angle_profile.max() + 1e-6)
    ax.plot(angle_ax, angle_profile_db, color="navy", lw=1.4,
             label=f"Доплер-бин {active_doppler}")
    for cls, (r_lo, a_lo, r_hi, a_hi) in frame.range_angle_boxes:
        # Angle range in CARRADA box
        a_center = (a_lo + a_hi) / 2
        a_deg = angle_ax[int(np.clip(a_center, 0, 255))]
        ax.axvline(a_deg, color="crimson", linestyle="--", alpha=0.7,
                    label=f"Истинный угол {CARRADA_CLASS_NAMES.get(cls, cls)}")
    ax.set_xlabel("Угол, град"); ax.set_ylabel("Магнитуда, дБ")
    ax.set_title("б) Профиль по углу: пик соответствует аннотации")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-90, 90); ax.set_ylim(-40, 5)

    return _save(fig, out_dir, "fig14_carrada_real")


def fig_carrada_classifier_eval(out_dir: Path, classifier=None, reconstructor=None,
                                  geometry=None, device: str = "cpu"):
    """Качественная валидация на реальных CARRADA-кадрах.

    Поскольку CARRADA хранит магнитуды angle-doppler карт (фаза недоступна),
    мы сравниваем угловой профиль реального радара с аннотацией ground truth
    и проверяем, что пик в угле совпадает с положением реальной цели.
    """
    carrada_root = Path(__file__).parent / "data" / "carrada" / "Carrada"
    scenes = scan_scenes(carrada_root)
    if not scenes:
        print("  CARRADA not found, skipping classifier_eval")
        return None
    scene_dir = carrada_root / scenes[0]
    # Используем только те кадры, для которых есть range_angle_numpy
    import os
    ra_numpy_dir = scene_dir / "range_angle_numpy"
    if not ra_numpy_dir.exists():
        print("  range_angle_numpy not yet extracted, fallback to AD")
        frame_ids = find_frames_with_targets(scene_dir, max_frames=6)
    else:
        ra_set = set(p.stem for p in ra_numpy_dir.glob("*.npy"))
        all_target = find_frames_with_targets(scene_dir, max_frames=400)
        frame_ids = []
        for fid in all_target:
            if fid not in ra_set:
                continue
            try:
                ftmp = load_frame(scene_dir, fid)
                if ftmp.range_angle_boxes:
                    r_lo = ftmp.range_angle_boxes[0][1][0]
                    if r_lo < 80:
                        frame_ids.append(fid)
            except Exception:
                pass
            if len(frame_ids) >= 6:
                break

    angle_ax = angle_axis_deg(256)
    range_ax = carrada_range_axis_m(256)

    if not frame_ids:
        print("  no close-target frames with RA data — skipping")
        return None

    selected = frame_ids[:3]
    rows = len(selected)
    fig, axes = plt.subplots(rows, 2, figsize=(11.5, 3.6 * rows))
    if rows == 1:
        axes = axes[None, :]

    n_hits = 0
    for i, fid in enumerate(selected):
        f = load_frame(scene_dir, fid)
        ra_raw = load_range_angle(scene_dir, fid)
        if ra_raw is None:
            continue
        cls_ra, (r_lo, a_lo, r_hi, a_hi) = f.range_angle_boxes[0]
        cls_label = CARRADA_CLASS_NAMES.get(cls_ra, str(cls_ra))
        r_center = (r_lo + r_hi) // 2
        a_center = (a_lo + a_hi) // 2
        r_gt_m = range_ax[r_center]
        a_gt_deg = angle_ax[a_center]

        # Угловой профиль на дальностном бине цели (срез RA-карты)
        ang_profile = ra_raw[r_center, :]
        ang_db = 20 * np.log10(ang_profile / ang_profile.max() + 1e-6)
        # Сужаем поиск пика до ±60° для устойчивости
        search_mask = np.abs(angle_ax) < 60
        peak_local_idx = int(np.argmax(np.where(search_mask, ang_db, -1e9)))
        peak_deg = angle_ax[peak_local_idx]
        err = abs(peak_deg - a_gt_deg)
        if err < 12:
            n_hits += 1

        # Левая колонка: range-angle карта с GT-боксом
        ax = axes[i, 0]
        ra_db = 20 * np.log10(ra_raw / ra_raw.max() + 1e-6)
        ax.imshow(ra_db, origin="lower", aspect="auto", cmap="viridis",
                   extent=[angle_ax.min(), angle_ax.max(),
                           range_ax.min(), range_ax.max()], vmin=-40, vmax=0)
        # Bounding box
        ax.add_patch(plt.Rectangle(
            (angle_ax[a_lo], range_ax[r_lo]),
            angle_ax[a_hi] - angle_ax[a_lo],
            range_ax[r_hi] - range_ax[r_lo],
            fill=False, edgecolor="red", lw=1.6,
            label=f"GT {cls_label} (r≈{r_gt_m:.1f} м, θ≈{a_gt_deg:+.1f}°)"
        ))
        ax.set_xlim(-60, 60); ax.set_ylim(0, 30)
        ax.set_xlabel("Угол θ, град"); ax.set_ylabel("Дальность, м")
        ax.set_title(f"Range-Angle карта, кадр {fid}")
        ax.legend(loc="upper right", fontsize=8)

        # Правая колонка: угловой профиль на дальности цели
        ax = axes[i, 1]
        ax.plot(angle_ax, ang_db, color="navy", lw=1.2,
                 label=f"Профиль на дальности r = {r_gt_m:.1f} м")
        ax.axvline(peak_deg, color="black", linestyle=":", alpha=0.8,
                    label=f"Пик: θ̂ = {peak_deg:+.1f}°")
        ax.axvline(a_gt_deg, color="red", linestyle="--", alpha=0.7,
                    label=f"GT: θ = {a_gt_deg:+.1f}°")
        ax.set_xlim(-60, 60); ax.set_ylim(-40, 5)
        ax.set_xlabel("Угол θ, град"); ax.set_ylabel("Магнитуда, дБ")
        ax.set_title(f"Ошибка оценки угла: {err:.1f}°")
        ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"Валидация на реальных CARRADA-кадрах: "
                  f"{n_hits} из {rows} кадров — ошибка по углу < 12°",
                  y=1.002, fontsize=12)
    return _save(fig, out_dir, "fig15_carrada_pipeline")


def fig_carrada_error_histogram(out_dir: Path):
    """Гистограмма ошибок оценки угла на всей выборке CARRADA-кадров."""
    import json
    path = Path(__file__).parent.parent / "carrada_eval_results.json"
    if not path.exists():
        print("  carrada_eval_results.json not found, skipping histogram")
        return None
    with open(path) as f:
        data = json.load(f)
    errors = np.array([r["error_deg"] for r in data["details"]])
    summary = data["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.hist(errors, bins=np.arange(0, 60, 3), color="navy", edgecolor="white", alpha=0.8)
    ax.axvline(summary["median_error_deg"], color="crimson", linestyle="--", lw=1.5,
                label=f"медиана = {summary['median_error_deg']:.1f}°")
    ax.axvline(summary["mean_error_deg"], color="darkgreen", linestyle="--", lw=1.5,
                label=f"среднее = {summary['mean_error_deg']:.1f}°")
    ax.axvline(12, color="black", linestyle=":", lw=1.2,
                label="порог 12°")
    ax.set_xlabel("Ошибка оценки угла, град")
    ax.set_ylabel("Количество кадров")
    ax.set_title(f"а) Гистограмма ошибок на {summary['n']} CARRADA-кадрах")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    sorted_err = np.sort(errors)
    cdf = np.arange(1, sorted_err.size + 1) / sorted_err.size * 100
    ax.plot(sorted_err, cdf, color="navy", lw=2)
    ax.axvline(12, color="black", linestyle=":", lw=1.2)
    ax.axhline(summary["hit_rate_below_12deg"] * 100, color="crimson",
                linestyle="--", lw=1.2,
                label=f"{summary['hit_rate_below_12deg']*100:.0f}% < 12°")
    ax.set_xlim(0, 50); ax.set_ylim(0, 102)
    ax.set_xlabel("Ошибка оценки угла, град")
    ax.set_ylabel("Доля кадров, %")
    ax.set_title("б) Кумулятивная функция распределения ошибок")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    return _save(fig, out_dir, "fig16_carrada_error_distribution")


def fig_fmcw_chain(out_dir: Path):
    """Визуализация полного FMCW-конвейера: ADC → Range-FFT → Doppler-FFT."""
    cfg = FMCWConfig()
    rng = np.random.default_rng(101)
    targets = [
        Target(range_m=5.0, velocity_ms=2.0, azimuth_deg=0.0, rcs=0.8),
        Target(range_m=10.0, velocity_ms=-3.0, azimuth_deg=15.0, rcs=0.5),
    ]
    adc = synthesize_adc_frame(targets, cfg, snr_db_per_target=20, rng=rng)
    rd = range_doppler_map(adc)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))

    # Один чирп (вещественная часть IF-сигнала)
    ax = axes[0, 0]
    t = np.arange(cfg.n_samples) / cfg.sample_rate * 1e6
    ax.plot(t, np.real(adc[:, 0, 0, 0]), color="navy", lw=0.8)
    ax.set_title("а) Re ADC-сигнала, один чирп (rx 0, tx 0)")
    ax.set_xlabel("Время, мкс"); ax.set_ylabel("Амплитуда")
    ax.grid(alpha=0.3)

    # Range-FFT для одного чирпа
    ax = axes[0, 1]
    range_fft_one = np.abs(np.fft.fft(np.hanning(cfg.n_samples) * adc[:, 0, 0, 0]))
    range_fft_one = range_fft_one[: cfg.n_samples // 2]
    r_ax = range_axis_m(cfg)[: cfg.n_samples // 2]
    ax.plot(r_ax, 20 * np.log10(range_fft_one / range_fft_one.max() + 1e-9),
            color="darkgreen")
    for tgt in targets:
        ax.axvline(tgt.range_m, color="crimson", linestyle="--", alpha=0.6)
    ax.set_title("б) Range-FFT, один чирп")
    ax.set_xlabel("Дальность, м"); ax.set_ylabel("|·|, дБ")
    ax.set_xlim(0, 15); ax.set_ylim(-40, 5)
    ax.grid(alpha=0.3)

    # Range-Doppler карта
    ax = axes[1, 0]
    rd_db = 20 * np.log10(rd / rd.max() + 1e-6)
    n_show = int(15 / cfg.range_resolution)
    img = ax.imshow(rd_db[:n_show, :], origin="lower", cmap="viridis",
                     aspect="auto",
                     extent=[doppler_axis_ms(cfg).min(),
                             doppler_axis_ms(cfg).max(),
                             0, 15], vmin=-40, vmax=0)
    for tgt in targets:
        ax.scatter(tgt.velocity_ms, tgt.range_m, marker="o", color="white",
                   edgecolor="red", s=60)
    ax.set_title("в) Карта Range-Doppler (дБ)")
    ax.set_xlabel("Радиальная скорость, м/с")
    ax.set_ylabel("Дальность, м")
    fig.colorbar(img, ax=ax, fraction=0.046)

    # Спектр Doppler для бина дальности первой цели
    ax = axes[1, 1]
    r_bin_1 = int(targets[0].range_m / cfg.range_resolution)
    doppler_bin = rd[r_bin_1, :]
    doppler_bin_db = 20 * np.log10(doppler_bin / doppler_bin.max() + 1e-9)
    ax.plot(doppler_axis_ms(cfg), doppler_bin_db, color="purple")
    ax.axvline(targets[0].velocity_ms, color="crimson", linestyle="--", alpha=0.6)
    ax.set_title(f"г) Doppler-FFT для бина дальности {targets[0].range_m:.1f} м")
    ax.set_xlabel("Радиальная скорость, м/с"); ax.set_ylabel("|·|, дБ")
    ax.set_xlim(-cfg.max_velocity, cfg.max_velocity); ax.set_ylim(-40, 5)
    ax.grid(alpha=0.3)

    return _save(fig, out_dir, "fig13_fmcw_chain")


def fig_cartesian_results(reconstructor, geometry, out_dir: Path, device: str):
    rng = np.random.default_rng(29)
    grid_angle = np.linspace(-90, 90, 1801)
    results_music, results_nn = [], []

    targets = [(-1.5, 8.0), (1.5, 7.8)]
    for _ in range(30):
        for tx, ty in targets:
            true_angle = np.degrees(np.arctan2(tx, ty))
            r_true = float(np.hypot(tx, ty))
            x, _ = generate_snapshots(geometry=geometry, scenario="single",
                                      angles_deg=[true_angle], snr_db=12.0,
                                      n_snapshots=8, rng=rng)
            R = sample_covariance(x)
            R_fb = forward_backward_smoothing(R)
            est_m, _ = music_doa(R_fb, geometry, 1, grid_angle)
            inp = torch.tensor(np.stack([R.real, R.imag])[None]
                                .astype(np.float32)).to(device)
            with torch.no_grad():
                R_hat_iq = reconstructor(inp).cpu().numpy()[0]
            R_hat = R_hat_iq[0] + 1j * R_hat_iq[1]
            R_hat = 0.5 * (R_hat + R_hat.conj().T)
            est_n, _ = music_doa(R_hat, geometry, 1, grid_angle)

            for est, results in [(est_m[0], results_music), (est_n[0], results_nn)]:
                jitter = float(rng.normal(0, 0.06))
                r_obs = r_true + jitter
                x_obs = r_obs * np.sin(np.deg2rad(est))
                y_obs = r_obs * np.cos(np.deg2rad(est))
                results.append((x_obs, y_obs))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, results, title in [
        (axes[0], results_music, "а) MUSIC"),
        (axes[1], results_nn, "б) MUSIC + реконструкция R̂"),
    ]:
        xs = [p[0] for p in results]; ys = [p[1] for p in results]
        ax.scatter(xs, ys, alpha=0.6, s=24, color="gray")
        for tx, ty in targets:
            ax.scatter(tx, ty, marker="*", color="crimson", s=180, zorder=3)
        ax.plot(0, 0, "ks", markersize=10); ax.text(0, -0.5, "Радар", ha="center")
        ax.set_xlim(-4, 4); ax.set_ylim(-1, 12); ax.set_aspect("equal")
        ax.set_xlabel("x, м")
        if ax is axes[0]:
            ax.set_ylabel("y, м")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    return _save(fig, out_dir, "fig12_cartesian_results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    out_dir = args.out_dir.resolve()
    print(f"writing figures to {out_dir}")
    geometry = ti_awr1843_geometry()
    M = geometry.n_virtual

    classifier = ScenarioClassifier(n_virtual=M, n_classes=3)
    reconstructor = CovReconstructor(n_virtual=M)
    cls_path = args.checkpoints / "classifier.pt"
    rec_path = args.checkpoints / "reconstructor.pt"
    if cls_path.exists():
        classifier.load_state_dict(torch.load(cls_path, map_location=args.device))
    if rec_path.exists():
        reconstructor.load_state_dict(torch.load(rec_path, map_location=args.device))
    classifier.to(args.device).eval()
    reconstructor.to(args.device).eval()

    fig_array_geometry(out_dir)
    fig_geometry_direct_vs_multipath(out_dir)
    fig_classifier_arch(out_dir)
    fig_reconstructor_arch(out_dir)
    fig_pipeline_overview(out_dir)

    hist_path = args.checkpoints / "history.json"
    if hist_path.exists():
        fig_training_curves(hist_path, out_dir)

    fig_confusion_matrix(classifier, geometry, out_dir, args.device)
    fig_pseudospectrum(reconstructor, geometry, out_dir, args.device)
    fig_rmse_vs_snr(reconstructor, geometry, out_dir, args.device)
    fig_resolution_probability(reconstructor, geometry, out_dir, args.device)
    fig_range_doppler_map(out_dir)
    fig_cartesian_results(reconstructor, geometry, out_dir, args.device)
    fig_fmcw_chain(out_dir)
    fig_carrada_real_sample(out_dir)
    fig_carrada_classifier_eval(out_dir, classifier, reconstructor,
                                  geometry, args.device)


if __name__ == "__main__":
    main()
