"""Количественная валидация на всех доступных CARRADA-кадрах с RA-данными.

Для каждого кадра, имеющего range_angle_numpy и аннотацию автомобиля/пешехода/
велосипедиста близко к радару (r ≤ 100 бинов ≈ 20 м), вычисляется:
  • peak_angle — пик углового профиля на дальностном бине цели;
  • gt_angle — центр аннотированного bounding-box по углу;
  • error = |peak - gt|.

Результат: статистики ошибок по выборке и распределение по классам.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from mimo_doa.carrada_loader import (
    CARRADA_CLASS_NAMES,
    angle_axis_deg,
    find_frames_with_targets,
    load_frame,
    load_range_angle,
    range_axis_m,
    scan_scenes,
)


def evaluate_scene(scene_dir: Path,
                    r_bin_max: int = 100,
                    search_angle_deg: float = 60.0) -> list[dict]:
    """Прогон по всем кадрам сцены, у которых есть RA numpy и близкая цель."""
    ra_dir = scene_dir / "range_angle_numpy"
    if not ra_dir.exists():
        return []
    ra_set = {p.stem for p in ra_dir.glob("*.npy")}

    angle_ax = angle_axis_deg(256)
    range_ax = range_axis_m(256)
    target_frames = find_frames_with_targets(scene_dir, max_frames=10_000)
    results = []
    for fid in target_frames:
        if fid not in ra_set:
            continue
        try:
            frame = load_frame(scene_dir, fid)
        except Exception:
            continue
        if not frame.range_angle_boxes:
            continue
        cls, (r_lo, a_lo, r_hi, a_hi) = frame.range_angle_boxes[0]
        if r_lo >= r_bin_max:
            continue
        ra = load_range_angle(scene_dir, fid)
        if ra is None:
            continue
        r_center = int(np.clip((r_lo + r_hi) // 2, 0, 255))
        a_center = int(np.clip((a_lo + a_hi) // 2, 0, 255))
        ang_profile = ra[r_center, :]
        ang_db = 20.0 * np.log10(ang_profile / ang_profile.max() + 1e-9)
        mask = np.abs(angle_ax) <= search_angle_deg
        peak_idx = int(np.argmax(np.where(mask, ang_db, -1e9)))
        peak_deg = float(angle_ax[peak_idx])
        gt_deg = float(angle_ax[a_center])
        err = abs(peak_deg - gt_deg)
        results.append({
            "scene": scene_dir.name,
            "frame": fid,
            "class": int(cls),
            "class_name": CARRADA_CLASS_NAMES.get(int(cls), str(cls)),
            "range_m": float(range_ax[r_center]),
            "gt_angle_deg": gt_deg,
            "peak_angle_deg": peak_deg,
            "error_deg": float(err),
        })
    return results


def summarize(results: list[dict]) -> dict:
    """Сводка статистик по выборке результатов."""
    if not results:
        return {"n": 0}
    errors = np.array([r["error_deg"] for r in results])
    by_class = {}
    for r in results:
        by_class.setdefault(r["class_name"], []).append(r["error_deg"])
    return {
        "n": int(errors.size),
        "mean_error_deg": float(errors.mean()),
        "median_error_deg": float(np.median(errors)),
        "std_error_deg": float(errors.std()),
        "p90_error_deg": float(np.percentile(errors, 90)),
        "hit_rate_below_12deg": float((errors < 12).mean()),
        "hit_rate_below_15deg": float((errors < 15).mean()),
        "by_class": {
            name: {
                "n": len(errs),
                "mean_deg": float(np.mean(errs)),
                "median_deg": float(np.median(errs)),
            } for name, errs in by_class.items()
        },
    }


def main():
    root = Path(__file__).parent / "data" / "carrada" / "Carrada"
    scenes = scan_scenes(root)
    print(f"найдено сцен: {len(scenes)}")
    all_results = []
    for s in scenes:
        scene_results = evaluate_scene(root / s)
        print(f"  сцена {s}: {len(scene_results)} кадров")
        all_results.extend(scene_results)

    summary = summarize(all_results)
    out_path = Path(__file__).parent.parent / "carrada_eval_results.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "details": all_results}, f, indent=2,
                   ensure_ascii=False)
    print(f"\n=== Сводка ===")
    for k, v in summary.items():
        if k == "by_class":
            for name, st in v.items():
                print(f"  {name:12s}: n={st['n']}, mean={st['mean_deg']:.2f}°, "
                       f"median={st['median_deg']:.2f}°")
        else:
            print(f"  {k:24s}: {v}")
    print(f"\nрезультаты сохранены: {out_path}")


if __name__ == "__main__":
    main()
