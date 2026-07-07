#!/usr/bin/env python3
"""
Benchmark batch sobre TUM RGB-D (v0.45)
=======================================

Corre el sistema sobre todas las secuencias TUM presentes en una carpeta y
emite una TABLA reproducible (ATE, frames perdidos, keyframes, bucles, relocs).
Es el deliverable de "benchmark batch por secuencia" de la hoja de ruta (v0.45).

    python scripts/benchmark_tum.py --data data/tum
    python scripts/benchmark_tum.py --data data/tum --max-frames 500   # humo rapido

Nota medida (docs/05 §7): con MATCHING GUIADO por reproyección (v0.45) el piso
de salud de keyframe (KF_HEALTH_INLIERS=45) ya no necesita ajuste por-secuencia
— el guiado sube los inliers y la inanición desaparece (fr2_desk: 105→22 cm,
1347→0 frames perdidos). --health-overrides queda por si se corre SIN guiado.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.evaluation import ate
from vslam.frontend.features import create_extractor
from vslam.frontend.matching import create_matcher
from vslam.frontend.tracker import PnPTracker
from vslam.io.dataset import (TUMRGBDLoader, associate_by_timestamp,
                              read_tum_trajectory, tum_camera)


def run_sequence(root: Path, detector: str, window: int, health: int,
                 max_frames: int) -> dict:
    camera = tum_camera(root.name)
    K, dist = camera.K, camera.dist
    tracker = PnPTracker(camera, extractor=create_extractor(detector),
                         matcher=create_matcher("ratio"),
                         local_window=window, local_ba=True, loop_closure=True)
    tracker.KF_HEALTH_INLIERS = health

    est_ts, est_pos, states = [], [], []
    for i, (ts, gray) in enumerate(TUMRGBDLoader(root)):
        if max_frames and i >= max_frames:
            break
        rect = cv2.undistort(gray, K, dist) if camera.has_distortion else gray
        T, info = tracker.process_frame(rect)
        est_ts.append(ts); est_pos.append(T[:3, 3].copy()); states.append(info["state"])
    est_ts = np.array(est_ts); est_pos = np.array(est_pos)

    gt_ts, gt_pos = read_tum_trajectory(root / "groundtruth.txt")
    start = next((i for i, s in enumerate(states) if s.startswith(("INIT-OK", "TRACK"))), 0)
    assoc = associate_by_timestamp(est_ts[start:], gt_ts, max_dt=0.02)
    ok = assoc >= 0
    m = ate(est_pos[start:][ok], gt_pos[assoc[ok]]) if ok.sum() >= 3 else {"rmse": float("nan")}

    # BA global offline + trayectoria FINAL de keyframes — la métrica del sistema.
    tracker.global_bundle_adjustment()
    kf_traj = tracker.keyframe_trajectory()
    kf_ts = np.array([est_ts[k] for k, _ in kf_traj])
    kf_pos = np.array([T[:3, 3] for _, T in kf_traj])
    a2 = associate_by_timestamp(kf_ts, gt_ts, max_dt=0.05)
    ok2 = a2 >= 0
    mk = ate(kf_pos[ok2], gt_pos[a2[ok2]]) if ok2.sum() >= 3 else {"rmse": float("nan")}
    return {
        "frames": len(est_ts),
        "ate_kf_cm": 100 * mk["rmse"],
        "ate_online_cm": 100 * m["rmse"],
        "lost": sum(1 for s in states if s in ("COAST", "GATE-REJECT")),
        "kfs": len(tracker._kf_ids),
        "loops": len(tracker.loop_events),
        "relocs": len(tracker.reloc_events),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--data", default="data/tum", help="carpeta con rgbd_dataset_*")
    parser.add_argument("--detector", default="orb")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--health", type=int, default=45)
    parser.add_argument("--health-overrides", default="",
                        help="ajustes por-secuencia 'substr:valor,...' (con matching "
                             "guiado ya no hace falta bajar el piso; ver docs/05)")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    overrides = {}
    for tok in filter(None, args.health_overrides.split(",")):
        sub, val = tok.split(":")
        overrides[sub] = int(val)

    seqs = sorted(p for p in Path(args.data).glob("rgbd_dataset_*") if p.is_dir())
    if not seqs:
        print(f"[error] no hay secuencias rgbd_dataset_* en {args.data}")
        return 1

    print(f"{'secuencia':32s} {'frames':>7} {'ATE-KF':>7} {'online':>7} "
          f"{'perdidos':>9} {'KFs':>5} {'bucles':>7} {'relocs':>7}")
    print("-" * 92)
    for root in seqs:
        health = next((v for sub, v in overrides.items() if sub in root.name), args.health)
        try:
            r = run_sequence(root, args.detector, args.window, health, args.max_frames)
        except Exception as exc:                       # noqa: BLE001
            print(f"{root.name:32s}  FALLO: {exc}")
            continue
        print(f"{root.name:32s} {r['frames']:7d} {r['ate_kf_cm']:7.1f} {r['ate_online_cm']:7.1f} "
              f"{r['lost']:9d} {r['kfs']:5d} {r['loops']:7d} {r['relocs']:7d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
