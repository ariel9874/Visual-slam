#!/usr/bin/env python3
"""
Ejemplo 06 — Datos REALES: EuRoC MAV (monocular) (v0.45)
=======================================================

Segundo dataset real, tras TUM (ejemplo 05). EuRoC es un DRON (Machine Hall /
Vicon Room) con movimiento agresivo 6-DoF: más rotación y velocidad que el
escritorio de TUM, buen estrés para la relocalización y la compuerta.

Dos particularidades que resuelve el loader (vslam/io/dataset.py):
  - Cámara en `mav0/cam0/sensor.yaml` (intrínsecos + distorsión radial-tangencial
    + extrínseco T_BS). Se pre-rectifica cada imagen como en el ejemplo 05.
  - El GROUND TRUTH está en el frame del CUERPO (IMU): se transforma al frame de
    la cámara con T_BS antes de comparar (si no, un error de brazo de palanca que
    rota con la pose contamina el ATE). Lo hace `read_euroc_groundtruth`.

    python examples/06_euroc.py --root data/euroc/MH_01_easy

Se reporta el ATE de la trayectoria FINAL de keyframes (bucles + BA global
offline, la métrica del sistema — ver ejemplo 05 y docs/05 lección 25).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.evaluation import ate
from vslam.frontend.features import available_extractors, create_extractor
from vslam.frontend.matching import create_matcher
from vslam.frontend.tracker import PnPTracker
from vslam.io.dataset import (EuRoCLoader, associate_by_timestamp,
                              euroc_camera, read_euroc_groundtruth)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", required=True, help="carpeta de la secuencia EuRoC")
    parser.add_argument("--output", default="output/euroc")
    parser.add_argument("--detector", default="orb", choices=available_extractors())
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--health", type=int, default=45)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    camera = euroc_camera(root)
    loader = EuRoCLoader(root)
    K, dist = camera.K, camera.dist
    print(f"Secuencia: {root.name} | {len(loader)} frames | "
          f"frontend: {args.detector}+ratio | cam fx={camera.fx:.1f}")

    tracker = PnPTracker(camera, extractor=create_extractor(args.detector),
                         matcher=create_matcher("ratio"),
                         local_window=args.window, local_ba=True, loop_closure=True)
    tracker.KF_HEALTH_INLIERS = args.health

    est_ts, est_pos, states = [], [], []
    lost = 0
    for i, (ts, gray) in enumerate(loader):
        if args.max_frames and i >= args.max_frames:
            break
        rect = cv2.undistort(gray, K, dist) if camera.has_distortion else gray
        T, info = tracker.process_frame(rect)
        est_ts.append(ts); est_pos.append(T[:3, 3].copy()); states.append(info["state"])
        if info["state"] in ("COAST", "GATE-REJECT"):
            lost += 1
        if "LOOP" in info["state"]:
            print(f"    frame {i}: BUCLE cerrado vs KF {tracker.loop_events[-1][1]}")
    est_ts = np.array(est_ts); est_pos = np.array(est_pos)
    print(f"    perdidos {lost} | bucles {len(tracker.loop_events)} | "
          f"relocs {len(tracker.reloc_events)}")

    gt_ts, gt_pos = read_euroc_groundtruth(root)
    start = next((i for i, s in enumerate(states) if s.startswith(("INIT-OK", "TRACK"))), 0)
    a = associate_by_timestamp(est_ts[start:], gt_ts, max_dt=0.02)
    ok = a >= 0
    if ok.sum() >= 3:
        m = ate(est_pos[start:][ok], gt_pos[a[ok]])
        print(f"\nATE ONLINE   : {100*m['rmse']:.1f} cm rmse | escala {m['scale']:.3f}")

    # BA global offline + trayectoria FINAL de keyframes (la métrica del sistema).
    tracker.global_bundle_adjustment()
    kf_traj = tracker.keyframe_trajectory()
    kf_ts = np.array([est_ts[k] for k, _ in kf_traj])
    kf_pos = np.array([T[:3, 3] for _, T in kf_traj])
    a2 = associate_by_timestamp(kf_ts, gt_ts, max_dt=0.05)
    ok2 = a2 >= 0
    if ok2.sum() >= 3:
        mk = ate(kf_pos[ok2], gt_pos[a2[ok2]])
        print(f"ATE FINAL-KF : {100*mk['rmse']:.1f} cm rmse | escala {mk['scale']:.3f} "
              f"| {ok2.sum()} KFs")

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "trajectory_est.txt", np.column_stack([est_ts, est_pos]), fmt="%.6f")
    print(f"trayectoria: {out / 'trajectory_est.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
