#!/usr/bin/env python3
"""
Ejemplo 05 — Datos REALES: TUM RGB-D (monocular) (v0.45)
========================================================

El primer contacto del sistema con imágenes de una cámara de verdad. Hasta v0.4
todo se midió en secuencias sintéticas (geometría exacta, sin distorsión, sin
rolling shutter, matching fácil). Aquí llega lo que el mundo real añade:

  - DISTORSIÓN de lente: se pre-rectifica cada imagen (cv2.undistort) con los
    coeficientes Brown-Conrady de la cámara Freiburg antes de tocar la geometría
    (la alternativa por-keypoint es PinholeCamera.undistort_points).
  - TIMESTAMPS reales: el ground truth de la mocap corre a otra frecuencia que
    el RGB; se asocia por timestamp más cercano (associate_by_timestamp).
  - Textura, iluminación y movimiento reales: es donde los umbrales calibrados
    en sintético (docs/05 §3.4) se ponen a prueba y, casi seguro, piden ajuste.

    python examples/05_tum_rgbd.py --root data/tum/rgbd_dataset_freiburg2_desk

Es un BRING-UP: el objetivo es que el sistema recorra la secuencia sin perderse
y dar un ATE medible, no todavía un número pulido (la re-calibración es el
siguiente paso, con barridos documentados).
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
from vslam.io.dataset import (TUMRGBDLoader, associate_by_timestamp,
                              read_tum_trajectory, tum_camera)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", required=True, help="carpeta de la secuencia TUM")
    parser.add_argument("--output", default="output/tum")
    parser.add_argument("--detector", default="orb", choices=available_extractors())
    parser.add_argument("--window", type=int, default=8, help="keyframes del mapa local")
    parser.add_argument("--health", type=int, default=45,
                        help="piso de inliers para insertar KF; 45 va bien en la mayoria. "
                             "Bajar SOLO si hay inanicion de KFs (fr2_desk); ver docs/05 §7")
    parser.add_argument("--no-ba", action="store_true")
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    camera = tum_camera(root.name)
    loader = TUMRGBDLoader(root)
    K, dist = camera.K, camera.dist
    print(f"Secuencia: {root.name} | {len(loader)} frames | "
          f"frontend: {args.detector}+ratio | cam fx={camera.fx:.1f}")

    tracker = PnPTracker(camera, extractor=create_extractor(args.detector),
                         matcher=create_matcher("ratio"),
                         local_window=args.window, local_ba=not args.no_ba,
                         loop_closure=not args.no_loop)
    # Piso de salud de KF (perilla de re-calibración, v0.45). MEDIDO: es un
    # trade-off dependiente de la secuencia, no una constante universal:
    #   fr1_xyz  → 45: 6.9 cm / 25: 18.4 cm  (bajarlo mete KFs basura, lección 8)
    #   fr2_desk → 45: 1347 perdidos / 25: 278 (subirlo ahoga el mapa, inanición)
    # 45 (= default de sintético) gana en la mayoría; se baja SOLO ante inanición.
    tracker.KF_HEALTH_INLIERS = args.health

    est_ts, est_pos, states = [], [], []
    lost = 0
    for i, (ts, gray) in enumerate(loader):
        if args.max_frames and i >= args.max_frames:
            break
        # Pre-rectificación: la geometría del repo asume el modelo ideal.
        rect = cv2.undistort(gray, K, dist) if camera.has_distortion else gray
        T, info = tracker.process_frame(rect)
        est_ts.append(ts)
        est_pos.append(T[:3, 3].copy())
        states.append(info["state"])
        if info["state"] in ("COAST", "GATE-REJECT"):
            lost += 1
        if "LOOP" in info["state"]:
            print(f"    frame {i}: BUCLE cerrado vs KF {tracker.loop_events[-1][1]}")
        if "RELOC" == info["state"]:
            print(f"    frame {i}: RELOC vs KF {tracker.reloc_events[-1][1]}")

    est_ts = np.array(est_ts)
    est_pos = np.array(est_pos)
    n_init = sum(1 for s in states if s == "INIT")
    print(f"    frames en INIT: {n_init} | perdidos (coast/gate): {lost} | "
          f"bucles: {len(tracker.loop_events)} | relocs: {len(tracker.reloc_events)}")

    # Evaluación: asociar el GT de la mocap a los frames trackeados.
    gt_ts, gt_pos = read_tum_trajectory(root / "groundtruth.txt")
    # Solo evaluamos desde que el mapa existe (tras INIT) hasta el final.
    start = next((i for i, s in enumerate(states) if s.startswith("INIT-OK")
                  or s.startswith("TRACK")), 0)
    assoc = associate_by_timestamp(est_ts[start:], gt_ts, max_dt=0.02)
    ok = assoc >= 0
    est_m = est_pos[start:][ok]
    gt_m = gt_pos[assoc[ok]]
    print(f"    frames evaluados (con GT): {ok.sum()} / {len(est_ts) - start}")

    if len(est_m) >= 3:
        m = ate(est_m, gt_m)
        print(f"\nATE ONLINE   : {100*m['rmse']:.1f} cm rmse | {100*m['mean']:.1f} mean | "
              f"{100*m['max']:.1f} max | escala {m['scale']:.3f}")
    else:
        print("\n[fallo] muy pocos frames con GT asociado para evaluar ATE")
        return 1

    # Refinamiento OFFLINE: un BA global sobre todo el mapa tras la secuencia
    # (los bucles ya registraron las observaciones puente que atan la escala).
    tracker.global_bundle_adjustment()
    # Trayectoria FINAL de keyframes (poses optimizadas) — la métrica estándar
    # del sistema completo, no la online (ver keyframe_trajectory / lección 25).
    kf_traj = tracker.keyframe_trajectory()
    kf_ts = np.array([est_ts[k] for k, _ in kf_traj])
    kf_pos = np.array([T[:3, 3] for _, T in kf_traj])
    a2 = associate_by_timestamp(kf_ts, gt_ts, max_dt=0.05)
    ok2 = a2 >= 0
    if ok2.sum() >= 3:
        mk = ate(kf_pos[ok2], gt_pos[a2[ok2]])
        print(f"ATE FINAL-KF : {100*mk['rmse']:.1f} cm rmse | {100*mk['mean']:.1f} mean | "
              f"{100*mk['max']:.1f} max | escala {mk['scale']:.3f} | {ok2.sum()} KFs")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "trajectory_est.txt",
               np.column_stack([est_ts, est_pos]), fmt="%.6f")
    print(f"trayectoria: {out / 'trajectory_est.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
