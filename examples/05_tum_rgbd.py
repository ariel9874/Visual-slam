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
    parser.add_argument("--matcher", default="ratio",
                        help="ratio (clásico) o lightglue (con --detector superpoint; "
                             "GPU). SuperPoint+LightGlue rescatan las fr1 handheld")
    parser.add_argument("--window", type=int, default=8, help="keyframes del mapa local")
    parser.add_argument("--health", type=int, default=45,
                        help="piso de inliers para insertar KF; 45 va bien en la mayoria. "
                             "Bajar SOLO si hay inanicion de KFs (fr2_desk); ver docs/05 §7")
    parser.add_argument("--ba", default="numpy", choices=["numpy", "gtsam", "isam2"],
                        help="backend del BA local (numpy=referencia didactica)")
    parser.add_argument("--fast", action="store_true",
                        help="stack de TIEMPO REAL de v0.5: isam2 + hilo de mapeo "
                             "(+ C++ guiado y BoW, que ya son auto). ~46 fps en fr2_desk")
    parser.add_argument("--depth", action="store_true",
                        help="modo RGB-D (v0.6): init instantánea + puntos desde "
                             "profundidad; el ATE se reporta MÉTRICO (sin escala)")
    parser.add_argument("--no-ba", action="store_true")
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--config", default=None,
                        help="YAML/JSON con perillas (v0.9; plantilla: "
                             "python -m vslam.config)")
    args = parser.parse_args()

    root = Path(args.root)
    camera = tum_camera(root.name)
    loader = TUMRGBDLoader(root, with_depth=args.depth)
    K, dist = camera.K, camera.dist
    print(f"Secuencia: {root.name} | {len(loader)} frames | "
          f"frontend: {args.detector}+ratio | cam fx={camera.fx:.1f}"
          + (" | RGB-D" if args.depth else ""))
    # Rectificación (v0.6): el mismo mapa de undistort para gris y profundidad,
    # pero la profundidad va con NEAREST — interpolar bilinealmente profundidades
    # a través de una discontinuidad inventa valores que no existen en la escena.
    maps = None
    if camera.has_distortion:
        maps = cv2.initUndistortRectifyMap(
            K, dist, None, K, (camera.width, camera.height), cv2.CV_32FC1)

    ba_backend = "isam2" if args.fast else args.ba
    cfg = None
    if args.config:
        from vslam.config import load_config
        cfg = load_config(args.config)
    tracker = PnPTracker(camera, extractor=create_extractor(args.detector),
                         matcher=create_matcher(args.matcher),
                         local_window=args.window, local_ba=not args.no_ba,
                         loop_closure=not args.no_loop,
                         ba_backend=ba_backend, async_mapping=args.fast,
                         config=cfg)
    # Piso de salud de KF (perilla de re-calibración, v0.45). MEDIDO: es un
    # trade-off dependiente de la secuencia, no una constante universal:
    #   fr1_xyz  → 45: 6.9 cm / 25: 18.4 cm  (bajarlo mete KFs basura, lección 8)
    #   fr2_desk → 45: 1347 perdidos / 25: 278 (subirlo ahoga el mapa, inanición)
    # 45 (= default de sintético) gana en la mayoría; se baja SOLO ante inanición.
    tracker.KF_HEALTH_INLIERS = args.health

    est_ts, est_pos, states = [], [], []
    lost = skipped = 0
    for i, item in enumerate(loader):
        if args.max_frames and i >= args.max_frames:
            break
        ts, gray = item[0], item[1]
        depth = item[2] if args.depth else None
        # RGB-D: la init ESPERA al primer frame con profundidad. Sin esto, si
        # el stream de profundidad arranca tarde (fr1_desk: 6 frames sin
        # pareja depth en la asociación), el tracker cae a la init MONOCULAR
        # y nace un mapa MIXTO — escala gauge del init + metros de los puntos
        # de profundidad — con _metric=False: ni bucle SE(3) ni residuo de
        # profundidad en el BA. Medido en fr1_desk: escala 1.008 de pura
        # casualidad (la mediana del escritorio es ~1 m ≈ el gauge mediana=1).
        if args.depth and depth is None and not tracker._initialized:
            skipped += 1
            continue
        # Pre-rectificación: la geometría del repo asume el modelo ideal.
        if maps is not None:
            rect = cv2.remap(gray, maps[0], maps[1], cv2.INTER_LINEAR)
            if depth is not None:
                depth = cv2.remap(depth, maps[0], maps[1], cv2.INTER_NEAREST)
        else:
            rect = gray
        T, info = tracker.process_frame(rect, depth)
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
    print(f"    frames en INIT: {n_init} (saltados sin depth: {skipped}) | "
          f"perdidos (coast/gate): {lost} | "
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

    # En RGB-D el ATE es MÉTRICO: alineación rígida (sin regalar la escala al
    # alineador). La escala de similitud se reporta aparte: ≈1.000 es la prueba
    # de que el mapa está de verdad en metros.
    with_scale = not args.depth
    label = "METRICO" if args.depth else "similitud"
    if len(est_m) >= 3:
        m = ate(est_m, gt_m, with_scale=with_scale)
        s_check = ate(est_m, gt_m)["scale"]
        print(f"\nATE ONLINE   ({label}): {100*m['rmse']:.1f} cm rmse | "
              f"{100*m['mean']:.1f} mean | {100*m['max']:.1f} max "
              f"| escala similitud {s_check:.3f}")
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
        mk = ate(kf_pos[ok2], gt_pos[a2[ok2]], with_scale=with_scale)
        sk = ate(kf_pos[ok2], gt_pos[a2[ok2]])["scale"]
        print(f"ATE FINAL-KF ({label}): {100*mk['rmse']:.1f} cm rmse | "
              f"{100*mk['mean']:.1f} mean | {100*mk['max']:.1f} max "
              f"| escala similitud {sk:.3f} | {ok2.sum()} KFs")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "trajectory_est.txt",
               np.column_stack([est_ts, est_pos]), fmt="%.6f")
    print(f"trayectoria: {out / 'trajectory_est.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
