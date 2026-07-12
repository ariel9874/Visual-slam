#!/usr/bin/env python3
"""Ejemplo 08 — Mapa denso 3DGS EN VIVO junto al tracker (v0.7 hito 5)
=====================================================================

La segunda mitad del criterio de v0.7: el tracking NO pierde frames por culpa
del mapper denso. Aquí el `GaussianSplattingMapper` corre en su PROPIO hilo
(`DenseMappingThread`, el tercer hilo de la arquitectura ORB-SLAM): el hilo de
tracking solo ENCOLA cada keyframe nuevo (una copia de memoria) y el worker
integra, siembra y optimiza con el presupuesto que sobra.

La medición del criterio: correr con --dense y sin él, y comparar frames
procesados (deben ser LOS MISMOS) y latencia por frame (mediana/p99):

    python examples/08_live_dense_mapping.py --root data/tum/rgbd_dataset_freiburg1_desk
    python examples/08_live_dense_mapping.py --root ... --dense --backend gsplat

(gsplat solo en el contenedor — lección 40; en Windows nativo usar
--backend tiled con --scale 4, más lento de converger pero mismo contrato.)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera
from vslam.frontend.features import create_extractor
from vslam.frontend.matching import create_matcher
from vslam.frontend.tracker import PnPTracker
from vslam.io.dataset import TUMRGBDLoader, tum_camera


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--root", required=True)
    parser.add_argument("--dense", action="store_true",
                        help="activa el mapeo denso 3DGS (en PROCESO propio)")
    parser.add_argument("--use-thread", action="store_true",
                        help="hilo en vez de proceso (leccion 42: el GIL "
                             "duplica la latencia del tracking — para comparar)")
    parser.add_argument("--backend", choices=["reference", "tiled", "gsplat"],
                        default="gsplat")
    parser.add_argument("--scale", type=int, default=2,
                        help="reduccion de las vistas de supervision del mapper")
    parser.add_argument("--seed-step", type=int, default=4)
    parser.add_argument("--chunk-iters", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--settle", type=float, default=0.0,
                        help="segundos extra de optimizacion tras la secuencia")
    args = parser.parse_args()

    root = Path(args.root)
    camera = tum_camera(root.name)
    loader = TUMRGBDLoader(root, with_depth=True)
    maps = cv2.initUndistortRectifyMap(camera.K, camera.dist, None, camera.K,
                                       (camera.width, camera.height), cv2.CV_32FC1)
    tracker = PnPTracker(camera, extractor=create_extractor("orb"),
                         matcher=create_matcher("ratio"),
                         local_window=8, local_ba=True, loop_closure=True)

    dense = None
    sc = args.scale
    hs, ws = camera.height // sc, camera.width // sc
    if args.dense:
        cam_s = PinholeCamera(fx=camera.fx / sc, fy=camera.fy / sc,
                              cx=camera.cx / sc, cy=camera.cy / sc,
                              width=ws, height=hs)
        if args.use_thread:
            from vslam.mapping.gaussian import GaussianSplattingMapper
            from vslam.mapping.dense_thread import DenseMappingThread
            mapper = GaussianSplattingMapper(cam_s, backend=args.backend)
            dense = DenseMappingThread(mapper, cam_s, seed_step=args.seed_step,
                                       chunk_iters=args.chunk_iters,
                                       depth_min=tracker.DEPTH_MIN,
                                       depth_max=tracker.DEPTH_MAX)
        else:
            # PROCESO (lección 42): sin GIL compartido, torch/CUDA solo en el hijo.
            from vslam.mapping.dense_thread import DenseMappingProcess
            dense = DenseMappingProcess(cam_s, backend=args.backend,
                                        seed_step=args.seed_step,
                                        chunk_iters=args.chunk_iters,
                                        depth_min=tracker.DEPTH_MIN,
                                        depth_max=tracker.DEPTH_MAX)

    mode = "OFF" if dense is None else ("hilo" if args.use_thread else "proceso")
    print(f"Secuencia: {root.name} | dense={mode}"
          f"{f' ({args.backend}, {ws}x{hs})' if args.dense else ''}", flush=True)
    times, n_frames, n_kfs_seen = [], 0, 0
    for i, (ts, gray, depth) in enumerate(loader):
        if args.max_frames and i >= args.max_frames:
            break
        if depth is None and not tracker._initialized:
            continue
        rect = cv2.remap(gray, maps[0], maps[1], cv2.INTER_LINEAR)
        drect = (cv2.remap(depth, maps[0], maps[1], cv2.INTER_NEAREST)
                 if depth is not None else None)
        t0 = time.perf_counter()
        tracker.process_frame(rect, drect)
        times.append(time.perf_counter() - t0)
        n_frames += 1
        # ¿Se promovió un keyframe? → encolarlo al hilo denso (barato).
        if dense is not None and len(tracker._kf_ids) > n_kfs_seen:
            n_kfs_seen = len(tracker._kf_ids)
            kf_id = tracker._kf_ids[-1]
            img_s = cv2.resize(rect, (ws, hs), interpolation=cv2.INTER_AREA)
            d_s = (cv2.resize(drect, (ws, hs), interpolation=cv2.INTER_NEAREST)
                   if drect is not None else None)
            dense.submit(kf_id, img_s, d_s, tracker.T_w_c)

    ms = np.array(times) * 1000
    print(f"\nTRACKING: {n_frames} frames | mediana {np.median(ms):.1f} ms | "
          f"p99 {np.percentile(ms, 99):.1f} ms | fps {1000/np.mean(ms):.1f} | "
          f"keyframes {len(tracker._kf_ids)} | perdidos {tracker.lost_frames if hasattr(tracker, 'lost_frames') else 'n/a'}",
          flush=True)

    if dense is not None:
        if args.settle > 0:
            time.sleep(args.settle)
        # Cierre de bucle/GBA en caliente: re-anclar el mapa denso a las poses
        # finales ANTES de medir. Viaja por la cola del worker (serializado:
        # mutar el mapa desde fuera chocaria con un optimize en vuelo).
        dense.update_poses(dict(tracker.keyframe_trajectory()))
        s = dense.stop()
        print(f"DENSO: integrados {s['integrated']}/{n_kfs_seen} KFs | "
              f"{s['opt_iters']} iters de optimizacion en vivo | "
              f"{s['n_gaussians']} gaussianas | PSNR {s['psnr']:.1f} dB | "
              f"fallos {s['failures']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
