#!/usr/bin/env python3
"""
Ejemplo 07 — Mapa denso 3DGS sobre TUM RGB-D (v0.7)
===================================================

La tesis de la arquitectura (docs/01 §3.2): el SLAM geométrico da la ESTRUCTURA
(poses métricas + nube dispersa, v0.6) y el `GaussianSplattingMapper` la vuelve
FOTO-REALISTA — sin tocar frontend ni backend, solo cambiando el mapper detrás
de `MapperBase`.

Flujo:
  1. Trackea fr1/desk en RGB-D (mapa MÉTRICO, ejemplo 05 --depth) y guarda la
     imagen de cada keyframe.
  2. BA global offline → poses finales de keyframes.
  3. SIEMBRA gaussianas desde la nube dispersa (media = punto 3D; color =
     muestra de la imagen del keyframe ancla).
  4. OPTIMIZA las gaussianas re-renderizando contra los keyframes (renderiza y
     compara) y reporta el PSNR medio — el criterio de v0.7 (> 30 dB).

    python examples/07_gaussian_mapping.py --root data/tum/rgbd_dataset_freiburg1_desk

NOTA de rendimiento: el rasterizador de referencia (gaussian_render.py) es denso
—O(N·H·W)— así que aquí se renderiza a resolución REDUCIDA (--scale). La ruta
full-res es la gemela gsplat (tiles + CUDA), pendiente como la fue el C++ del
matching o el GTSAM del BA (regla 3 de la hoja de ruta).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera
from vslam.frontend.features import create_extractor
from vslam.frontend.matching import create_matcher
from vslam.frontend.tracker import PnPTracker
from vslam.io.dataset import TUMRGBDLoader, tum_camera
from vslam.mapping.gaussian import GaussianSplattingMapper


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="output/gaussian")
    parser.add_argument("--scale", type=int, default=4,
                        help="factor de reducción del render (4 = 160x120)")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--max-points", type=int, default=8000,
                        help="tope de gaussianas (memoria del render denso); bajar si OOM")
    parser.add_argument("--seed-step", type=int, default=2,
                        help="submuestreo de la rejilla de siembra por profundidad")
    parser.add_argument("--backend", choices=["reference", "tiled", "gsplat"],
                        default="tiled",
                        help="render: reference (denso, O(N.H.W)), tiled (PyTorch por tiles, memoria acotada) o gsplat (tiles+CUDA)")
    parser.add_argument("--refine-poses", action="store_true",
                        help="delta SE(3) por keyframe junto al mapa (leccion 41: "
                             "el ATE de ~cm son PIXELES a 1 m; sin refinar, blur)")
    parser.add_argument("--exposure", action="store_true",
                        help="gana/sesgo afin por keyframe (auto-exposicion de TUM)")
    parser.add_argument("--densify-every", type=int, default=0,
                        help="clonar/dividir donde el gradiente acumulado es alto y "
                             "podar opacidades muertas, cada N iters (0 = off)")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    camera = tum_camera(root.name)
    loader = TUMRGBDLoader(root, with_depth=True)
    maps = cv2.initUndistortRectifyMap(camera.K, camera.dist, None, camera.K,
                                       (camera.width, camera.height), cv2.CV_32FC1)
    tracker = PnPTracker(camera, extractor=create_extractor("orb"),
                         matcher=create_matcher("ratio"),
                         local_window=8, local_ba=True, loop_closure=True)

    # 1) Trackea y guarda la imagen (reducida) de CADA frame por su índice del
    #    tracker (= kf_id cuando ese frame se promueve a keyframe).
    sc = args.scale
    hs, ws = camera.height // sc, camera.width // sc
    images: dict = {}
    depths: dict = {}
    print(f"Secuencia: {root.name} | RGB-D | render {ws}x{hs} (scale {sc}) | backend {args.backend}",
          flush=True)
    for i, (ts, gray, depth) in enumerate(loader):
        if args.max_frames and i >= args.max_frames:
            break
        if depth is None and not tracker._initialized:
            continue
        rect = cv2.remap(gray, maps[0], maps[1], cv2.INTER_LINEAR)
        drect = (cv2.remap(depth, maps[0], maps[1], cv2.INTER_NEAREST)
                 if depth is not None else None)
        tracker.process_frame(rect, drect)
        idx = tracker._frame_idx
        images[idx] = cv2.resize(rect, (ws, hs), interpolation=cv2.INTER_AREA)
        if drect is not None:
            depths[idx] = cv2.resize(drect, (ws, hs), interpolation=cv2.INTER_NEAREST)
    tracker.global_bundle_adjustment()
    kf_traj = dict(tracker.keyframe_trajectory())
    print(f"    keyframes: {len(kf_traj)} | con profundidad: "
          f"{sum(1 for k in kf_traj if k in depths)}", flush=True)

    # 2) Cámara REDUCIDA (los intrínsecos se escalan con la imagen).
    cam_s = PinholeCamera(fx=camera.fx / sc, fy=camera.fy / sc,
                          cx=camera.cx / sc, cy=camera.cy / sc,
                          width=ws, height=hs)
    mapper = GaussianSplattingMapper(cam_s, init_scale=0.03, backend=args.backend)

    # 3) Keyframes de supervisión.
    from vslam.core.frame import Frame
    for kf_id, T in kf_traj.items():
        if kf_id in images:
            mapper.integrate_keyframe(Frame(frame_id=kf_id, timestamp=0.0,
                                            image=images[kf_id], T_w_c=T))

    # 4) SIEMBRA DENSA desde la profundidad de cada keyframe (estilo RGB-D 3DGS,
    #    SplaTAM/MonoGS): retro-proyectar una rejilla submuestreada de píxeles
    #    con profundidad válida da MILES de gaussianas bien colocadas y coloreadas
    #    — la nube dispersa del SLAM (lección 39: 1968 pts → 15 dB) es demasiado
    #    rala para foto-realismo. Cada gaussiana se ancla a SU keyframe.
    fx_s, fy_s, cx_s, cy_s = cam_s.fx, cam_s.fy, cam_s.cx, cam_s.cy
    step = args.seed_step
    grid_v, grid_u = np.mgrid[0:hs:step, 0:ws:step]
    gu, gv = grid_u.ravel(), grid_v.ravel()
    pos_l, col_l, anc_l, sc_l = [], [], [], []
    for kf_id, T in kf_traj.items():
        d = depths.get(kf_id)
        if d is None:
            continue
        z = d[gv, gu].astype(np.float32)
        ok = (z > tracker.DEPTH_MIN) & (z < tracker.DEPTH_MAX)
        if not ok.any():
            continue
        u, v, z = gu[ok], gv[ok], z[ok]
        Xc = np.stack([(u - cx_s) / fx_s * z, (v - cy_s) / fy_s * z, z], axis=1)
        pos_l.append((T[:3, :3] @ Xc.T).T + T[:3, 3])
        col_l.append(images[kf_id][v, u].astype(np.float32) / 255.0)
        anc_l.append(np.full(len(z), kf_id))
        # Escala inicial POR PUNTO = huella de la celda de siembra en el mundo,
        # step·z/fx (el "vecino más cercano" del 3DGS original). Una escala fija
        # sobredimensionada es un techo de blur: medido 15.5 dB con 3 cm a full-res.
        sc_l.append(step * z / fx_s)
    pos = np.concatenate(pos_l)
    colors = np.concatenate(col_l)
    anchors = np.concatenate(anc_l)
    seed_scales = np.concatenate(sc_l)
    if len(pos) > args.max_points:                   # tope por memoria (render denso)
        sel = np.random.default_rng(0).choice(len(pos), args.max_points, replace=False)
        pos, colors, anchors = pos[sel], colors[sel], anchors[sel]
        seed_scales = seed_scales[sel]
    print(f"    gaussianas sembradas: {len(pos)} | escala mediana "
          f"{np.median(seed_scales)*100:.2f} cm", flush=True)
    mapper.add_points(pos, colors[:, None], anchors, scales=seed_scales)

    # 5) Optimiza y mide.
    db0 = mapper.mean_psnr()
    print(f"    PSNR de siembra: {db0:.1f} dB -> optimizando {args.iters} iters"
          f"{' + poses' if args.refine_poses else ''}"
          f"{' + exposicion' if args.exposure else ''}...", flush=True)
    db = mapper.optimize(iters=args.iters, log_every=max(1, args.iters // 10),
                         refine_poses=args.refine_poses, exposure=args.exposure,
                         densify_every=args.densify_every)
    per_kf = list(mapper.psnr_per_kf().values())
    print(f"\nPSNR medio de re-render: {db0:.1f} dB (siembra) -> {db:.1f} dB "
          f"(tras {args.iters} iters)  [criterio v0.7: > 30]")
    print(f"    por keyframe: min {min(per_kf):.1f} | mediana "
          f"{float(np.median(per_kf)):.1f} | max {max(per_kf):.1f}  "
          f"(dispersion = inconsistencia multi-vista)")

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    kf0 = next(iter(kf_traj))
    T_kf0 = mapper._kfs[kf0]["T"].cpu().numpy()      # pose refinada si la hubo
    render = mapper.render_view(T_kf0, hs, ws)
    cv2.imwrite(str(out / "render_kf0.png"), (render[..., 0] * 255).astype(np.uint8))
    cv2.imwrite(str(out / "target_kf0.png"), images[kf0])
    print(f"render de ejemplo: {out / 'render_kf0.png'} (vs target_kf0.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
