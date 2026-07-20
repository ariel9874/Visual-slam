#!/usr/bin/env python3
"""
Ejemplo 06 — Datos REALES: EuRoC MAV (monocular o ESTÉREO) (v0.45 / v0.6)
========================================================================

Segundo dataset real, tras TUM (ejemplo 05). EuRoC es un DRON (Machine Hall /
Vicon Room) con movimiento agresivo 6-DoF: más rotación y velocidad que el
escritorio de TUM, buen estrés para la relocalización y la compuerta.

Dos particularidades que resuelve el loader (vslam/io/dataset.py):
  - Cámara en `mav0/cam0/sensor.yaml` (intrínsecos + distorsión radial-tangencial
    + extrínseco T_BS). Se pre-rectifica cada imagen como en el ejemplo 05.
  - El GROUND TRUTH está en el frame del CUERPO (IMU): se transforma al frame de
    la cámara con T_BS antes de comparar (si no, un error de brazo de palanca que
    rota con la pose contamina el ATE). Lo hace `read_euroc_groundtruth`.

    python examples/06_euroc.py --root data/euroc/MH_01_easy            # monocular
    python examples/06_euroc.py --root data/euroc/MH_01_easy --stereo   # ESTÉREO métrico

--stereo (v0.6): usa cam0+cam1, rectifica el par, triangula profundidad por
disparidad (StereoSGBM) y alimenta la MISMA ruta RGB-D métrica del tracker (la
cámara derecha VIRTUAL de RGB-D se vuelve REAL). El ATE se mide con alineación
RÍGIDA (sin regalar escala) — la escala de similitud ≈ 1.0 confirma metros de
verdad. En monocular la escala es gauge y se alinea con similitud.

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
from vslam.io.dataset import (EuRoCLoader, EuRoCStereoLoader,
                              associate_by_timestamp, euroc_camera,
                              read_euroc_groundtruth)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", required=True, help="carpeta de la secuencia EuRoC")
    parser.add_argument("--output", default="output/euroc")
    parser.add_argument("--detector", default="orb", choices=available_extractors())
    parser.add_argument("--matcher", default="ratio", help="ratio o lightglue (GPU)")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--health", type=int, default=45)
    parser.add_argument("--stereo", action="store_true",
                        help="estéreo métrico (cam0+cam1, disparidad → profundidad)")
    parser.add_argument("--ba", default="numpy", choices=["numpy", "gtsam", "isam2"],
                        help="backend del BA local (como en examples/05)")
    parser.add_argument("--fast", action="store_true",
                        help="stack de tiempo real de v0.5: iSAM2 + hilo de mapeo (gtsam)")
    parser.add_argument("--imu", action="store_true",
                        help="VIO (v1.1): factor IMU en iSAM2; requiere --stereo y "
                             "--fast o --ba isam2")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    ba_backend = "isam2" if args.fast else args.ba
    if args.imu and not (args.stereo and ba_backend == "isam2"):
        parser.error("--imu requiere --stereo y el backend isam2 "
                     "(--fast o --ba isam2: el grafo VI vive en iSAM2)")

    root = Path(args.root)
    # ESTÉREO: la cámara es la izquierda RECTIFICADA (del rig); la profundidad
    # viene por disparidad. MONOCULAR: cam0 con su distorsión, pre-rectificada.
    if args.stereo:
        loader = EuRoCStereoLoader(root)
        camera = loader.camera
        metric = True
    else:
        loader = EuRoCLoader(root)
        camera = euroc_camera(root)
        metric = False
    K, dist = camera.K, camera.dist
    print(f"Secuencia: {root.name} | {len(loader)} frames | "
          f"frontend: {args.detector}+ratio | cam fx={camera.fx:.1f}"
          + (f" | ESTÉREO (bf={loader.stereo_bf:.1f})" if args.stereo else ""))

    tracker = PnPTracker(camera, extractor=create_extractor(args.detector),
                         matcher=create_matcher(args.matcher),
                         local_window=args.window, local_ba=True, loop_closure=True,
                         ba_backend=ba_backend, async_mapping=args.fast)
    tracker.KF_HEALTH_INLIERS = args.health
    if args.stereo:
        tracker.STEREO_BF = loader.stereo_bf   # el MISMO bf que sintetiza u_R
        tracker.DEPTH_MAX = loader.max_depth   # escenas de sala: metros, no cm

    # ── VIO (v1.1 hito 3b): el DRIVER es el dueño del reloj ──────────────────
    # El tracker no conoce timestamps (deuda consciente): aquí se leen el IMU
    # crudo y la init estática (lección 48), se lleva la gravedad al frame del
    # MAPA (la izquierda RECTIFICADA del rig — el mapa ancla en ella) y se le
    # da al tracker un PROVEEDOR de segmentos entre frame-ids.
    ts_by_frame: dict = {}
    if args.imu:
        from vslam.backend.imu_init import static_imu_init
        from vslam.backend.imu_preintegration import ImuNoiseParams
        from vslam.io.dataset import euroc_imu_params, read_euroc_imu
        imu_ts, imu_gyro, imu_accel = read_euroc_imu(root)
        imu_noise = ImuNoiseParams(**euroc_imu_params(root))
        vi_init = static_imu_init(imu_ts, imu_gyro, imu_accel)
        T_cam_imu = loader.rig.T_cam_imu
        g_map = T_cam_imu[:3, :3] @ vi_init.gravity_body
        # (el dron sigue QUIETO entre la ventana estática y el frame 0: la
        #  actitud de la ventana vale para el ancla del mapa)

        def imu_segment(kf_a: int, kf_b: int):
            """(ts, gyro, accel) del intervalo [t(kf_a), t(kf_b)): los cortes
            en searchsorted hacen que segmentos consecutivos PARTICIONEN el
            tiempo (sin huecos ni solapes; borde ≤ 5 ms a 200 Hz)."""
            i0 = int(np.searchsorted(imu_ts, ts_by_frame[kf_a]))
            i1 = int(np.searchsorted(imu_ts, ts_by_frame[kf_b]))
            return imu_ts[i0:i1 + 1], imu_gyro[i0:i1], imu_accel[i0:i1]

        tracker.enable_imu(imu_noise, g_map, T_cam_imu=T_cam_imu,
                           init_gyro_bias=vi_init.gyro_bias,
                           segment_provider=imu_segment)
        print(f"    VIO: ventana estatica [{vi_init.t_start - imu_ts[0]:.1f}, "
              f"{vi_init.t_end - imu_ts[0]:.1f}] s | b_g inicial "
              f"{np.round(vi_init.gyro_bias, 4)} | |g_map| {np.linalg.norm(g_map):.2f}")

    est_ts, est_pos, states = [], [], []
    lost = 0
    for i, item in enumerate(loader):
        if args.max_frames and i >= args.max_frames:
            break
        ts, gray = item[0], item[1]
        depth = item[2] if args.stereo else None
        ts_by_frame[i] = ts              # ANTES de process_frame: el worker
        #                                  puede pedir el segmento de este id
        if args.stereo:
            rect = gray                         # el loader ya entrega rectificado
        else:
            rect = cv2.undistort(gray, K, dist) if camera.has_distortion else gray
        T, info = tracker.process_frame(rect, depth)
        est_ts.append(ts); est_pos.append(T[:3, 3].copy()); states.append(info["state"])
        if info["state"] in ("COAST", "GATE-REJECT"):
            lost += 1
        if "LOOP" in info["state"]:
            print(f"    frame {i}: BUCLE cerrado vs KF {tracker.loop_events[-1][1]}")
    est_ts = np.array(est_ts); est_pos = np.array(est_pos)
    print(f"    perdidos {lost} | bucles {len(tracker.loop_events)} | "
          f"relocs {len(tracker.reloc_events)}")

    # ESTÉREO = mapa MÉTRICO → ATE rígido (sin regalar escala); la escala de
    # similitud es el chequeo. MONOCULAR = escala gauge → alinear con similitud.
    unit = "MÉTRICO" if metric else "escala libre"
    gt_ts, gt_pos = read_euroc_groundtruth(root)
    start = next((i for i, s in enumerate(states) if s.startswith(("INIT-OK", "TRACK"))), 0)
    a = associate_by_timestamp(est_ts[start:], gt_ts, max_dt=0.02)
    ok = a >= 0
    if ok.sum() >= 3:
        m = ate(est_pos[start:][ok], gt_pos[a[ok]], with_scale=not metric)
        s = ate(est_pos[start:][ok], gt_pos[a[ok]])["scale"]
        print(f"\nATE ONLINE   ({unit}): {100*m['rmse']:.1f} cm rmse | escala similitud {s:.3f}")

    # BA global offline + trayectoria FINAL de keyframes (la métrica del sistema).
    tracker.global_bundle_adjustment()
    kf_traj = tracker.keyframe_trajectory()
    kf_ts = np.array([est_ts[k] for k, _ in kf_traj])
    kf_pos = np.array([T[:3, 3] for _, T in kf_traj])
    a2 = associate_by_timestamp(kf_ts, gt_ts, max_dt=0.05)
    ok2 = a2 >= 0
    if ok2.sum() >= 3:
        mk = ate(kf_pos[ok2], gt_pos[a2[ok2]], with_scale=not metric)
        sk = ate(kf_pos[ok2], gt_pos[a2[ok2]])["scale"]
        print(f"ATE FINAL-KF ({unit}): {100*mk['rmse']:.1f} cm rmse | "
              f"escala similitud {sk:.3f} | {ok2.sum()} KFs")

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "trajectory_est.txt", np.column_stack([est_ts, est_pos]), fmt="%.6f")
    print(f"trayectoria: {out / 'trajectory_est.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
