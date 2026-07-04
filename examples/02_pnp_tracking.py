#!/usr/bin/env python3
"""
Ejemplo 02 — Tracking 3D-2D con mapa disperso (PnP)
===================================================

El salto conceptual respecto al ejemplo 01:

    Ejemplo 01 (2D-2D):  frame ←→ frame anterior. Cada paso re-estima la
        geometría desde cero, con dirección ruidosa y ESCALA re-sorteada
        (||t|| = 1 en cada par) → el error se INTEGRA y la trayectoria
        zigzaguea (míralo en el benchmark).

    Ejemplo 02 (3D-2D):  frame ←→ MAPA persistente. El sistema:
        1. Inicializa una sola vez con 2D-2D + TRIANGULACIÓN (DLT) y fija el
           gauge de escala (profundidad mediana = 1).
        2. Cada frame se localiza contra el mapa con PnP: una medición
           ABSOLUTA en el marco del mapa (los errores no se apilan igual) y
           con la escala heredada del mapa (sin deriva de escala por frame).
        3. Cuando los puntos visibles se agotan, promueve un KEYFRAME y
           triangula puntos nuevos: el mapa crece con el recorrido.

La implementación vive en el paquete (¡ya no inline!): este ejemplo muestra
cómo se usa y dónde mirar la matemática de cada pieza:

    vslam/frontend/tracker.py   PnPTracker: ciclo de vida, gauge de escala
    vslam/core/geometry.py      triangulación DLT + filtros, PnP + Rodrigues
    vslam/mapping/sparse.py     mapa disperso, anclaje y re-anclaje de puntos

Uso:
    python examples/02_pnp_tracking.py --images data/synthetic/images \
        --calib data/synthetic/calib.txt --output output/pnp \
        --gt data/synthetic/groundtruth.txt [--detector orb] [--show]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera
from vslam.core.trajectory import Trajectory
from vslam.evaluation import ate, load_tum_positions
from vslam.frontend.features import available_extractors, create_extractor
from vslam.frontend.matching import available_matchers, create_matcher
from vslam.frontend.tracker import PnPTracker
from vslam.io.dataset import ImageSequenceLoader
from vslam.mapping.sparse import SparsePointMapper


def draw_residuals(gray: np.ndarray, info: dict) -> np.ndarray:
    """Dibuja el residuo de reproyección de los inliers: punto del mapa
    proyectado (rojo) → observación en el frame (verde). Cuanto más cortas
    las líneas, mejor explica el mapa lo que la cámara ve."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if info["pts_prev"] is not None:
        for (x0, y0), (x1, y1) in zip(info["pts_prev"], info["pts_curr"]):
            cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)
            cv2.circle(vis, (int(x1), int(y1)), 2, (0, 255, 0), -1)
    cv2.putText(vis, f"{info['state']}  inliers {info['n_inliers']}  mapa {info['n_map']} pts",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return vis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--images", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--output", default="output/pnp")
    parser.add_argument("--gt", default="", help="ground truth TUM: imprime el ATE al final")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--detector", default="orb", choices=available_extractors())
    parser.add_argument("--matcher", default="ratio", choices=available_matchers())
    args = parser.parse_args()

    camera = PinholeCamera.from_file(args.calib)
    loader = ImageSequenceLoader(args.images)
    mapper = SparsePointMapper()
    tracker = PnPTracker(camera,
                         extractor=create_extractor(args.detector),
                         matcher=create_matcher(args.matcher),
                         mapper=mapper)
    print(f"Secuencia: {len(loader)} imágenes | frontend: {args.detector}+{args.matcher}")

    trajectory = Trajectory()
    first_track = None   # frame donde el sistema quedó inicializado
    for i, (timestamp, gray) in enumerate(loader):
        if args.max_frames and i >= args.max_frames:
            break
        T_w_c, info = tracker.process_frame(gray)
        trajectory.append(timestamp, T_w_c)
        if first_track is None and info["state"] == "INIT-OK":
            first_track = i

        if i % 20 == 0 or "KF" in info["state"] or info["state"] == "COAST":
            x, y, z = T_w_c[:3, 3]
            print(f"frame {i:5d} | {info['state']:8s} | inliers {info['n_inliers']:4d} "
                  f"| mapa {info['n_map']:5d} pts | pos [{x:+7.2f} {y:+7.2f} {z:+7.2f}]")

        if args.show:
            cv2.imshow("vslam - ejemplo 02 (ESC para salir)", draw_residuals(gray, info))
            if cv2.waitKey(1) == 27:
                break
    if args.show:
        cv2.destroyAllWindows()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    trajectory.save_tum(out / "trajectory.txt")
    mapper.save_ply(out / "map.ply")
    print(f"\nTrayectoria ({len(trajectory)} poses): {out / 'trajectory.txt'}")
    print(f"Mapa disperso ({len(mapper)} puntos):   {out / 'map.ply'}  (ábrelo en MeshLab)")

    if args.gt:
        # Se evalúa DESDE la inicialización: antes de ella el sistema espera
        # paralaje anclado al origen (por diseño) y esos frames no miden nada.
        start = first_track or 0
        gt = load_tum_positions(args.gt)[: len(trajectory)]
        m = ate(trajectory.positions[start:], gt[start:])
        print(f"ATE vs ground truth (desde init en frame {start}): "
              f"rmse {100 * m['rmse']:.1f} cm ({m['rmse_pct']:.1f}% del recorrido) "
              f"| max {100 * m['max']:.1f} cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
