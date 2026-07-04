#!/usr/bin/env python3
"""
Ejemplo 03 — Grafo de poses y cierre de bucle (backend, v0.3)
=============================================================

El experimento canónico de los backends de SLAM, con todas las piezas a la
vista y sin necesitar imágenes:

  1. Una trayectoria REAL en bucle (círculo) — el ground truth.
  2. Una odometría imperfecta: cada paso lleva un pequeño sesgo de guiñada y
     ruido. Integrada, produce la clásica "banana": la deriva crece sin
     límite y el bucle no cierra (es lo que le pasa a los ejemplos 01/02
     en recorridos largos).
  3. UN solo factor de cierre de bucle (el último frame re-observa el
     primero) + optimización del grafo de poses: el error se REDISTRIBUYE
     por toda la cadena y la trayectoria se "cose".
  4. El mapa se deforma en consecuencia: los puntos anclados a cada keyframe
     se re-anclan con la corrección de su pose (MapperBase.update_poses —
     exactamente el mecanismo que 3DGS-SLAM aplica a sus submapas).

Dónde está cada matemática:
    vslam/core/lie.py               Exp/Log de SE(3) (variedades ↔ vectores)
    vslam/backend/factor_graph.py   MAP → mínimos cuadrados, factores, gauge
    vslam/backend/pose_graph.py     whitening, jacobianos, LM, kernel Huber
    vslam/mapping/sparse.py         re-anclaje rígido de puntos por keyframe

Uso:
    python examples/03_pose_graph_loop.py --output output/pose_graph
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.backend.pose_graph import GaussNewtonPoseGraph
from vslam.core.frame import Frame
from vslam.core.geometry import invert_se3
from vslam.core.lie import se3_exp
from vslam.mapping.sparse import SparsePointMapper


def circle_pose(k: int, n: int, radius: float) -> np.ndarray:
    """Pose k de un círculo en el plano x-z con la guiñada tangente."""
    a = 2 * np.pi * k / n
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    T[:3, 3] = [radius * np.sin(a), 0.0, radius * (1 - np.cos(a))]
    return T


def rmse(traj_a, traj_b) -> float:
    d = np.array([np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) for Ta, Tb in zip(traj_a, traj_b)])
    return float(np.sqrt((d ** 2).mean()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--output", default="output/pose_graph")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # 1) Ground truth: un bucle perfecto.
    gt = [circle_pose(k, args.frames, args.radius) for k in range(args.frames)]

    # 2) Odometría imperfecta: T̂_k,k+1 = T_rel · Exp(sesgo + ruido).
    #    El sesgo de guiñada (0.4°/paso) es lo que curva la "banana".
    bias = np.array([0.0, 0.0, 0.0, 0.0, np.deg2rad(0.4), 0.0])
    noise_std = np.array([0.01, 0.002, 0.01, 0.001, 0.002, 0.001])
    odometry = [gt[0]]
    measurements = []
    for k in range(args.frames - 1):
        T_rel = invert_se3(gt[k]) @ gt[k + 1]
        T_meas = T_rel @ se3_exp(bias + rng.normal(0, 1, 6) * noise_std)
        measurements.append(T_meas)
        odometry.append(odometry[-1] @ T_meas)

    # Un mapa "construido" por el sistema drifteado: 6 puntos por keyframe,
    # anclados a la pose DERIVADA (donde el sistema cree que está).
    mapper = SparsePointMapper()
    for k, T in enumerate(odometry):
        mapper.integrate_keyframe(Frame(frame_id=k, timestamp=float(k), T_w_c=T,
                                        is_keyframe=True))
        local = rng.uniform([-1.5, -0.5, 2.0], [1.5, 0.5, 5.0], (6, 3))
        world = (T[:3, :3] @ local.T).T + T[:3, 3]
        mapper.add_points(world, np.zeros((6, 32), np.uint8), anchor_kf_id=k)
    map_before = mapper.get_map().copy()

    # 3) Grafo de poses: odometría + UN factor de bucle, y a optimizar.
    #    La información codifica la confianza: el bucle (medición directa
    #    entre frames lejanos, sin acumulación) pesa mucho más que un paso.
    graph = GaussNewtonPoseGraph()
    graph.add_pose(0, odometry[0], fixed=True)          # ancla el gauge
    for k in range(1, args.frames):
        graph.add_pose(k, odometry[k])                  # guess = odometría
    for k, T_meas in enumerate(measurements):
        graph.add_odometry_factor(k, k + 1, T_meas, np.eye(6) * 1e2)
    T_loop = invert_se3(gt[-1]) @ gt[0]                 # re-observación limpia
    graph.add_loop_factor(args.frames - 1, 0, T_loop, np.eye(6) * 1e4)

    optimized = graph.optimize(iterations=30)
    corrected = [optimized[k] for k in range(args.frames)]

    # 4) El mapa se deforma con sus keyframes (re-anclaje rígido).
    mapper.update_poses(optimized)
    map_after = mapper.get_map()

    print(f"ATE de la odometría (deriva):   {rmse(odometry, gt):7.3f} m")
    print(f"ATE tras el cierre de bucle:    {rmse(corrected, gt):7.3f} m")
    gap_b = np.linalg.norm(odometry[-1][:3, 3] - gt[-1][:3, 3])
    gap_a = np.linalg.norm(corrected[-1][:3, 3] - gt[-1][:3, 3])
    # (sin caracteres fuera de cp1252: la consola de Windows los rechaza)
    print(f"error del ultimo frame:         {gap_b:7.3f} m -> {gap_a:.3f} m")

    # Gráfica: la historia completa en una imagen.
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        for traj, style, label in [
            (gt, dict(color="0.6", lw=2, ls="--"), "ground truth"),
            (odometry, dict(color="tab:red", lw=1.5), "odometría (deriva)"),
            (corrected, dict(color="tab:green", lw=1.8), "optimizada (bucle cerrado)"),
        ]:
            p = np.array([T[:3, 3] for T in traj])
            ax.plot(p[:, 0], p[:, 2], label=label, **style)
        ax.scatter(map_before[:, 0], map_before[:, 2], s=4, c="tab:red", alpha=0.25,
                   label="mapa antes")
        ax.scatter(map_after[:, 0], map_after[:, 2], s=4, c="tab:green", alpha=0.35,
                   label="mapa re-anclado")
        ax.set_xlabel("x [m]"), ax.set_ylabel("z [m]")
        ax.set_title("Cierre de bucle: el error se redistribuye por el grafo")
        ax.axis("equal"), ax.grid(alpha=0.3), ax.legend(loc="upper right", fontsize=9)
        fig.savefig(out / "loop_closure.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"gráfica: {out / 'loop_closure.png'}")
    except ImportError:
        print("[aviso] matplotlib no instalado: se omite la gráfica")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
