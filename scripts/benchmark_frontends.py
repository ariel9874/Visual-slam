#!/usr/bin/env python3
"""Benchmark de frontends: compara detectores/matchers sobre una secuencia con GT.

Corre el MISMO pipeline de VO (examples/01) cambiando solo el frontend, y mide
lo que importa: robustez (frames en coasting), calidad geométrica (inliers),
costo (FPS) y exactitud global (ATE alineado, vslam/evaluation.py). Es la
herramienta para elegir con datos — no con fe — la configuración del frontend
(guía de selección: docs/03_detectores_y_matchers.md).

Uso:
    python scripts/make_synthetic_sequence.py --output data/synthetic
    python scripts/benchmark_frontends.py
    python scripts/benchmark_frontends.py --detectors orb,akaze --matchers ratio,crosscheck
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vslam.core.camera import PinholeCamera
from vslam.core.trajectory import Trajectory
from vslam.evaluation import ate, load_tum_positions
from vslam.frontend.features import create_extractor
from vslam.frontend.matching import create_matcher
from vslam.io.dataset import ImageSequenceLoader


def _load_vo_class():
    """Importa MonocularVO desde examples/01 (nombre con dígito: importlib).

    En v0.2 el tracker migrará a vslam/frontend/tracker.py y este truco
    desaparecerá; mientras tanto, el ejemplo es la única implementación
    (sin duplicar lógica).
    """
    path = REPO / "examples" / "01_monocular_vo.py"
    spec = importlib.util.spec_from_file_location("monocular_vo_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MonocularVO


def run_combo(vo_cls, camera, images_dir, detector: str, matcher: str, max_frames: int):
    """Corre la VO con un frontend concreto y devuelve métricas crudas."""
    extractor = create_extractor(detector)
    matcher_obj = create_matcher(matcher)
    vo = vo_cls(camera, extractor=extractor, matcher=matcher_obj)

    trajectory = Trajectory()
    inliers, matches, coasting, elapsed = [], [], 0, 0.0
    for i, (timestamp, gray) in enumerate(ImageSequenceLoader(images_dir)):
        if max_frames and i >= max_frames:
            break
        t0 = time.perf_counter()
        T_w_c, info = vo.process_frame(gray)
        elapsed += time.perf_counter() - t0
        trajectory.append(timestamp, T_w_c)
        if i > 0:  # el primer frame solo fija el origen
            inliers.append(info["n_inliers"])
            matches.append(info["n_matches"])
            coasting += 0 if info["tracked"] else 1
    n = len(trajectory)
    return {
        "positions": trajectory.positions,
        "frames": n,
        "fps": n / elapsed if elapsed > 0 else float("inf"),
        "mean_matches": sum(matches) / max(len(matches), 1),
        "mean_inliers": sum(inliers) / max(len(inliers), 1),
        "coasting": coasting,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--images", default="data/synthetic/images")
    parser.add_argument("--calib", default="data/synthetic/calib.txt")
    parser.add_argument("--gt", default="data/synthetic/groundtruth.txt")
    parser.add_argument("--detectors",
                        default="orb,gftt-orb,brisk,akaze,sift,kaze",
                        help="lista separada por comas (los aprendidos se "
                             "saltan limpiamente si faltan dependencias)")
    parser.add_argument("--matchers", default="ratio",
                        help="lista separada por comas")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    camera = PinholeCamera.from_file(args.calib)
    gt = load_tum_positions(args.gt)
    if args.max_frames:
        gt = gt[: args.max_frames]
    vo_cls = _load_vo_class()

    header = (f"{'detector':<10} {'matcher':<10} {'matches':>8} {'inliers':>8} "
              f"{'coast':>6} {'fps':>7} {'ATE cm':>8} {'ATE %':>7}")
    print(header)
    print("-" * len(header))

    for detector in args.detectors.split(","):
        for matcher in args.matchers.split(","):
            detector, matcher = detector.strip(), matcher.strip()
            try:
                r = run_combo(vo_cls, camera, args.images, detector, matcher,
                              args.max_frames)
                m = ate(r["positions"], gt[: r["frames"]])
                print(f"{detector:<10} {matcher:<10} {r['mean_matches']:>8.0f} "
                      f"{r['mean_inliers']:>8.0f} {r['coasting']:>6d} "
                      f"{r['fps']:>7.1f} {100 * m['rmse']:>8.1f} "
                      f"{m['rmse_pct']:>6.1f}%")
            except ImportError as exc:
                print(f"{detector:<10} {matcher:<10} SKIP: {str(exc).splitlines()[0]}")
            except Exception as exc:  # un frontend roto no debe tumbar la tabla
                print(f"{detector:<10} {matcher:<10} ERROR: {exc}")

    print("\nATE: RMSE tras alineación de similitud (Umeyama) contra ground truth;")
    print("'coast' = frames donde el tracking falló y se usó velocidad constante.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
