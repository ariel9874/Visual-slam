#!/usr/bin/env python3
"""
Ejemplo 04 — Cierre de bucle visual de punta a punta (v0.35)
============================================================

El sistema completo trabajando junto, sobre imágenes reales (sintéticas):

    frontend (PnP, ej. 02) + BA local (backend) + MAPA LOCAL
        → la deriva reaparece (¡a propósito!)
    reconocimiento de lugar + verificación PnP + grafo de poses (ej. 03)
        → la deriva se corrige al re-visitar el inicio

El mapa LOCAL (matching solo contra los últimos N keyframes) es la decisión
de ingeniería clave: acota el costo del tracking (no crece con el recorrido)
a cambio de re-introducir deriva — y con ella, la necesidad del cierre de
bucle. Es el trato que hacen todos los SLAM reales.

Este ejemplo corre la MISMA secuencia dos veces (con y sin cierre de bucle)
e imprime el ATE de ambas. La secuencia debe re-visitar el inicio:

    python scripts/make_synthetic_sequence.py --output data/synthetic_loop --motion loop --frames 140
    python examples/04_loop_closure.py --images data/synthetic_loop/images \
        --calib data/synthetic_loop/calib.txt --gt data/synthetic_loop/groundtruth.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera
from vslam.core.trajectory import Trajectory
from vslam.evaluation import ate, load_tum_positions
from vslam.frontend.features import available_extractors, create_extractor
from vslam.frontend.matching import create_matcher
from vslam.frontend.tracker import PnPTracker
from vslam.io.dataset import ImageSequenceLoader


def run(args, camera, loop_closure: bool):
    """Una pasada completa; devuelve (posiciones, primer frame trackeado, tracker)."""
    tracker = PnPTracker(camera,
                         extractor=create_extractor(args.detector),
                         matcher=create_matcher("ratio"),
                         local_window=args.window,
                         local_ba=not args.no_ba,
                         loop_closure=loop_closure)
    trajectory = Trajectory()
    first_track = None
    for i, (timestamp, gray) in enumerate(ImageSequenceLoader(args.images)):
        if args.max_frames and i >= args.max_frames:
            break
        T_w_c, info = tracker.process_frame(gray)
        trajectory.append(timestamp, T_w_c)
        if first_track is None and info["state"] == "INIT-OK":
            first_track = i
        if "LOOP" in info["state"]:
            print(f"    frame {i}: BUCLE cerrado contra el keyframe {tracker.loop_events[-1][1]}")
    return trajectory.positions, first_track or 0, tracker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--images", default="data/synthetic_loop/images")
    parser.add_argument("--calib", default="data/synthetic_loop/calib.txt")
    parser.add_argument("--gt", default="data/synthetic_loop/groundtruth.txt")
    parser.add_argument("--output", default="output/loop")
    parser.add_argument("--detector", default="orb", choices=available_extractors())
    parser.add_argument("--window", type=int, default=4, help="keyframes del mapa local")
    parser.add_argument("--no-ba", action="store_true", help="desactiva el BA local")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    camera = PinholeCamera.from_file(args.calib)
    gt = load_tum_positions(args.gt)

    # Prints en ASCII puro: la consola de Windows (cp1252) rechaza el resto.
    print("[1/2] SIN cierre de bucle (solo mapa local)")
    pos_off, start_off, _ = run(args, camera, loop_closure=False)
    m_off = ate(pos_off[start_off:], gt[start_off:len(pos_off)])

    print("[2/2] CON cierre de bucle")
    pos_on, start_on, tracker = run(args, camera, loop_closure=True)
    m_on = ate(pos_on[start_on:], gt[start_on:len(pos_on)])

    print(f"\nATE sin bucle: {100 * m_off['rmse']:6.1f} cm ({m_off['rmse_pct']:.1f}%)")
    print(f"ATE con bucle: {100 * m_on['rmse']:6.1f} cm ({m_on['rmse_pct']:.1f}%)"
          f"  | bucles cerrados: {len(tracker.loop_events)}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from vslam.evaluation import umeyama_alignment

        # La ida-y-vuelta se solapa sobre sí misma en planta: la historia se
        # cuenta mejor como SERIE TEMPORAL (x por frame y error por frame).
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax1.plot(range(len(gt)), gt[:, 0], "--", color="0.5", lw=2,
                 label="ground truth")
        for pos, start, label, color in [
            (pos_off, start_off, "sin cierre de bucle", "tab:red"),
            (pos_on, start_on, "con cierre de bucle", "tab:green"),
        ]:
            s, R, t = umeyama_alignment(pos[start:], gt[start:len(pos)])
            aligned = (s * (R @ pos[start:].T)).T + t
            frames = range(start, len(pos))
            ax1.plot(frames, aligned[:, 0], color=color, lw=1.4, label=label)
            err = np.linalg.norm(aligned - gt[start:len(pos)], axis=1)
            ax2.plot(frames, 100 * err, color=color, lw=1.4)
        for f, kf in tracker.loop_events:
            ax2.axvline(f, color="tab:green", ls=":", alpha=0.7)
            ax2.text(f, ax2.get_ylim()[1] * 0.9, f" bucle vs KF{kf}",
                     fontsize=8, color="tab:green")
        ax1.set_ylabel("x [m]"), ax1.legend(), ax1.grid(alpha=0.3)
        ax1.set_title("Mapa local + cierre de bucle (ida-y-vuelta por un corredor)")
        ax2.set_xlabel("frame"), ax2.set_ylabel("error [cm]"), ax2.grid(alpha=0.3)
        fig.savefig(out / "loop_comparison.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"gráfica: {out / 'loop_comparison.png'}")
    except ImportError:
        print("[aviso] matplotlib no instalado: se omite la gráfica")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
