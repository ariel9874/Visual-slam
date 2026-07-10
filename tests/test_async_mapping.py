#!/usr/bin/env python3
"""Test del HILO DE MAPEO (v0.5, async_mapping=True).

Corre el corredor sintético completo con el mapeo (BA + cierre de bucle +
culling) en el worker y verifica el contrato de la arquitectura de dos hilos:
(a) el ATE se mantiene (paridad con el modo síncrono, tolerancia RANSAC);
(b) los bucles SE CIERRAN y su corrección llega al tracking (vía el delta
    pendiente — sin él, la trayectoria posterior quedaría en el marco viejo);
(c) el worker no sufre excepciones (map_failures == 0) y se apaga limpio.
Requiere gtsam (usa ba_backend="isam2", el emparejamiento natural del hilo);
se salta limpio sin él. Regenera data/synthetic_loop si falta.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera
from vslam.evaluation import ate, load_tum_positions
from vslam.frontend.features import create_extractor
from vslam.frontend.matching import create_matcher
from vslam.frontend.tracker import PnPTracker
from vslam.io.dataset import ImageSequenceLoader

DATA = Path("data/synthetic_loop")


def _has_gtsam() -> bool:
    try:
        import gtsam  # noqa: F401
        return True
    except ImportError:
        return False


def _ensure_data() -> None:
    if (DATA / "images").is_dir() and (DATA / "groundtruth.txt").is_file():
        return
    print("[setup] generando", DATA, "...")
    subprocess.run([sys.executable, "scripts/make_synthetic_sequence.py",
                    "--output", str(DATA), "--motion", "loop",
                    "--frames", "200"], check=True)


def test_async_mapping_corridor():
    camera = PinholeCamera.from_file(str(DATA / "calib.txt"))
    gt = load_tum_positions(str(DATA / "groundtruth.txt"))
    tracker = PnPTracker(camera, extractor=create_extractor("orb"),
                         matcher=create_matcher("ratio"),
                         local_window=4, local_ba=True, loop_closure=True,
                         ba_backend="isam2", async_mapping=True)
    positions, first = [], None
    for i, (ts, gray) in enumerate(ImageSequenceLoader(str(DATA / "images"))):
        T, info = tracker.process_frame(gray)
        positions.append(T[:3, 3].copy())
        if first is None and info["state"] == "INIT-OK":
            first = i
    tracker.wait_mapping()

    # (c) worker sano y apagado limpio.
    assert tracker.map_failures == 0, f"{tracker.map_failures} excepciones en el worker"
    tracker.stop_mapping()
    assert not tracker._map_thread.is_alive(), "el worker no se apagó"

    # (b) los bucles se cerraron desde el worker.
    assert len(tracker.loop_events) >= 1, "ningún bucle cerrado en modo async"

    # (a) ATE online razonable (paridad con sync ~1.7-2.2 cm; margen RANSAC).
    positions = np.array(positions)
    first = first or 0
    m = ate(positions[first:], gt[first:len(positions)])
    assert m["rmse"] < 0.05, f"ATE async: {100*m['rmse']:.1f} cm (>5)"


def main() -> int:
    if not _has_gtsam():
        print("SKIP: gtsam no instalado (el hilo de mapeo usa isam2).")
        return 0
    _ensure_data()
    test_async_mapping_corridor()
    print("OK: el hilo de mapeo pasa (ATE, bucles via delta, worker limpio).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
