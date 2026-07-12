#!/usr/bin/env python3
"""Test de estrés de la concurrencia (v0.9 hito 3).

Los tres frentes concurrentes del sistema y su contrato:
  (1) HILO DE MAPEO (v0.5, async_mapping): el worker hace BA/bucles mientras
      el tracking sigue insertando — cero excepciones (map_failures == 0) y
      el run termina (sin deadlock del _map_lock).
  (2) LECTORES EXTERNOS (el patrón de los nodos ROS, v0.8): otro hilo pide
      keyframe_trajectory()/get_map() EN CALIENTE mientras se trackea — las
      lecturas van bajo el lock del mapper y no deben romper ni bloquear.
  (3) El RESET de mapa (v0.9) en pleno vuelo con async_mapping: el caso feo
      (el worker puede tener un KF viejo entre manos) — las excepciones del
      worker se capturan (map_failures las cuenta) y el tracker se recupera.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera
from vslam.frontend.tracker import PnPTracker

W, H = 320, 240
CAM = PinholeCamera(fx=260.0, fy=260.0, cx=W / 2, cy=H / 2, width=W, height=H)


def _scene(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), np.uint8)
    for _ in range(220):
        u, v = rng.integers(6, W - 6), rng.integers(6, H - 6)
        img[v - 3:v + 3, u - 3:u + 3] = rng.integers(80, 255)
    return img


def test_async_mapping_with_hot_readers_and_reset():
    tracker = PnPTracker(CAM, loop_closure=True, async_mapping=True,
                         config={"tracker": {"lost_reset_after": 10,
                                             "min_init_points": 30,
                                             "kf_max_gap": 2,   # forzar MUCHOS KFs
                                             "kf_min_gap": 1,
                                             "min_map_matches": 20,
                                             "kf_min_inliers": 25,
                                             "kf_health_inliers": 10}})
    depth = np.full((H, W), 2.0, np.float32)
    stop = threading.Event()
    reader_errors: list = []

    def reader():                       # el "nodo ROS": lee en caliente
        while not stop.is_set():
            try:
                tracker.keyframe_trajectory()
                tracker.mapper.get_map()
            except Exception as e:      # pragma: no cover
                reader_errors.append(repr(e))
            time.sleep(0.002)

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    scene = _scene(0)
    black = np.zeros((H, W), np.uint8)
    t0 = time.time()
    for phase, frames in (("track", 25), ("apagon", 14), ("recupera", 25)):
        img = black if phase == "apagon" else scene if phase == "track" \
            else _scene(1)
        for _ in range(frames):
            tracker.process_frame(img, depth)
    elapsed = time.time() - t0
    stop.set()
    th.join(5.0)

    assert elapsed < 120, f"posible deadlock: {elapsed:.0f}s para 64 frames"
    assert not reader_errors, f"lecturas en caliente rotas: {reader_errors[:2]}"
    assert tracker.map_failures == 0, \
        f"{tracker.map_failures} excepciones en el hilo de mapeo"
    assert tracker.reset_events, "el apagon debio resetear el mapa"
    assert tracker._initialized, "no se recupero tras el reset"
    assert len(tracker.keyframe_trajectory()) >= 2


def main() -> int:
    test_async_mapping_with_hot_readers_and_reset()
    print("OK: el test de estres de concurrencia (v0.9) pasa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
