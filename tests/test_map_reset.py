#!/usr/bin/env python3
"""Test de la degradación elegante (v0.9 hito 2): reset de mapa tras pérdida
irrecuperable.

Escenario del "apagón": el tracker inicializa RGB-D sobre una escena texturada,
la cámara "se tapa" (frames negros: cero features, la reloc no puede enganchar)
durante más de LOST_RESET_AFTER frames, y al destaparse debe:
  (1) haber ARCHIVADO la sesión muerta (reset_events + trayectoria preservada),
  (2) RE-INICIALIZAR una sesión nueva sin crash (initialized + métrico),
  (3) reportar la trayectoria COMPLETA (archivada + nueva) en
      keyframe_trajectory().
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera
from vslam.frontend.tracker import PnPTracker

W, H = 320, 240
CAM = PinholeCamera(fx=260.0, fy=260.0, cx=W / 2, cy=H / 2, width=W, height=H)


def _textured(seed: int = 0) -> np.ndarray:
    """Escena con esquinas de sobra para ORB (parches binarios aleatorios)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), np.uint8)
    for _ in range(220):
        u, v = rng.integers(6, W - 6), rng.integers(6, H - 6)
        img[v - 3:v + 3, u - 3:u + 3] = rng.integers(80, 255)
    return img


def test_blackout_reset_and_recovery():
    tracker = PnPTracker(CAM, loop_closure=False,
                         config={"tracker": {"lost_reset_after": 12,
                                             "min_init_points": 30,
                                             "min_map_matches": 20,
                                             "kf_min_inliers": 25,
                                             "kf_health_inliers": 10}})
    scene = _textured(0)
    depth = np.full((H, W), 2.0, np.float32)     # plano a 2 m (métrico)

    # 1) Sesión 1: init + unos frames estáticos.
    for _ in range(6):
        tracker.process_frame(scene, depth)
    assert tracker._initialized and tracker._metric
    kfs_before = len(tracker._kf_ids)
    assert kfs_before >= 1

    # 2) APAGÓN: negros hasta cruzar LOST_RESET_AFTER.
    black = np.zeros((H, W), np.uint8)
    states = []
    for _ in range(20):
        _, info = tracker.process_frame(black, depth)
        states.append(info["state"])
    assert tracker.reset_events, "nunca reseteo el mapa tras el apagon"
    assert "RESET" in states, f"sin estado RESET: {set(states)}"
    assert len(tracker.reset_events) == 1, "debe resetear UNA vez por apagon"
    assert not tracker._initialized, "tras el reset queda en modo init"

    # 3) Se destapa: re-init de la sesión 2, sin crash y métrico otra vez.
    scene2 = _textured(1)
    for _ in range(6):
        tracker.process_frame(scene2, depth)
    assert tracker._initialized and tracker._metric, "no re-inicializo"

    # 4) La trayectoria completa incluye la sesión archivada + la nueva.
    traj = tracker.keyframe_trajectory()
    assert len(traj) >= kfs_before + 1, "perdio la trayectoria archivada"
    ids = [k for k, _ in traj]
    assert ids == sorted(ids), "las sesiones deben ir en orden temporal"


def main() -> int:
    test_blackout_reset_and_recovery()
    print("OK: el test de degradacion elegante (reset de mapa, v0.9) pasa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
