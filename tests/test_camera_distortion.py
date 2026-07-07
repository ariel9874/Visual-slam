#!/usr/bin/env python3
"""Tests de la distorsión de lente (v0.45): PinholeCamera.undistort_points.

Verifica el puente entre un dataset crudo y la geometría ideal del repo:
(1) con distorsión nula es la identidad (el sintético no se entera);
(2) con la distorsión real de TUM fr1, des-distorsionar recupera el píxel
    ideal — usamos cv2.projectPoints (que SÍ aplica la distorsión directa) para
    generar el píxel distorsionado y comprobamos que undistort_points lo revierte.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera

# Intrínsecos y distorsión de la cámara Freiburg 1 (TUM RGB-D).
FR1 = dict(fx=517.306408, fy=516.469215, cx=318.643040, cy=255.313989,
           width=640, height=480)
FR1_DIST = (0.262383, -0.953104, -0.005358, 0.002628, 1.163314)


def _random_points(n=300, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.45, 0.45, size=(n, 2))     # normalizadas (dentro del FOV)
    z = rng.uniform(1.0, 6.0, size=(n, 1))
    return np.hstack([xy * z, z])                  # (N, 3) en frame de cámara


def test_identity_without_distortion():
    cam = PinholeCamera(**FR1)                      # dist = 0 por defecto
    assert not cam.has_distortion
    px = _random_points()[:, :2] * 100 + 200        # píxeles cualquiera
    out = cam.undistort_points(px)
    assert np.allclose(out, px), "sin distorsión debe ser la identidad"


def test_roundtrip_recovers_ideal_pixels():
    cam = PinholeCamera(**FR1, distortion=FR1_DIST)
    assert cam.has_distortion
    pts3d = _random_points()

    ideal = cam.project(pts3d)                      # proyección SIN distorsión
    distorted, _ = cv2.projectPoints(               # proyección CON distorsión
        pts3d, np.zeros(3), np.zeros(3), cam.K, cam.dist)
    distorted = distorted.reshape(-1, 2)

    # La distorsión debe MOVER los píxeles (si no, el test no probaría nada).
    assert np.linalg.norm(distorted - ideal, axis=1).max() > 1.0

    recovered = cam.undistort_points(distorted)
    err = np.linalg.norm(recovered - ideal, axis=1)
    assert err.max() < 0.2, f"undistort no recupera el ideal: max {err.max():.3f}px"


def test_from_file_parses_distortion(tmp_path=None):
    import tempfile
    d = Path(tempfile.mkdtemp())
    calib = d / "calib.txt"
    calib.write_text("# fx fy cx cy w h k1 k2 p1 p2 k3\n"
                     "517.31 516.47 318.64 255.31 640 480 "
                     "0.2624 -0.9531 -0.0054 0.0026 1.1633\n", encoding="utf-8")
    cam = PinholeCamera.from_file(calib)
    assert cam.has_distortion and abs(cam.dist[0] - 0.2624) < 1e-6
    assert cam.width == 640 and cam.height == 480


def main() -> int:
    test_identity_without_distortion()
    test_roundtrip_recovers_ideal_pixels()
    test_from_file_parses_distortion()
    print("OK: los 3 tests de distorsion de lente pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
