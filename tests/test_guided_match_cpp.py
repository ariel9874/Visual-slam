#!/usr/bin/env python3
"""Test de EQUIVALENCIA Python ↔ C++ del matching guiado (v0.5).

La regla de v0.5 (docs/04, regla 3): la gemela C++ (cpp/src/guided_match.cpp,
módulo `vslam_cpp`) resuelve EXACTAMENTE el mismo problema que la referencia
Python (PnPTracker._guided_match) — mismos pares, mismas distancias, mismo
orden. Se compara par a par sobre escenas sintéticas con casos límite:
puntos detrás de la cámara, fuera de imagen, empates de Hamming (donde manda
la semántica de np.argmin: gana el índice menor) y ambos tipos de descriptor
(uint8/Hamming y float32/L2). Se salta limpio si el módulo no está compilado.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera
from vslam.frontend.tracker import PnPTracker


def _has_cpp() -> bool:
    try:
        import vslam_cpp  # noqa: F401
        return True
    except ImportError:
        return False


def _scene(dtype, seed):
    """Mapa + keypoints sintéticos con solape parcial y casos límite."""
    rng = np.random.default_rng(seed)
    cam = PinholeCamera(fx=450.0, fy=450.0, cx=320.0, cy=240.0,
                        width=640, height=480)
    M, D = 300, 32 if dtype == np.uint8 else 64

    # Puntos: la mayoría delante y visibles; algunos detrás / fuera de imagen.
    pts = np.column_stack([rng.uniform(-3, 3, M), rng.uniform(-2, 2, M),
                           rng.uniform(2.0, 8.0, M)])
    pts[:20, 2] = -rng.uniform(1, 5, 20)            # detrás de la cámara
    pts[20:40, 0] = rng.uniform(20, 40, 20)         # fuera del FOV

    if dtype == np.uint8:
        map_desc = rng.integers(0, 256, (M, D), dtype=np.uint8)
    else:
        map_desc = rng.normal(0, 1, (M, D)).astype(np.float32)

    # Keypoints: proyección (pose verdad = identidad) + ruido de pocos px, con
    # el descriptor del punto LIGERAMENTE corrompido; más 400 kps de distracción.
    T_true = np.eye(4)
    z = pts[:, 2]
    uv = np.column_stack([450.0 * pts[:, 0] / z + 320.0,
                          450.0 * pts[:, 1] / z + 240.0])
    kp_xy, kp_desc = [], []
    for i in range(M):
        if z[i] <= 0 or not (0 <= uv[i, 0] < 640 and 0 <= uv[i, 1] < 480):
            continue
        kp_xy.append(uv[i] + rng.normal(0, 3.0, 2))
        if dtype == np.uint8:
            d = map_desc[i].copy()
            flip = rng.integers(0, D, 3)
            d[flip] ^= (1 << rng.integers(0, 8, 3)).astype(np.uint8)
            kp_desc.append(d)
        else:
            kp_desc.append((map_desc[i] + rng.normal(0, 0.02, D)).astype(np.float32))
    for _ in range(400):
        kp_xy.append(np.array([rng.uniform(0, 640), rng.uniform(0, 480)]))
        if dtype == np.uint8:
            kp_desc.append(rng.integers(0, 256, D, dtype=np.uint8))
        else:
            kp_desc.append(rng.normal(0, 1, D).astype(np.float32))
    order = rng.permutation(len(kp_xy))              # desordenar índices
    kp_xy = [kp_xy[i] for i in order]
    kp_desc = np.stack([kp_desc[i] for i in order])
    kps = [cv2.KeyPoint(float(x), float(y), 8.0) for x, y in kp_xy]

    # Pose predicha: la verdad con una perturbación pequeña (prior imperfecto).
    T_pred = np.eye(4)
    T_pred[:3, 3] = rng.normal(0, 0.01, 3)
    return cam, kps, kp_desc, T_pred, pts, map_desc


def _compare(dtype, seed):
    cam, kps, desc, T_pred, map_pts, map_desc = _scene(dtype, seed)
    tr = PnPTracker(cam)

    tr.use_cpp = False
    ref = tr._guided_match(kps, desc, T_pred, map_pts, map_desc)
    tr.use_cpp = True
    fast = tr._guided_match(kps, desc, T_pred, map_pts, map_desc)

    assert len(ref) > 50, f"escena degenerada: solo {len(ref)} matches"
    assert len(ref) == len(fast), f"{len(ref)} vs {len(fast)} matches"
    for a, b in zip(ref, fast):
        assert a.queryIdx == b.queryIdx and a.trainIdx == b.trainIdx, \
            f"par distinto: ({a.queryIdx},{a.trainIdx}) vs ({b.queryIdx},{b.trainIdx})"
        assert abs(a.distance - b.distance) < 1e-6, \
            f"distancia: {a.distance} vs {b.distance}"


def test_hamming_equivalence():
    for seed in (0, 1, 2):
        _compare(np.uint8, seed)


def test_l2_equivalence():
    for seed in (3, 4):
        _compare(np.float32, seed)


def main() -> int:
    if not _has_cpp():
        print("SKIP: vslam_cpp no compilado (ver cpp/CMakeLists.txt).")
        return 0
    test_hamming_equivalence()
    test_l2_equivalence()
    print("OK: el matching guiado C++ equivale a la referencia Python (5 escenas).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
