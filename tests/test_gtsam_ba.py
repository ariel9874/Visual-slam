#!/usr/bin/env python3
"""Test de EQUIVALENCIA: el BA de GTSAM ≡ la referencia NumPy (v0.5).

La regla de v0.5 (docs/04): la ruta de rendimiento (aquí GTSAM) resuelve el
MISMO problema que la referencia y pasa los mismos tests. Con observaciones
exactas el mínimo global es la verdad, así que ambos backends deben (a)
recuperarla y (b) coincidir entre sí. Se salta limpio si no hay gtsam.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.backend.bundle_adjustment import local_bundle_adjustment
from tests.test_bundle_adjustment import _scene   # reutiliza el mismo escenario


def _has_gtsam() -> bool:
    try:
        import gtsam  # noqa: F401
        return True
    except ImportError:
        return False


def test_gtsam_ba_matches_numpy_reference():
    from vslam.backend.gtsam_ba import gtsam_bundle_adjustment
    rng = np.random.default_rng(1)
    camera, gt_poses, gt_points, obs, X = _scene()

    noisy_poses = {k: T.copy() for k, T in gt_poses.items()}
    for k in noisy_poses:
        if k not in (0, 1):
            noisy_poses[k][:3, 3] += rng.normal(0, 0.03, 3)
    noisy_points = {p: x + rng.normal(0, 0.05, 3) for p, x in gt_points.items()}

    np_poses, np_points = local_bundle_adjustment(
        camera, noisy_poses, noisy_points, obs, fixed_kfs={0, 1}, iterations=15)
    gt_poses_opt, gt_points_opt = gtsam_bundle_adjustment(
        camera, noisy_poses, noisy_points, obs, fixed_kfs={0, 1}, iterations=15)

    # (a) GTSAM recupera la verdad (mismas cotas que el test de la referencia).
    for k, T in gt_poses.items():
        assert np.linalg.norm(gt_poses_opt[k][:3, 3] - T[:3, 3]) < 1e-4, f"pose {k}"
    err_gt = max(np.linalg.norm(gt_points_opt[p] - X[p]) for p in gt_points)
    assert err_gt < 1e-3, f"peor punto GTSAM vs verdad: {err_gt:.5f}"

    # (b) GTSAM ≡ referencia NumPy (convergen al mismo mínimo).
    dp = max(np.linalg.norm(gt_poses_opt[k][:3, 3] - np_poses[k][:3, 3])
             for k in gt_poses)
    dx = max(np.linalg.norm(gt_points_opt[p] - np_points[p]) for p in gt_points)
    assert dp < 1e-3 and dx < 2e-3, f"GTSAM vs NumPy: poses {dp:.5f} puntos {dx:.5f}"


def main() -> int:
    if not _has_gtsam():
        print("SKIP: gtsam no instalado (Windows sin conda-forge / sin [gtsam]).")
        return 0
    test_gtsam_ba_matches_numpy_reference()
    print("OK: el BA de GTSAM equivale a la referencia NumPy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
