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


def test_gtsam_stereo_factor_makes_scale_observable():
    """El factor estéreo de GTSAM (v0.6) hace OBSERVABLE la escala, igual que el
    residuo de profundidad de la referencia NumPy. Misma escena que el test
    RGB-D: dos cámaras con UNA sola fija (gauge monocular = espacio nulo). El BA
    2D deja el offset de 15% de escala; con el factor estéreo (u_R medido) ambos
    backends recuperan la escala métrica y el baseline de la cámara libre.
    """
    from vslam.backend.gtsam_ba import gtsam_bundle_adjustment
    from vslam.core.camera import PinholeCamera

    cam = PinholeCamera(fx=450.0, fy=450.0, cx=320.0, cy=240.0,
                        width=640, height=480)
    rng = np.random.default_rng(3)
    n = 60
    pts_true = np.column_stack([rng.uniform(-0.5, 0.5, n),
                                rng.uniform(-0.4, 0.4, n),
                                rng.uniform(1.2, 3.0, n)])
    centers = {0: np.zeros(3), 1: np.array([0.25, 0.0, 0.0])}
    bf = 40.0
    obs = []
    for k, c in centers.items():
        rel = pts_true - c
        u = 450.0 * rel[:, 0] / rel[:, 2] + 320.0
        v = 450.0 * rel[:, 1] / rel[:, 2] + 240.0
        u_r = u - bf / rel[:, 2]
        obs += [(k, p, np.array([u[p], v[p], u_r[p]])) for p in range(n)]

    poses = {k: np.eye(4) for k in centers}
    for k, c in centers.items():
        poses[k][:3, 3] = 1.15 * c        # familia de gauge: 15% de escala
    pts = {p: 1.15 * pts_true[p] for p in range(n)}

    def median_scale(opt):
        return float(np.median([np.linalg.norm(opt[p]) / np.linalg.norm(pts_true[p])
                                for p in range(n)]))

    # Sin factor estéreo (bf=0): la escala es espacio nulo → el 15% se queda.
    _, mono = gtsam_bundle_adjustment(cam, poses, pts, obs, fixed_kfs={0},
                                      iterations=20)
    assert median_scale(mono) > 1.10, "GTSAM 2D no debería corregir el gauge"

    # Con factor estéreo: escala recuperada (y coincide con la referencia NumPy).
    gp, gpts = gtsam_bundle_adjustment(cam, poses, pts, obs, fixed_kfs={0},
                                       iterations=20, stereo_bf=bf)
    _, npts = local_bundle_adjustment(cam, poses, pts, obs, fixed_kfs={0},
                                      iterations=20, stereo_bf=bf)
    assert abs(median_scale(gpts) - 1.0) < 0.02, "GTSAM estéreo no recuperó la escala"
    assert abs(median_scale(npts) - 1.0) < 0.02, "NumPy estéreo no recuperó la escala"
    assert np.allclose(gp[1][:3, 3], centers[1], atol=0.02), "baseline libre mal"


def main() -> int:
    if not _has_gtsam():
        print("SKIP: gtsam no instalado (Windows sin conda-forge / sin [gtsam]).")
        return 0
    test_gtsam_ba_matches_numpy_reference()
    test_gtsam_stereo_factor_makes_scale_observable()
    print("OK: el BA de GTSAM equivale a la referencia NumPy (incl. factor estéreo).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
