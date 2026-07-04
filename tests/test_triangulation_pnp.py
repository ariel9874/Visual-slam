"""Tests de la geometría de v0.2: triangulación DLT, PnP y re-anclaje del mapa.

Todo con geometría sintética exacta: si estas piezas fallan, el PnPTracker no
tiene sobre qué sostenerse. Ejecutar: pytest tests/ (o python este archivo).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera
from vslam.core.frame import Frame
from vslam.core.geometry import invert_se3, solve_pnp, triangulate_two_views
from vslam.mapping.sparse import SparsePointMapper


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _scene(n: int = 200, seed: int = 11):
    rng = np.random.default_rng(seed)
    camera = PinholeCamera(fx=450.0, fy=450.0, cx=320.0, cy=240.0)
    points_w = np.column_stack([
        rng.uniform(-4, 4, n), rng.uniform(-2.5, 2.5, n), rng.uniform(4, 10, n),
    ])
    return camera, points_w


def _pose(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = C
    return T


def _project(camera, points_w, T_w_c):
    T_c_w = invert_se3(T_w_c)
    return camera.project((T_c_w[:3, :3] @ points_w.T).T + T_c_w[:3, 3])


def test_triangulation_recovers_3d():
    """Con observaciones exactas y buen baseline, la DLT recupera los puntos."""
    camera, X = _scene()
    T0, T1 = np.eye(4), _pose(_rot_y(0.03), np.array([0.5, 0.0, 0.1]))
    rec, valid = triangulate_two_views(camera, T0, T1, _project(camera, X, T0),
                                       _project(camera, X, T1))
    assert valid.sum() > 190, f"válidos: {valid.sum()}"
    err = np.linalg.norm(rec[valid] - X[valid], axis=1)
    assert err.max() < 1e-6, f"error máx: {err.max()}"


def test_triangulation_rejects_low_parallax():
    """Baseline ínfimo → rayos casi paralelos → el filtro de paralaje debe
    rechazarlo todo (un punto mal condicionado es peor que ningún punto)."""
    camera, X = _scene()
    T0, T1 = np.eye(4), _pose(np.eye(3), np.array([0.001, 0.0, 0.0]))
    _, valid = triangulate_two_views(camera, T0, T1, _project(camera, X, T0),
                                     _project(camera, X, T1), min_parallax_deg=0.3)
    assert valid.sum() == 0, f"aceptó {valid.sum()} puntos sin paralaje"


def test_pnp_recovers_pose_with_noise():
    """PnP debe recuperar la pose con ruido de píxel realista (σ = 0.5 px)."""
    rng = np.random.default_rng(3)
    camera, X = _scene()
    T_gt = _pose(_rot_y(np.deg2rad(5.0)), np.array([0.4, -0.1, 0.2]))
    pixels = _project(camera, X, T_gt) + rng.normal(0, 0.5, (len(X), 2))

    T_est, inliers = solve_pnp(camera, X, pixels)
    assert T_est is not None and inliers.sum() > 150
    # Posición a milímetros (unidades de escena) y rotación a décimas de grado.
    assert np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]) < 0.01
    cos = (np.trace(T_est[:3, :3] @ T_gt[:3, :3].T) - 1) / 2
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))) < 0.2


def test_pnp_survives_outliers():
    """Un 30% de matches basura no debe torcer la pose (para eso está RANSAC)."""
    rng = np.random.default_rng(4)
    camera, X = _scene()
    T_gt = _pose(_rot_y(0.05), np.array([0.3, 0.0, 0.1]))
    pixels = _project(camera, X, T_gt)
    bad = rng.random(len(X)) < 0.3
    pixels[bad] = rng.uniform(0, 640, (bad.sum(), 2))     # matches aleatorios

    T_est, inliers = solve_pnp(camera, X, pixels)
    assert T_est is not None
    assert np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]) < 0.01
    assert inliers[bad].mean() < 0.1, "marcó outliers como inliers"


def test_mapper_reanchors_points_on_pose_update():
    """update_poses debe aplicar a los puntos el delta de su keyframe ancla
    (el mecanismo que usará el cierre de bucle en v0.3)."""
    mapper = SparsePointMapper()
    T_old = _pose(np.eye(3), np.array([1.0, 0.0, 0.0]))
    mapper.integrate_keyframe(Frame(frame_id=7, timestamp=0.0, T_w_c=T_old,
                                    is_keyframe=True))
    p = np.array([[2.0, 1.0, 5.0]])
    mapper.add_points(p, np.zeros((1, 32), np.uint8), anchor_kf_id=7)

    T_new = _pose(_rot_y(0.1), np.array([1.2, 0.0, -0.1]))
    mapper.update_poses({7: T_new})

    delta = T_new @ invert_se3(T_old)
    expected = delta[:3, :3] @ p[0] + delta[:3, 3]
    assert np.allclose(mapper.get_map()[0], expected, atol=1e-12)


if __name__ == "__main__":
    test_triangulation_recovers_3d()
    test_triangulation_rejects_low_parallax()
    test_pnp_recovers_pose_with_noise()
    test_pnp_survives_outliers()
    test_mapper_reanchors_points_on_pose_update()
    print("OK: los 5 tests de triangulación/PnP/mapa pasan.")
