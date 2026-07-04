"""Verifica las convenciones geométricas del repo con datos sintéticos exactos.

Estos tests son el "contrato ejecutable" de la convención de poses
(docs/02_arquitectura.md §4). Si alguien duda de en qué sentido va R o t
—la fuente clásica de bugs en SLAM— la respuesta está aquí, comprobada.

Ejecutar:  pytest tests/  (o directamente: python tests/test_pose_recovery.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera
from vslam.core.trajectory import rotation_to_quaternion


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _synthetic_two_views():
    """Dos vistas de una nube 3D con pose relativa conocida y exacta."""
    rng = np.random.default_rng(42)
    camera = PinholeCamera(fx=450.0, fy=450.0, cx=320.0, cy=240.0)
    points_w = np.column_stack([
        rng.uniform(-4, 4, 300),
        rng.uniform(-2.5, 2.5, 300),
        rng.uniform(4, 10, 300),
    ])
    # Cámara 1 en el origen (frame mundo = frame cámara 1).
    # Cámara 2: trasladada y con un giro pequeño (ground truth).
    R_w_c2 = _rot_y(np.deg2rad(2.0))
    C2 = np.array([0.3, 0.05, 0.1])

    x1 = camera.project(points_w)                       # vista 1
    pts_c2 = (R_w_c2.T @ (points_w - C2).T).T
    x2 = camera.project(pts_c2)                         # vista 2
    return camera, x1, x2, R_w_c2, C2


def test_recover_pose_convention():
    """cv2.recoverPose devuelve T_c2<-c1: x_c2 = R·x_c1 + t (con ||t||=1).

    Este es EXACTAMENTE el supuesto sobre el que examples/01_monocular_vo.py
    compone la trayectoria; si OpenCV cambiara la convención, este test avisa.
    """
    camera, x1, x2, R_w_c2, C2 = _synthetic_two_views()

    E, mask = cv2.findEssentialMat(x1, x2, camera.K, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    assert E is not None and E.shape == (3, 3)
    n_inliers, R, t, _ = cv2.recoverPose(E, x1, x2, camera.K, mask=mask)
    assert n_inliers > 200, f"pocos inliers: {n_inliers}"

    # Ground truth de T_c2<-c1: R_gt = R_w_c2^T, t_gt = -R_w_c2^T · C2.
    R_gt = R_w_c2.T
    t_gt = -R_w_c2.T @ C2
    t_gt_dir = t_gt / np.linalg.norm(t_gt)

    # Rotación: distancia geodésica en SO(3). R·R_gtᵀ es "lo que falta por
    # girar" entre estimación y verdad; su ángulo de eje-ángulo sale de la
    # identidad tr(R_err) = 1 + 2·cos(θ):  θ = arccos((tr(R·R_gtᵀ) − 1)/2).
    cos_angle = (np.trace(R @ R_gt.T) - 1.0) / 2.0
    angle_deg = np.rad2deg(np.arccos(np.clip(cos_angle, -1, 1)))
    assert angle_deg < 0.5, f"error de rotación {angle_deg:.3f} grados"

    # Traslación: solo la DIRECCIÓN es observable (escala monocular).
    dot = float(t.ravel() @ t_gt_dir)
    assert dot > 0.999, f"dirección de traslación incorrecta (dot={dot:.4f})"


def test_recover_pose_small_baseline_needs_distance_thresh():
    """Regresión de una trampa real de OpenCV (documentada en examples/01).

    La sobrecarga básica de cv2.recoverPose limita la quiralidad a 50 unidades
    de profundidad — que con ||t||=1 son MÚLTIPLOS DEL BASELINE. Con movimiento
    pequeño frente a la profundidad de la escena (depth/baseline > 50, p. ej.
    KITTI frame a frame), los inliers colapsan a ~0 aunque la geometría sea
    perfecta. La sobrecarga con distanceThresh lo corrige.
    """
    rng = np.random.default_rng(3)
    camera = PinholeCamera(fx=450.0, fy=450.0, cx=320.0, cy=240.0)
    points_w = np.column_stack([
        rng.uniform(-4, 4, 300),
        rng.uniform(-2.5, 2.5, 300),
        rng.uniform(5, 12, 300),          # profundidad media ~8.5
    ])
    C2 = np.array([0.05, 0.0, 0.01])      # baseline 0.051 -> depth/baseline ~170
    x1 = camera.project(points_w)
    x2 = camera.project(points_w - C2)    # rotación identidad, solo traslación

    E, mask = cv2.findEssentialMat(x1, x2, camera.K, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    # Con el umbral corregido, la gran mayoría de puntos son inliers válidos:
    n_ok, _, t, _, _ = cv2.recoverPose(E, x1, x2, camera.K,
                                       distanceThresh=2000.0, mask=mask.copy())
    assert n_ok > 200, f"con distanceThresh deberían sobrevivir casi todos: {n_ok}"
    t_gt_dir = -C2 / np.linalg.norm(C2)   # T_c2<-c1: t = -R^T·C2 con R = I
    assert float(t.ravel() @ t_gt_dir) > 0.99


def test_quaternion_roundtrip():
    """rotation_to_quaternion produce un cuaternión unitario consistente."""
    R = _rot_y(np.deg2rad(37.0))
    q = rotation_to_quaternion(R)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9
    # Reconstrucción estándar (qx,qy,qz,qw) -> R y comparación.
    x, y, z, w = q
    R_back = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    assert np.allclose(R, R_back, atol=1e-9)


if __name__ == "__main__":
    test_recover_pose_convention()
    test_recover_pose_small_baseline_needs_distance_thresh()
    test_quaternion_roundtrip()
    print("OK: los 3 tests de convenciones geométricas pasan.")
