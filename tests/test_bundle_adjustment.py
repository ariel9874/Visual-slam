"""Tests del bundle adjustment local (v0.35).

Ejecutar:  pytest tests/  (o directamente: python tests/test_bundle_adjustment.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.backend.bundle_adjustment import local_bundle_adjustment
from vslam.core.camera import PinholeCamera
from vslam.core.geometry import invert_se3


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _scene(n_pts=120, n_cams=4, seed=0):
    """Cámaras en línea con leve guiñada mirando una nube 3D; obs exactas."""
    rng = np.random.default_rng(seed)
    camera = PinholeCamera(fx=450.0, fy=450.0, cx=320.0, cy=240.0)
    X = np.column_stack([rng.uniform(-4, 4, n_pts),
                         rng.uniform(-2.5, 2.5, n_pts),
                         rng.uniform(5, 11, n_pts)])
    poses = {}
    for k in range(n_cams):
        T = np.eye(4)
        T[:3, :3] = _rot_y(0.02 * k)
        T[:3, 3] = [0.3 * k, 0.0, 0.05 * k]
        poses[k] = T
    obs = []
    for k, T in poses.items():
        T_c_w = invert_se3(T)
        pc = (T_c_w[:3, :3] @ X.T).T + T_c_w[:3, 3]
        uv = np.column_stack([camera.fx * pc[:, 0] / pc[:, 2] + camera.cx,
                              camera.fy * pc[:, 1] / pc[:, 2] + camera.cy])
        obs += [(k, p, uv[p]) for p in range(n_pts)]
    points = {p: X[p] for p in range(n_pts)}
    return camera, poses, points, obs, X


def test_ba_recovers_ground_truth_from_noisy_guess():
    """Con observaciones EXACTAS, el mínimo global es la verdad: partiendo de
    poses y puntos perturbados, el BA debe volver a ella (validando de paso
    los signos de los jacobianos analíticos).

    Se fijan DOS cámaras: con una sola, la escala monocular queda libre y la
    solución converge a un múltiplo de la verdad (gauge de 7 gdl — ver el
    docstring de bundle_adjustment; lo aprendimos depurando este test)."""
    rng = np.random.default_rng(1)
    camera, gt_poses, gt_points, obs, X = _scene()

    noisy_poses = {k: T.copy() for k, T in gt_poses.items()}
    for k in noisy_poses:
        if k not in (0, 1):                       # KFs 0 y 1 anclan el gauge
            noisy_poses[k][:3, 3] += rng.normal(0, 0.03, 3)
    noisy_points = {p: x + rng.normal(0, 0.05, 3) for p, x in gt_points.items()}

    opt_poses, opt_points = local_bundle_adjustment(
        camera, noisy_poses, noisy_points, obs, fixed_kfs={0, 1}, iterations=15)

    for k, T in gt_poses.items():
        assert np.linalg.norm(opt_poses[k][:3, 3] - T[:3, 3]) < 1e-4, k
    err = np.array([np.linalg.norm(opt_points[p] - X[p]) for p in gt_points])
    assert err.max() < 1e-3, f"peor punto: {err.max():.5f}"


def test_ba_huber_resists_outlier_observations():
    """Un 10% de observaciones corruptas no debe arrastrar la solución
    (el kernel de Huber las degrada a empuje lineal).

    Dos lecciones de diseño aprendidas depurando este test:
    · La corrupción debe ser de dirección ALEATORIA: con sesgo consistente
      (todos los outliers hacia +x) el mínimo robusto legítimamente se
      desplaza para absorberlo — Huber atenúa outliers aleatorios, no errores
      sistemáticos (eso es tarea de la calibración).
    · Con pocas observaciones por punto no hay milagros: un punto con 2
      outliers de 4 obs queda en un empate lineal 2-vs-2 (valle plano del
      kernel) y puede alejarse sin costo (por eso aquí hay 6 cámaras).
    · Huber NO anula los outliers: los degrada de empuje cuadrático a empuje
      LINEAL. Queda un sesgo residual proporcional a la tasa de outliers
      (medido: ~0.2% de la escala de la escena con 10% de corrupción). La
      afirmación correcta no es "error cero", sino "mucho mejor que el
      kernel cuadrático" — que es exactamente lo que se comprueba."""
    rng = np.random.default_rng(2)
    camera, gt_poses, gt_points, obs, X = _scene(n_cams=6)
    corrupted_pts = set()
    obs_c = []
    for k, p, uv in obs:
        if rng.random() < 0.1:
            uv = uv + rng.uniform(30, 80, 2) * rng.choice([-1.0, 1.0], 2)
            corrupted_pts.add(p)
        obs_c.append((k, p, uv))

    noisy_points = {p: x + rng.normal(0, 0.03, 3) for p, x in gt_points.items()}
    clean = [p for p in gt_points if p not in corrupted_pts]
    assert len(clean) > 30

    def run(huber):
        _, opt_points = local_bundle_adjustment(
            camera, gt_poses, noisy_points, obs_c, fixed_kfs={0, 1},
            iterations=20, huber_px=huber)
        return float(np.median([np.linalg.norm(opt_points[p] - X[p])
                                for p in clean]))

    err_huber = run(2.5)
    err_quad = run(1e6)              # kernel efectivamente cuadrático
    # Cota de sanidad absoluta + la protección relativa que Huber promete.
    assert err_huber < 0.03, f"mediana limpios con Huber: {err_huber:.5f}"
    assert err_huber < 0.5 * err_quad, \
        f"Huber {err_huber:.5f} vs cuadrático {err_quad:.5f}: no protege"


if __name__ == "__main__":
    test_ba_recovers_ground_truth_from_noisy_guess()
    test_ba_huber_resists_outlier_observations()
    print("OK: los 2 tests de bundle adjustment pasan.")
