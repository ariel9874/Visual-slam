"""Tests del backend v0.3: álgebra de Lie SE(3) y grafo de poses.

Ejecutar:  pytest tests/  (o directamente: python tests/test_pose_graph.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.backend.pose_graph import GaussNewtonPoseGraph
from vslam.core.geometry import invert_se3
from vslam.core.lie import se3_exp, se3_log, so3_exp, so3_log


def test_lie_exp_log_roundtrip():
    """Log(Exp(ξ)) = ξ en todos los regímenes numéricos (0, normal, cerca de π)."""
    rng = np.random.default_rng(0)
    cases = [rng.normal(0, 1, 6) for _ in range(20)]
    cases += [np.zeros(6), np.full(6, 1e-10)]                       # θ ≈ 0
    cases += [np.concatenate([rng.normal(0, 1, 3),                  # θ ≈ π
                              3.1415 * np.array([0.0, 1.0, 0.0])])]
    for xi in cases:
        assert np.allclose(se3_log(se3_exp(xi)), xi, atol=1e-6), xi


def test_so3_log_exact_pi():
    """El caso θ = π exacto (traza = −1) no debe explotar ni perder el eje."""
    k = np.array([1.0, 2.0, -0.5])
    k /= np.linalg.norm(k)
    R = so3_exp(np.pi * k)
    w = so3_log(R)
    # El signo del eje es ambiguo en π: comparar la ROTACIÓN, no el vector.
    assert np.allclose(so3_exp(w), R, atol=1e-6)
    assert abs(np.linalg.norm(w) - np.pi) < 1e-6


def _circle_poses(n: int, radius: float) -> list:
    """Trayectoria circular en el plano x-z con guiñada tangente (SE(3))."""
    poses = []
    for k in range(n):
        a = 2 * np.pi * k / n
        c, s = np.cos(a), np.sin(a)
        T = np.eye(4)
        T[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        T[:3, 3] = [radius * np.sin(a), 0.0, radius * (1 - np.cos(a))]
        poses.append(T)
    return poses


def test_chain_with_exact_measurements_recovers_poses():
    """Con medidas EXACTAS y guess ruidoso, el óptimo es la verdad (costo 0)."""
    rng = np.random.default_rng(1)
    gt = _circle_poses(8, radius=3.0)
    graph = GaussNewtonPoseGraph()
    info = np.eye(6) * 100.0
    for k, T in enumerate(gt):
        noise = se3_exp(rng.normal(0, 0.05, 6)) if k else np.eye(4)
        graph.add_pose(k, T @ noise, fixed=(k == 0))
    for k in range(len(gt) - 1):
        graph.add_odometry_factor(k, k + 1, invert_se3(gt[k]) @ gt[k + 1], info)

    result = graph.optimize()
    for k, T in enumerate(gt):
        assert np.linalg.norm(result[k][:3, 3] - T[:3, 3]) < 1e-5, k


def test_loop_closure_removes_drift():
    """El escenario canónico: odometría con sesgo acumula deriva; UN factor de
    bucle la redistribuye por toda la cadena (docs de factor_graph.py)."""
    rng = np.random.default_rng(2)
    gt = _circle_poses(24, radius=5.0)
    bias = np.array([0.0, 0.0, 0.0, 0.0, 0.006, 0.0])   # guiñada extra por paso

    graph = GaussNewtonPoseGraph()
    odo_info = np.eye(6) * 1e2
    loop_info = np.eye(6) * 1e4
    drift = [gt[0]]
    graph.add_pose(0, gt[0], fixed=True)
    for k in range(len(gt) - 1):
        T_rel = invert_se3(gt[k]) @ gt[k + 1]
        T_meas = T_rel @ se3_exp(bias + rng.normal(0, 0.002, 6))
        drift.append(drift[-1] @ T_meas)                # guess = odometría cruda
        graph.add_pose(k + 1, drift[-1])
        graph.add_odometry_factor(k, k + 1, T_meas, odo_info)
    # Cierre de bucle: el último frame re-observa el primero (medida limpia).
    graph.add_loop_factor(len(gt) - 1, 0, invert_se3(gt[-1]) @ gt[0], loop_info)

    result = graph.optimize(iterations=30)
    err_before = np.array([np.linalg.norm(drift[k][:3, 3] - gt[k][:3, 3])
                           for k in range(len(gt))])
    err_after = np.array([np.linalg.norm(result[k][:3, 3] - gt[k][:3, 3])
                          for k in range(len(gt))])
    rmse_b = np.sqrt((err_before ** 2).mean())
    rmse_a = np.sqrt((err_after ** 2).mean())
    assert rmse_a < 0.25 * rmse_b, f"antes {rmse_b:.3f} después {rmse_a:.3f}"


if __name__ == "__main__":
    test_lie_exp_log_roundtrip()
    test_so3_log_exact_pi()
    test_chain_with_exact_measurements_recovers_poses()
    test_loop_closure_removes_drift()
    print("OK: los 4 tests del backend (Lie + grafo de poses) pasan.")
