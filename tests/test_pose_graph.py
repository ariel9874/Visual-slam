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
from vslam.core.lie import (hat, se3_exp, se3_log, sim3_exp, sim3_inv,
                            sim3_log, so3_exp, so3_log)


def _expm(A: np.ndarray, order: int = 30) -> np.ndarray:
    """Exponencial de matrices por serie con scaling-and-squaring: la 'verdad'
    numérica contra la que se validan las fórmulas cerradas de Lie."""
    n = max(0, int(np.ceil(np.log2(max(np.linalg.norm(A), 1e-9)))) + 2)
    A2 = A / (2.0 ** n)
    X = np.eye(A.shape[0])
    term = np.eye(A.shape[0])
    for k in range(1, order):
        term = term @ A2 / k
        X = X + term
    for _ in range(n):
        X = X @ X
    return X


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


def test_sim3_exp_log_roundtrip_and_matches_series():
    """sim3_exp debe coincidir con la exponencial de matrices del elemento de
    álgebra ξ^ = [[λI + [ω]ₓ, ρ], [0, 0]] (valida los coeficientes A, B, C de
    W en todas sus ramas de Taylor), y Log debe invertir Exp exactamente."""
    rng = np.random.default_rng(5)
    cases = [rng.normal(0, 0.8, 7) for _ in range(15)]
    cases += [np.zeros(7),
              np.r_[rng.normal(0, 1, 3), np.zeros(3), 0.4],     # θ≈0, λ grande
              np.r_[rng.normal(0, 1, 3), rng.normal(0, 1, 3), 1e-12],  # λ≈0
              np.r_[0.5, -0.2, 0.1, 1e-12, 0, 0, -0.6]]         # θ≈0 y λ<0
    for xi in cases:
        S = sim3_exp(xi)
        xi_hat = np.zeros((4, 4))
        xi_hat[:3, :3] = xi[6] * np.eye(3) + hat(xi[3:6])
        xi_hat[:3, 3] = xi[:3]
        assert np.allclose(S, _expm(xi_hat), atol=1e-9), xi
        assert np.allclose(sim3_log(S), xi, atol=1e-7), xi
        assert np.allclose(sim3_inv(S) @ S, np.eye(4), atol=1e-9), xi


def test_sim3_graph_fixes_scale_drift_where_se3_cannot():
    """El experimento de Strasdat (RSS 2010), reproducido en 40 líneas: la
    odometría monocular encoge un 1% POR PASO (deriva de escala). Un grafo
    SE(3) no tiene dónde meter esa inconsistencia; el Sim(3) tiene el 7º
    grado de libertad exactamente para ella."""
    n, radius, lam_bias = 24, 5.0, -0.01
    gt = _circle_poses(n, radius)

    # Cadena Sim(3) con escala derivante: cada medida relativa lleva s=e^λb.
    sim_meas, chain = [], [np.eye(4)]
    for k in range(n - 1):
        S_rel = invert_se3(gt[k]) @ gt[k + 1]
        S_rel = S_rel.copy()
        S_rel[:3, :3] *= np.exp(lam_bias)
        sim_meas.append(S_rel)
        chain.append(chain[-1] @ S_rel)
    chain = [gt[0] @ S for S in chain]           # arranca en la pose real

    def se3_part(S):
        s = np.linalg.det(S[:3, :3]) ** (1 / 3)
        T = np.eye(4)
        T[:3, :3] = S[:3, :3] / s
        T[:3, 3] = S[:3, 3]
        return T

    def ate(poses):
        d = [np.linalg.norm(poses[k][:3, 3] - gt[k][:3, 3]) for k in range(n)]
        return float(np.sqrt(np.mean(np.square(d))))

    T_loop = invert_se3(gt[-1]) @ gt[0]          # el bucle, limpio y sin escala

    # ── grafo SE(3): la odometría que registraría un sistema ciego a escala.
    g_se3 = GaussNewtonPoseGraph("se3")
    se3_chain = [se3_part(S) for S in chain]
    g_se3.add_pose(0, se3_chain[0], fixed=True)
    for k in range(1, n):
        g_se3.add_pose(k, se3_chain[k])
    for k in range(n - 1):
        g_se3.add_odometry_factor(k, k + 1,
                                  invert_se3(se3_chain[k]) @ se3_chain[k + 1],
                                  np.eye(6) * 1e2)
    g_se3.add_loop_factor(n - 1, 0, T_loop, np.eye(6) * 1e4)
    ate_se3 = ate(g_se3.optimize(iterations=30))

    # ── grafo Sim(3): mismos datos, un grado de libertad más por nodo.
    g_sim = GaussNewtonPoseGraph("sim3")
    g_sim.add_pose(0, chain[0], fixed=True)
    for k in range(1, n):
        g_sim.add_pose(k, chain[k])
    for k in range(n - 1):
        g_sim.add_odometry_factor(k, k + 1, sim_meas[k], np.eye(7) * 1e2)
    loop_sim = np.eye(4)
    loop_sim[:3, :4] = T_loop[:3, :4]            # medida rígida: s = 1
    g_sim.add_loop_factor(n - 1, 0, loop_sim, np.eye(7) * 1e4)
    result = g_sim.optimize(iterations=30)
    ate_sim3 = ate({k: se3_part(result[k]) for k in result})

    ate_drift = ate({k: se3_chain[k] for k in range(n)})
    assert ate_sim3 < 0.5 * ate_se3, \
        f"deriva {ate_drift:.3f} | se3 {ate_se3:.3f} | sim3 {ate_sim3:.3f}"
    # Y la escala del último nodo debe volver a ~1 (la deriva era e^(-0.23)≈0.79).
    s_last = np.linalg.det(result[n - 1][:3, :3]) ** (1 / 3)
    assert abs(s_last - 1.0) < 0.08, f"escala final: {s_last:.3f}"


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
    test_sim3_exp_log_roundtrip_and_matches_series()
    test_chain_with_exact_measurements_recovers_poses()
    test_loop_closure_removes_drift()
    test_sim3_graph_fixes_scale_drift_where_se3_cannot()
    print("OK: los 6 tests del backend (Lie SE3/Sim3 + grafos de poses) pasan.")
