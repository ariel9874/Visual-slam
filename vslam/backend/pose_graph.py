"""Optimizador de grafo de poses en NumPy puro — la referencia legible del backend.

Implementa de verdad lo que la interfaz FactorGraphBackend promete (la teoría
MAP → mínimos cuadrados está en factor_graph.py): Gauss-Newton con
amortiguación de Levenberg-Marquardt sobre SE(3), con todo el mecanismo a la
vista. GTSAM/g2o hacen esto mismo con jacobianos analíticos y álgebra
dispersa; esta versión prioriza que se pueda LEER (y en grafos de cientos de
keyframes sigue siendo interactiva).

─── La matemática: los detalles que faltaban ─────────────────────────────────
Blanqueo (whitening). El costo Σ e_kᵀ·Λ_k·e_k se convierte en un problema de
mínimos cuadrados ORDINARIO factorizando la información Λ = Lᵀ·L (Cholesky):

    ‖e‖²_Λ = eᵀLᵀL e = ‖L·e‖²      →  residuo blanqueado r = L·e

Así el solver no necesita saber de covarianzas: optimiza ½‖r(Θ)‖².

Actualización en la variedad. Cada pose se perturba POR LA DERECHA:

    T_i ← T_i · Exp(δ_i),    δ_i ∈ ℝ⁶

y el jacobiano J = ∂r/∂δ se evalúa NUMÉRICAMENTE (diferencias finitas sobre
los 6 grados de cada pose implicada). Es la decisión didáctica clave: el
jacobiano analítico de Log(T̂⁻¹Ti⁻¹Tj) exige jacobianos adjuntos de SE(3) que
ocultan el bosque; el numérico cuesta 12 evaluaciones de residuo por arista
(gratis a esta escala) y es imposible equivocarse de convención.

Levenberg-Marquardt. Se resuelve  (H + λ·diag(H))·δ = −Jᵀr  con H = JᵀJ:
λ pequeño ⇒ paso de Gauss-Newton (rápido cerca del óptimo); λ grande ⇒ paso
corto tipo gradiente (seguro lejos). λ baja si el costo mejora y sube si no.

Gauge. Sin anclaje, el costo es invariante ante mover TODO el grafo junto
(H singular). Se fija declarando una pose `fixed=True` (típicamente la
primera); si nadie lo hace, se ancla la primera automáticamente.

Kernel robusto (Huber). Los factores de bucle pasan por IRLS: si el residuo
blanqueado excede δ_huber, su peso decae w = δ/‖r‖ — un falso positivo de
cierre de bucle "empuja" linealmente, no cuadráticamente, y no destroza el
grafo entero.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from vslam.backend.factor_graph import FactorGraphBackend
from vslam.core.geometry import invert_se3
from vslam.core.lie import se3_exp, se3_log


class GaussNewtonPoseGraph(FactorGraphBackend):
    """Backend de grafo de poses: NumPy + jacobianos numéricos + LM."""

    HUBER_DELTA = 1.0     # umbral (residuo blanqueado) del kernel robusto
    JACOBIAN_EPS = 1e-6   # paso de las diferencias finitas

    def __init__(self) -> None:
        self._poses: Dict[int, np.ndarray] = {}
        self._fixed: set = set()
        self._edges: List[dict] = []

    # ── construcción del grafo ───────────────────────────────────────────────

    def add_pose(self, node_id: int, T_w_c: np.ndarray, fixed: bool = False) -> None:
        self._poses[node_id] = np.asarray(T_w_c, dtype=float).copy()
        if fixed:
            self._fixed.add(node_id)

    def add_odometry_factor(self, id_from, id_to, T_rel, information) -> None:
        self._add_edge(id_from, id_to, T_rel, information, robust=False)

    def add_loop_factor(self, id_from, id_to, T_rel, information) -> None:
        # Los bucles llevan kernel robusto: un reconocimiento de lugar
        # equivocado no debe poder doblar todo el grafo (ver docstring).
        self._add_edge(id_from, id_to, T_rel, information, robust=True)

    def _add_edge(self, i, j, T_rel, information, robust) -> None:
        L = np.linalg.cholesky(np.asarray(information, dtype=float)).T  # Λ = LᵀL
        self._edges.append({
            "i": i, "j": j,
            "T_meas_inv": invert_se3(np.asarray(T_rel, dtype=float)),
            "sqrt_info": L,
            "robust": robust,
        })

    # ── optimización ─────────────────────────────────────────────────────────

    def _residual(self, edge, poses) -> np.ndarray:
        """r = L · Log( T̂_ij⁻¹ · T_i⁻¹ · T_j )  (blanqueado)."""
        e = se3_log(edge["T_meas_inv"] @ invert_se3(poses[edge["i"]]) @ poses[edge["j"]])
        return edge["sqrt_info"] @ e

    def optimize(self, iterations: int = 20) -> Dict[int, np.ndarray]:
        poses = {k: v.copy() for k, v in self._poses.items()}
        if not self._fixed and poses:                 # gauge: anclar la primera
            self._fixed.add(min(poses))
        free = sorted(k for k in poses if k not in self._fixed)
        index = {k: 6 * n for n, k in enumerate(free)}
        n_vars = 6 * len(free)
        if n_vars == 0 or not self._edges:
            return poses

        lam = 1e-6
        cost = self._total_cost(poses)
        for _ in range(iterations):
            J = np.zeros((6 * len(self._edges), n_vars))
            r = np.zeros(6 * len(self._edges))

            for k, edge in enumerate(self._edges):
                rk = self._residual(edge, poses)
                # IRLS-Huber: reescala residuo y jacobiano por √w.
                w = 1.0
                if edge["robust"]:
                    norm = np.linalg.norm(rk)
                    w = 1.0 if norm <= self.HUBER_DELTA else self.HUBER_DELTA / norm
                sw = np.sqrt(w)
                r[6 * k: 6 * k + 6] = sw * rk

                # Jacobiano numérico: perturbar cada pose implicada por la
                # derecha en cada una de sus 6 direcciones tangentes.
                for node in (edge["i"], edge["j"]):
                    if node not in index:
                        continue
                    col = index[node]
                    T_orig = poses[node]
                    for d in range(6):
                        delta = np.zeros(6)
                        delta[d] = self.JACOBIAN_EPS
                        poses[node] = T_orig @ se3_exp(delta)
                        rk_pert = self._residual(edge, poses)
                        J[6 * k: 6 * k + 6, col + d] = sw * (rk_pert - rk) / self.JACOBIAN_EPS
                    poses[node] = T_orig

            # Paso LM:  (H + λ·diag(H)) δ = −Jᵀ r
            H = J.T @ J
            g = -J.T @ r
            delta = np.linalg.solve(H + lam * np.diag(np.diag(H)) + 1e-12 * np.eye(n_vars), g)

            trial = {k: v.copy() for k, v in poses.items()}
            for node, col in index.items():
                trial[node] = trial[node] @ se3_exp(delta[col: col + 6])
            trial_cost = self._total_cost(trial)

            if trial_cost < cost:                     # mejora: aceptar y confiar más
                poses, cost, lam = trial, trial_cost, max(lam / 3.0, 1e-9)
            else:                                     # empeora: paso más conservador
                lam *= 5.0
            if np.linalg.norm(delta) < 1e-10:
                break

        return poses

    def _total_cost(self, poses) -> float:
        """½ Σ w_k·‖r_k‖² (con el mismo peso robusto que usa el paso)."""
        total = 0.0
        for edge in self._edges:
            rk = self._residual(edge, poses)
            norm = np.linalg.norm(rk)
            w = 1.0
            if edge["robust"] and norm > self.HUBER_DELTA:
                w = self.HUBER_DELTA / norm
            total += 0.5 * w * float(norm ** 2)
        return total
