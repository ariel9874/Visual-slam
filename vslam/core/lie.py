"""Álgebra de Lie de SO(3) y SE(3): los mapas Exp y Log.

Son el puente entre el mundo de las MATRICES (donde se componen poses) y el
mundo de los VECTORES (donde se optimiza). El backend los usa para calcular
residuos y aplicar actualizaciones sin salirse de la variedad.

─── La matemática: por qué hacen falta ───────────────────────────────────────
SO(3) y SE(3) son grupos de Lie: variedades curvas con estructura de grupo.
No se puede "sumar ruido" ni "dar un paso de gradiente" sumando matrices —
I + δ ya no es una rotación. La solución: trabajar en el ESPACIO TANGENTE en
la identidad (el álgebra de Lie), que sí es un espacio vectorial:

    so(3) = { [ω]_× : ω ∈ ℝ³ }          (matrices antisimétricas)
    se(3) = { (ρ, ω) ∈ ℝ⁶ }             (traslación, rotación)

y moverse entre ambos mundos con la exponencial de matrices y su inversa:

    Exp: ℝ³/ℝ⁶ → grupo      (un "paso" vectorial → un movimiento rígido)
    Log: grupo → ℝ³/ℝ⁶      (un movimiento rígido → su paso vectorial)

Fórmulas cerradas (θ = ‖ω‖, k = ω/θ):

    Exp_SO3(ω) = I + sin θ·[k]_× + (1 − cos θ)·[k]_×²          (Rodrigues)

    Exp_SE3(ρ, ω) = [[Exp_SO3(ω),  V·ρ], [0, 1]]
    V = I + (1 − cos θ)/θ²·[ω]_× + (θ − sin θ)/θ³·[ω]_×²

V (el "jacobiano izquierdo" de SO(3)) aparece porque al girar MIENTRAS se
avanza, la traslación efectiva se curva: V·ρ es el arco recorrido, no la
cuerda. Con θ → 0, V → I y todo degenera suavemente al caso euclidiano
(por eso cada fórmula tiene su serie de Taylor de guardia).

Convención del repo: el vector tangente es ξ = [ρ, ω] (traslación primero).
GTSAM usa (ω, ρ): cuidado al comparar con su documentación.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def hat(v: np.ndarray) -> np.ndarray:
    """Matriz antisimétrica [v]_× tal que [v]_×·u = v × u."""
    x, y, z = v
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]])


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Exp de SO(3): vector eje-ángulo (3,) → matriz de rotación (Rodrigues)."""
    theta = np.linalg.norm(omega)
    W = hat(omega)
    if theta < _EPS:
        # Taylor: sin θ/θ → 1, (1−cos θ)/θ² → 1/2
        return np.eye(3) + W + 0.5 * (W @ W)
    return (np.eye(3)
            + (np.sin(theta) / theta) * W
            + ((1.0 - np.cos(theta)) / theta ** 2) * (W @ W))


def so3_log(R: np.ndarray) -> np.ndarray:
    """Log de SO(3): matriz de rotación → vector eje-ángulo (3,).

    Tres regímenes numéricos (la fórmula ingenua ω = θ/(2 sin θ)·vee(R − Rᵀ)
    divide por sin θ, que se anula en θ = 0 y θ = π):
      · θ ≈ 0: serie de Taylor, ω ≈ ½·vee(R − Rᵀ).
      · θ ≈ π: R ≈ 2kkᵀ − I ⇒ kkᵀ = (R + I)/2; se extrae k de la columna
        con mayor diagonal (el signo de k es irrelevante: Exp(πk) = Exp(−πk)).
      · resto: la fórmula estándar.
    """
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(trace)
    vee = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < _EPS:
        return vee                                # ω ≈ ½ vee(R − Rᵀ)
    if np.pi - theta < 1e-6:
        S = (R + np.eye(3)) / 2.0                 # = kkᵀ en θ = π
        i = int(np.argmax(np.diag(S)))
        k = S[:, i] / np.sqrt(S[i, i])
        return theta * k
    return (theta / np.sin(theta)) * vee


def _left_jacobian(omega: np.ndarray) -> np.ndarray:
    """V(ω): acopla rotación y traslación en Exp_SE3 (fórmula en el módulo)."""
    theta = np.linalg.norm(omega)
    W = hat(omega)
    if theta < _EPS:
        return np.eye(3) + 0.5 * W + (W @ W) / 6.0
    return (np.eye(3)
            + ((1.0 - np.cos(theta)) / theta ** 2) * W
            + ((theta - np.sin(theta)) / theta ** 3) * (W @ W))


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exp de SE(3): vector tangente ξ = [ρ, ω] (6,) → transformación 4x4."""
    rho, omega = np.asarray(xi[:3], float), np.asarray(xi[3:], float)
    T = np.eye(4)
    T[:3, :3] = so3_exp(omega)
    T[:3, 3] = _left_jacobian(omega) @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """Log de SE(3): transformación 4x4 → vector tangente ξ = [ρ, ω] (6,).

    ρ se recupera resolviendo V·ρ = t (V es invertible para θ < 2π; resolver
    el sistema es más simple y estable que la fórmula cerrada de V⁻¹).
    """
    omega = so3_log(T[:3, :3])
    rho = np.linalg.solve(_left_jacobian(omega), T[:3, 3])
    return np.concatenate([rho, omega])
