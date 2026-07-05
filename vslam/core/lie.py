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


# ═══════════════════════════ Sim(3): similitudes ═════════════════════════════
# El grupo de las SIMILITUDES rígidas: S = [[s·R, t], [0, 1]], que actúa como
# x' = s·R·x + t. Es el grupo natural del SLAM MONOCULAR: como la escala es
# inobservable, la deriva vive en 7 grados de libertad (no 6), y un cierre de
# bucle solo puede redistribuirla si el grafo de poses optimiza en Sim(3)
# (Strasdat et al., "Scale Drift-Aware Large Scale Monocular SLAM", RSS 2010 —
# lo comprobamos empíricamente en v0.35: el grafo SE(3) EMPEORABA el ATE con
# 14% de deriva de escala).
#
# ─── La matemática ───
# Álgebra: ξ = [ρ, ω, λ] ∈ ℝ⁷ (traslación, rotación, log-escala), con
# representación matricial  ξ^ = [[λ·I + [ω]_×, ρ], [0, 0]]  (4×4). Como λ·I
# conmuta con [ω]_×, la exponencial del bloque superior factoriza:
#
#     exp(λ·I + [ω]_×) = e^λ · Exp_SO3(ω) = s·R
#
# y la traslación es t = W·ρ, donde W = ∫₀¹ exp(u·(λI + [ω]_×)) du generaliza
# la V de SE(3) acoplando giro Y escala al avance:
#
#     W = A·[ω]_× + B·[ω]_×² + C·I
#
# con coeficientes que degeneran suavemente a los de SE(3) cuando λ → 0
# (C → 1) y al caso euclidiano cuando además θ → 0 (A → ½, B → ⅙). Las
# cuatro ramas de Taylor están en _sim3_W; el test del repo las valida contra
# la exponencial de matrices por serie (la verdad numérica sin fórmulas).


def _sim3_W(omega: np.ndarray, lam: float) -> np.ndarray:
    """W(ω, λ) = ∫₀¹ exp(u·(λI + [ω]ₓ)) du — el acoplador de Sim(3)."""
    theta = np.linalg.norm(omega)
    Wx = hat(omega)
    s = np.exp(lam)
    if abs(lam) < _EPS:
        C = 1.0
        if theta < _EPS:
            A, B = 0.5, 1.0 / 6.0
        else:
            A = (1.0 - np.cos(theta)) / theta ** 2
            B = (theta - np.sin(theta)) / theta ** 3
    else:
        C = (s - 1.0) / lam
        if theta < _EPS:
            A = ((lam - 1.0) * s + 1.0) / lam ** 2
            B = (s * (0.5 * lam ** 2 - lam + 1.0) - 1.0) / lam ** 3
        else:
            a = s * np.sin(theta)
            b = s * np.cos(theta)
            c = theta ** 2 + lam ** 2
            A = (a * lam + (1.0 - b) * theta) / (theta * c)
            B = (C - ((b - 1.0) * lam + a * theta) / c) / theta ** 2
    return A * Wx + B * (Wx @ Wx) + C * np.eye(3)


def sim3_exp(xi: np.ndarray) -> np.ndarray:
    """Exp de Sim(3): ξ = [ρ, ω, λ] (7,) → matriz 4x4 [[e^λ·R, W·ρ], [0, 1]]."""
    rho, omega, lam = np.asarray(xi[:3], float), np.asarray(xi[3:6], float), float(xi[6])
    S = np.eye(4)
    S[:3, :3] = np.exp(lam) * so3_exp(omega)
    S[:3, 3] = _sim3_W(omega, lam) @ rho
    return S


def sim3_log(S: np.ndarray) -> np.ndarray:
    """Log de Sim(3): matriz 4x4 → ξ = [ρ, ω, λ] (7,).

    La escala se extrae del determinante (det(s·R) = s³ porque det R = 1),
    y ρ resolviendo W·ρ = t (mismo truco de estabilidad que en se3_log).
    """
    sR = S[:3, :3]
    s = np.linalg.det(sR) ** (1.0 / 3.0)
    lam = np.log(s)
    omega = so3_log(sR / s)
    rho = np.linalg.solve(_sim3_W(omega, lam), S[:3, 3])
    return np.concatenate([rho, omega, [lam]])


def sim3_inv(S: np.ndarray) -> np.ndarray:
    """Inversa cerrada: si x' = s·R·x + t, entonces x = (1/s)·Rᵀ·(x' − t).

    ⇒  S⁻¹ = [[(1/s)·Rᵀ, −(1/s)·Rᵀ·t], [0, 1]]
    """
    sR, t = S[:3, :3], S[:3, 3]
    s2 = np.linalg.det(sR) ** (2.0 / 3.0)      # s², para invertir s·R de golpe
    Si = np.eye(4)
    Si[:3, :3] = sR.T / s2                      # (s·R)ᵀ/s² = (1/s)·Rᵀ
    Si[:3, 3] = -(Si[:3, :3] @ t)
    return Si
