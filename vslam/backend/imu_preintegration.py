"""Preintegración de IMU en la variedad (referencia NumPy, v1.1 hito 1).

Lupton & Sukkarieh (2012) plantearon el truco; Forster, Carlone, Dellaert y
Scaramuzza (RSS 2015 / TRO 2017) lo formularon en la variedad de SO(3). Esta
es la referencia LEGIBLE del repo — la gemela de rendimiento es
`gtsam.PreintegratedImuMeasurements` (mismo modelo, motor C++), verificada por
el test de equivalencia, como GTSAM↔NumPy en el BA.

─── La matemática: por qué preintegrar ───────────────────────────────────────
El IMU muestrea a 200 Hz; los keyframes llegan a ~10 Hz. Entre dos nodos i, j
del grafo hay ~20-100 medidas (ω̃_k, ã_k). Integrarlas da el estado en j…
pero DEPENDE del estado en i: cada vez que el optimizador mueva (R_i, v_i)
habría que re-integrar todo el paquete. La salida es integrar cantidades
RELATIVAS, expresadas en el cuerpo en i, que solo dependen de las medidas y
del sesgo asumido b̄ = (b_g, b_a):

    ΔR_ij = ∏_k Exp((ω̃_k − b_g)·Δt)                      (actitud relativa)
    Δv_ij = Σ_k ΔR_ik·(ã_k − b_a)·Δt                      (velocidad relativa)
    Δp_ij = Σ_k [Δv_ik·Δt + ½·ΔR_ik·(ã_k − b_a)·Δt²]      (posición relativa)

Se integran UNA vez. El estado en j se recompone añadiendo lo que el frame
del cuerpo no ve — la gravedad y la velocidad de arrastre:

    R_j = R_i·ΔR_ij
    v_j = v_i + g·Δt_ij + R_i·Δv_ij
    p_j = p_i + v_i·Δt_ij + ½·g·Δt_ij² + R_i·Δp_ij

OJO al modelo del sensor: el acelerómetro mide FUERZA ESPECÍFICA en el cuerpo
(f = Rᵀ·(a_mundo − g)), no aceleración: un IMU en reposo lee +9.81 en su eje
vertical. Por eso la g reaparece en la recomposición con signo positivo.

─── El sesgo, a primer orden ─────────────────────────────────────────────────
Las Δ dependen del sesgo b̄ usado al integrar. Si el optimizador lo corrige
δb, NO se re-integra: se acumulan durante la integración los jacobianos
∂Δ/∂b y se corrige linealmente (Forster, ec. 44):

    ΔR(b̄+δb) ≈ ΔR(b̄)·Exp(J_R_bg·δb_g)
    Δv(b̄+δb) ≈ Δv(b̄) + J_v_bg·δb_g + J_v_ba·δb_a
    Δp(b̄+δb) ≈ Δp(b̄) + J_p_bg·δb_g + J_p_ba·δb_a

─── La covarianza ────────────────────────────────────────────────────────────
El error δ = [δφ, δv, δp] (9,) se propaga en cada paso con el modelo lineal
δ⁺ = A·δ + B_g·η_g + B_a·η_a (las matrices exactas están en `integrate`).
Las densidades del datasheet son CONTINUAS (σ_c, unidades/√Hz); la covarianza
de una medida discreta a paso Δt es σ_d² = σ_c²/Δt — así la incertidumbre
acumulada crece con el TIEMPO integrado, no con el número de muestras.

El residuo del futuro factor IMU (hito 3) queda documentado en `residual`.

Convenciones vs GTSAM (para leer su documentación sin sustos): aquí el orden
del error es [φ, v, p]; GTSAM usa (θ, p, v) en `preintMeasCov`. Y su
`imuBias.ConstantBias(b_a, b_g)` recibe el ACELERÓMETRO primero.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from vslam.core.lie import _left_jacobian, hat, so3_exp, so3_log

# Magnitud nominal de la gravedad [m/s²]; el vector por defecto del mundo es
# (0, 0, −GRAVITY) — mundo z-arriba (EuRoC Vicon). El frame ÓPTICO del tracker
# (z-delante, y-abajo) NO es este mundo: la conversión llegará con el hito 3.
GRAVITY = 9.81


def _right_jacobian(omega: np.ndarray) -> np.ndarray:
    """Jacobiano derecho de SO(3): Jr(ω) = Jl(−ω).

    Aparece al perturbar el argumento de Exp por la derecha:
    Exp(ω + δ) ≈ Exp(ω)·Exp(Jr(ω)·δ) — es como el ruido del giróscopo (que
    vive en el integrando) se convierte en error de la actitud integrada.
    """
    return _left_jacobian(-np.asarray(omega, dtype=np.float64))


@dataclass(frozen=True)
class ImuNoiseParams:
    """Densidades espectrales CONTINUAS del datasheet (o del sensor.yaml de
    EuRoC — `euroc_imu_params` entrega exactamente estos nombres).

    Los random walk no entran en la covarianza de la preintegración: modelan
    la DIFUSIÓN del sesgo entre keyframes (el factor de sesgo del hito 3).
    """
    gyro_noise_density: float          # [rad/s/√Hz]   ruido blanco del gyro
    accel_noise_density: float         # [m/s²/√Hz]    ruido blanco del acel.
    gyro_random_walk: float = 0.0      # [rad/s²/√Hz]  difusión del sesgo gyro
    accel_random_walk: float = 0.0     # [m/s³/√Hz]    difusión del sesgo acel.
    rate_hz: float = 200.0


class ImuPreintegration:
    """Acumula medidas IMU entre dos nodos del grafo (ver módulo).

    Uso:
        pim = ImuPreintegration(noise, gyro_bias=bg, accel_bias=ba)
        for (omega, accel, dt) in medidas:
            pim.integrate(omega, accel, dt)
        R_j, v_j, p_j = pim.predict(R_i, v_i, p_i)
    """

    def __init__(self, noise: ImuNoiseParams,
                 gyro_bias: Optional[np.ndarray] = None,
                 accel_bias: Optional[np.ndarray] = None) -> None:
        self.noise = noise
        self.gyro_bias = (np.zeros(3) if gyro_bias is None
                          else np.asarray(gyro_bias, dtype=np.float64).copy())
        self.accel_bias = (np.zeros(3) if accel_bias is None
                           else np.asarray(accel_bias, dtype=np.float64).copy())
        self.reset()

    def reset(self) -> None:
        """Vuelve al elemento neutro (Δ = identidad, covarianza 0)."""
        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)
        self.delta_t = 0.0
        # Jacobianos ∂Δ/∂b acumulados (corrección de sesgo a primer orden).
        self.J_R_bg = np.zeros((3, 3))
        self.J_v_bg = np.zeros((3, 3))
        self.J_v_ba = np.zeros((3, 3))
        self.J_p_bg = np.zeros((3, 3))
        self.J_p_ba = np.zeros((3, 3))
        # Covarianza del error [δφ, δv, δp] (9×9).
        self.cov = np.zeros((9, 9))
        self.n_samples = 0

    # ── integración ──────────────────────────────────────────────────────────

    def integrate(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        """Una medida (ω̃, ã) vigente durante dt (retención de orden cero).

        ─── La matemática: un paso de Euler en la variedad ───
        Con ω = ω̃ − b_g y a = ã − b_a (fuerza específica corregida):

            Δp ← Δp + Δv·dt + ½·ΔR·a·dt²      (usa los valores PRE-paso:
            Δv ← Δv + ΔR·a·dt                  el orden de estas líneas
            ΔR ← ΔR·Exp(ω·dt)                  es parte del modelo)

        Jacobianos de sesgo y covarianza se propagan ANTES, también con los
        valores viejos (Forster, apéndice A):

            A = [ Exp(ω·dt)ᵀ        0      0 ]   δ⁺ = A·δ + B_g·η_g + B_a·η_a
                [ −ΔR·[a]ₓ·dt       I      0 ]   η ~ N(0, σ_c²/dt · I)
                [ −½·ΔR·[a]ₓ·dt²    I·dt   I ]
            B_g = [Jr(ω·dt)·dt; 0; 0]        B_a = [0; ΔR·dt; ½·ΔR·dt²]
        """
        if dt <= 0.0:
            raise ValueError(f"dt debe ser positivo, llego {dt}")
        omega = np.asarray(gyro, dtype=np.float64) - self.gyro_bias
        a = np.asarray(accel, dtype=np.float64) - self.accel_bias
        dR_k = so3_exp(omega * dt)
        Jr = _right_jacobian(omega * dt)
        R, ax = self.delta_R, hat(a)
        Ra_dt = self.delta_R @ a * dt                      # ΔR·a·dt (reusado)

        # Covarianza (con ΔR pre-paso). σ_d² = σ_c²/dt: ver el módulo.
        A = np.eye(9)
        A[0:3, 0:3] = dR_k.T
        A[3:6, 0:3] = -R @ ax * dt
        A[6:9, 0:3] = -0.5 * R @ ax * dt * dt
        A[6:9, 3:6] = np.eye(3) * dt
        B_g = np.zeros((9, 3))
        B_g[0:3] = Jr * dt
        B_a = np.zeros((9, 3))
        B_a[3:6] = R * dt
        B_a[6:9] = 0.5 * R * dt * dt
        sig_g = self.noise.gyro_noise_density ** 2 / dt
        sig_a = self.noise.accel_noise_density ** 2 / dt
        self.cov = (A @ self.cov @ A.T
                    + sig_g * (B_g @ B_g.T) + sig_a * (B_a @ B_a.T))

        # Jacobianos de sesgo (orden: p usa los J de v viejos; v usa J_R viejo).
        self.J_p_bg += self.J_v_bg * dt - 0.5 * R @ ax @ self.J_R_bg * dt * dt
        self.J_p_ba += self.J_v_ba * dt - 0.5 * R * dt * dt
        self.J_v_bg += -R @ ax @ self.J_R_bg * dt
        self.J_v_ba += -R * dt
        self.J_R_bg = dR_k.T @ self.J_R_bg - Jr * dt

        # Los deltas, en el orden del docstring.
        self.delta_p += self.delta_v * dt + 0.5 * Ra_dt * dt
        self.delta_v += Ra_dt
        self.delta_R = self.delta_R @ dR_k
        self.delta_t += dt
        self.n_samples += 1

    # ── consumo ──────────────────────────────────────────────────────────────

    def corrected_deltas(self, gyro_bias: np.ndarray, accel_bias: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(ΔR, Δv, Δp) corregidos a PRIMER ORDEN para un sesgo distinto del
        integrado (Forster ec. 44; fórmulas en el módulo). Válido para δb
        pequeños — si el optimizador mueve el sesgo mucho, re-integrar."""
        dbg = np.asarray(gyro_bias, dtype=np.float64) - self.gyro_bias
        dba = np.asarray(accel_bias, dtype=np.float64) - self.accel_bias
        dR = self.delta_R @ so3_exp(self.J_R_bg @ dbg)
        dv = self.delta_v + self.J_v_bg @ dbg + self.J_v_ba @ dba
        dp = self.delta_p + self.J_p_bg @ dbg + self.J_p_ba @ dba
        return dR, dv, dp

    def predict(self, R_i: np.ndarray, v_i: np.ndarray, p_i: np.ndarray,
                gravity: Optional[np.ndarray] = None,
                gyro_bias: Optional[np.ndarray] = None,
                accel_bias: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estado en j desde el estado en i (recomposición con gravedad —
        fórmulas en el módulo). Con sesgos: usa los deltas corregidos."""
        g = (np.array([0.0, 0.0, -GRAVITY]) if gravity is None
             else np.asarray(gravity, dtype=np.float64))
        if gyro_bias is None and accel_bias is None:
            dR, dv, dp = self.delta_R, self.delta_v, self.delta_p
        else:
            dR, dv, dp = self.corrected_deltas(
                self.gyro_bias if gyro_bias is None else gyro_bias,
                self.accel_bias if accel_bias is None else accel_bias)
        Dt = self.delta_t
        R_j = R_i @ dR
        v_j = v_i + g * Dt + R_i @ dv
        p_j = p_i + v_i * Dt + 0.5 * g * Dt * Dt + R_i @ dp
        return R_j, v_j, p_j

    def residual(self, R_i: np.ndarray, v_i: np.ndarray, p_i: np.ndarray,
                 R_j: np.ndarray, v_j: np.ndarray, p_j: np.ndarray,
                 gravity: Optional[np.ndarray] = None,
                 gyro_bias: Optional[np.ndarray] = None,
                 accel_bias: Optional[np.ndarray] = None) -> np.ndarray:
        """Residuo (9,) del factor IMU entre los estados i y j — el corazón
        del hito 3 (aquí sirve de contrato y de test).

        ─── La matemática: medir la discrepancia en el frame de i ───
            r_φ = Log(ΔRᵀ·R_iᵀ·R_j)
            r_v = R_iᵀ·(v_j − v_i − g·Δt) − Δv
            r_p = R_iᵀ·(p_j − p_i − v_i·Δt − ½·g·Δt²) − Δp
        Cero exactamente cuando los estados cumplen la recomposición de
        `predict`. Su covarianza es `self.cov` (orden [φ, v, p]).
        """
        g = (np.array([0.0, 0.0, -GRAVITY]) if gravity is None
             else np.asarray(gravity, dtype=np.float64))
        if gyro_bias is None and accel_bias is None:
            dR, dv, dp = self.delta_R, self.delta_v, self.delta_p
        else:
            dR, dv, dp = self.corrected_deltas(
                self.gyro_bias if gyro_bias is None else gyro_bias,
                self.accel_bias if accel_bias is None else accel_bias)
        Dt = self.delta_t
        r_phi = so3_log(dR.T @ R_i.T @ R_j)
        r_v = R_i.T @ (v_j - v_i - g * Dt) - dv
        r_p = R_i.T @ (p_j - p_i - v_i * Dt - 0.5 * g * Dt * Dt) - dp
        return np.concatenate([r_phi, r_v, r_p])


def preintegrate_between(imu_ts: np.ndarray, gyro: np.ndarray,
                         accel: np.ndarray, t0: float, t1: float,
                         noise: ImuNoiseParams,
                         gyro_bias: Optional[np.ndarray] = None,
                         accel_bias: Optional[np.ndarray] = None
                         ) -> ImuPreintegration:
    """Preintegra las medidas del intervalo [t0, t1) — p. ej. entre dos frames
    de cámara. Retención de orden cero: la medida k gobierna [ts_k, ts_{k+1})
    y los bordes se RECORTAN al intervalo (a 200 Hz el error de borde es
    sub-milirradián; interpolar medidas no paga su complejidad aquí)."""
    if t1 <= t0:
        raise ValueError(f"intervalo vacio: [{t0}, {t1})")
    pim = ImuPreintegration(noise, gyro_bias, accel_bias)
    # Solo las medidas cuyo tramo [ts_k, ts_k+1) toca [t0, t1).
    k0 = max(0, int(np.searchsorted(imu_ts, t0, side="right")) - 1)
    for k in range(k0, len(imu_ts) - 1):
        lo = max(float(imu_ts[k]), t0)
        hi = min(float(imu_ts[k + 1]), t1)
        if hi <= lo:
            if imu_ts[k] >= t1:
                break
            continue
        pim.integrate(gyro[k], accel[k], hi - lo)
    return pim
