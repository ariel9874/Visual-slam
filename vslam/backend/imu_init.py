"""Inicialización visual-inercial ESTÁTICA (v1.1 hito 2).

Antes de fusionar el IMU al grafo hay que responder tres preguntas que la
visión sola no responde: ¿cuál es el sesgo del giróscopo?, ¿dónde está
"abajo"? (la gravedad fija el roll/pitch absoluto — en visión pura son gauge)
y ¿a qué velocidad vamos? EuRoC arranca con el dron apoyado: esa ventana
quieta regala las tres respuestas.

─── La matemática: qué es observable en reposo (y qué no) ────────────────────
En reposo, ω_verdadera = 0 y a_mundo = 0. Las medidas quedan:

    ω̃ = b_g + η_g            →  la MEDIA del gyro ES el sesgo (exacto;
                                 error solo por ruido: σ_c·√(rate)/√N)
    ã = −Rᵀ·g + b_a + η_a    →  la media apunta "arriba" en el cuerpo…
                                 DESPLAZADA por b_a.

De ahí, tres verdades medidas (sonda del hito, las 3 secuencias V1):
  1. b_g sale con error ~2e-3 rad/s aun VIBRANDO (motores encendidos:
     la vibración es de media cero; el GT confirma |v| ≤ 0.015 m/s).
  2. La dirección de g está ENTRELAZADA con b_a: el error angular es
     ≈ |b_a⊥|/g. Medido: 2.7° en V1_01 (|b_a| = 0.55 m/s², el ADIS16448
     arranca torcido) y 0.5-0.8° en V1_02/V1_03; corrigiendo con el b_a
     del GT, 0.35-0.54° en las TRES — el método toca el techo, el sesgo
     del acelerómetro no es observable sin movimiento (se refina en el
     grafo, hito 3).
  3. El YAW no es observable: g es invariante a rotaciones sobre la
     vertical. `attitude_from_gravity` devuelve la rotación MÍNIMA que
     alinea el "arriba" medido con +z del mundo (convención yaw = 0).

─── El detector: "quieto" NO es "std pequeña" ────────────────────────────────
Con motores en marcha, EuRoC en reposo vibra: std_acc 0.3-1.1 m/s² (¡y 2.4
en vuelo!), std_gyro 0.02-0.08 rad/s. Un umbral estricto (p. ej. std_gyro
< 0.01) declara "NUNCA estático" en V1_01/V1_02 (medido). El detector usa:
  · umbrales LAXOS calibrados para separar reposo-vibrando de vuelo
    (std_gyro < 0.06 rad/s, std_acc < 1.0 m/s²),
  · |f̄| ≈ 9.81 (en vuelo la media pierde módulo por el movimiento),
  · CONSISTENCIA entre mitades: las medias de dos sub-ventanas deben
    coincidir (gyro < 6e-3 rad/s, dirección de f̄ < 1°) — una deriva lenta
    (lo levantan en mano) pasa los umbrales de std pero falla aquí.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from vslam.core.lie import so3_exp

_G = 9.81


@dataclass(frozen=True)
class StaticImuInit:
    """Resultado de la init estática. Consumidores (hito 3): b_g como prior
    del grafo, R_wb como actitud inicial (yaw = 0), v inicial = 0."""
    t_start: float
    t_end: float
    n_samples: int
    gyro_bias: np.ndarray       # (3,) [rad/s] — media del gyro en la ventana
    accel_mean: np.ndarray      # (3,) [m/s²] — fuerza específica media (crudo,
                                #     para poder re-corregir cuando haya b_a)
    gravity_body: np.ndarray    # (3,) g en el CUERPO (norma 9.81, con el b_a
                                #     dentro — ver módulo)
    R_wb: np.ndarray            # (3,3) actitud inicial cuerpo→mundo (yaw = 0)
    gyro_std: np.ndarray        # (3,) diagnóstico de la ventana
    accel_std: np.ndarray       # (3,)


def attitude_from_gravity(accel_mean: np.ndarray) -> np.ndarray:
    """R_wb (yaw = 0) desde la fuerza específica media en reposo.

    f̄ apunta "arriba" en el cuerpo (f = −Rᵀ·g). Se busca la rotación MÍNIMA
    que lleva up_b = f̄/‖f̄‖ a e_z: eje = up_b × e_z, ángulo = arccos(up_b·e_z).
    Mínima ⇒ sin componente de giro sobre la vertical (yaw = 0 por convención;
    el yaw verdadero no es observable — ver módulo).
    """
    up_b = np.asarray(accel_mean, dtype=np.float64)
    up_b = up_b / np.linalg.norm(up_b)
    e_z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up_b, e_z)
    s, c = np.linalg.norm(axis), float(np.dot(up_b, e_z))
    if s < 1e-12:
        # up_b ya es ±e_z: identidad, o media vuelta sobre x si está invertido.
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    return so3_exp(axis / s * np.arccos(np.clip(c, -1.0, 1.0)))


def find_static_window(ts: np.ndarray, gyro: np.ndarray, accel: np.ndarray,
                       *, window: float = 2.0, step: float = 0.1,
                       max_search: float = 15.0,
                       max_gyro_std: float = 0.06,
                       max_accel_std: float = 1.0,
                       gravity_tol: float = 0.5,
                       half_gyro_tol: float = 6e-3,
                       half_angle_tol_deg: float = 1.0
                       ) -> Optional[Tuple[int, int]]:
    """Primera ventana quieta del arranque → (i0, i1) índices en ts, o None.

    Umbrales calibrados con la sonda del hito (V1_01/02/03; ver módulo):
    reposo-vibrando pasa, vuelo (std_gyro ≥ 0.1, std_acc ≥ 1.5) no. Se toma
    la PRIMERA candidata: cuanto más temprana, más lejos del despegue.
    """
    t_begin = float(ts[0])
    t0 = t_begin
    while t0 + window <= t_begin + max_search:
        i0 = int(np.searchsorted(ts, t0))
        i1 = int(np.searchsorted(ts, t0 + window))
        if i1 >= len(ts):
            break
        g_seg, a_seg = gyro[i0:i1], accel[i0:i1]
        n = i1 - i0
        if n >= 50:
            f_mean = a_seg.mean(axis=0)
            mid = i0 + n // 2
            g1, g2 = gyro[i0:mid].mean(axis=0), gyro[mid:i1].mean(axis=0)
            f1, f2 = accel[i0:mid].mean(axis=0), accel[mid:i1].mean(axis=0)
            cosang = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
            half_deg = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
            if (np.all(g_seg.std(axis=0) < max_gyro_std)
                    and np.all(a_seg.std(axis=0) < max_accel_std)
                    and abs(np.linalg.norm(f_mean) - _G) < gravity_tol
                    and np.linalg.norm(g1 - g2) < half_gyro_tol
                    and half_deg < half_angle_tol_deg):
                return i0, i1
        t0 += step
    return None


def static_imu_init(ts: np.ndarray, gyro: np.ndarray, accel: np.ndarray,
                    **kwargs) -> StaticImuInit:
    """Detecta la ventana quieta y estima b_g, g en el cuerpo y R_wb.
    Lanza RuntimeError si no hay ventana (secuencia que arranca en vuelo:
    tocará la alineación dinámica — anotada como alternativa en docs/05 §7)."""
    found = find_static_window(ts, gyro, accel, **kwargs)
    if found is None:
        raise RuntimeError(
            "no se encontro ventana estatica al inicio de la secuencia "
            "(¿arranca en vuelo? la init dinamica no esta implementada)")
    i0, i1 = found
    accel_mean = accel[i0:i1].mean(axis=0)
    gravity_body = -accel_mean / np.linalg.norm(accel_mean) * _G
    return StaticImuInit(
        t_start=float(ts[i0]), t_end=float(ts[i1 - 1]),
        n_samples=i1 - i0,
        gyro_bias=gyro[i0:i1].mean(axis=0),
        accel_mean=accel_mean,
        gravity_body=gravity_body,
        R_wb=attitude_from_gravity(accel_mean),
        gyro_std=gyro[i0:i1].std(axis=0),
        accel_std=accel[i0:i1].std(axis=0),
    )
