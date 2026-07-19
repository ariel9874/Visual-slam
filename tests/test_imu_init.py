#!/usr/bin/env python3
"""Tests de la init visual-inercial estática (v1.1 hito 2).

(1) SINTÉTICO: dron "quieto pero VIBRANDO" (el caso real de EuRoC: ruido de
    media cero, std grande) 2.5 s + vuelo con sinusoides. El detector debe
    elegir una ventana DENTRO del tramo quieto; b_g se recupera al nivel del
    ruido; la dirección de g queda desplazada por el b_a sintético (la
    observabilidad del módulo) y R_wb alinea el "arriba" medido con +z exacto.
(2) SIN REPOSO: secuencia que arranca en vuelo → find_static_window devuelve
    None y static_imu_init lanza RuntimeError.
(3) EuRoC REAL (las V1 que estén en data/euroc/): contra el GT de estado —
    b_g < 4e-3 rad/s; dir(g) CRUDA < 3.5 grados (V1_01 está limitada por su
    |b_a| = 0.55 m/s², medido 2.6) y < 1.0 grados CORRIGIENDO con el b_a del
    GT (medido 0.35-0.63): el método toca el criterio del hito; lo que falta
    es b_a, no observable en reposo — se refina en el grafo (hito 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.backend.imu_init import (attitude_from_gravity, find_static_window,
                                    static_imu_init)
from vslam.core.lie import so3_exp

G_W = np.array([0.0, 0.0, -9.81])
DATA = Path(__file__).resolve().parents[1] / "data" / "euroc"


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _synthetic(static_s: float = 2.5, total_s: float = 6.0, rate: float = 200.0,
               seed: int = 0):
    """IMU sintético: reposo vibrando y luego vuelo. Devuelve también la
    verdad (R_wb, b_g, b_a) para comparar."""
    rng = np.random.default_rng(seed)
    n = int(total_s * rate)
    ts = np.arange(n) / rate
    R_wb = so3_exp(np.array([0.15, -0.10, 0.70]))    # roll/pitch/yaw de verdad
    b_g = np.array([0.01, -0.02, 0.076])
    b_a = np.array([0.05, -0.03, 0.08])
    f_static = R_wb.T @ (-G_W)                        # fuerza específica quieta
    gyro = b_g + rng.normal(0.0, 0.02, (n, 3))        # vibración: media cero
    accel = f_static + b_a + rng.normal(0.0, 0.30, (n, 3))
    fly = ts >= static_s                              # vuelo: sinusoides gordas
    t_f = ts[fly, None]
    gyro[fly] += 1.5 * np.sin(2.0 * np.pi * 1.3 * t_f + [0.0, 1.0, 2.0])
    accel[fly] += 4.0 * np.sin(2.0 * np.pi * 0.9 * t_f + [0.5, 1.5, 2.5])
    return ts, gyro, accel, R_wb, b_g, b_a


def test_static_synthetic():
    ts, gyro, accel, R_wb, b_g, b_a = _synthetic()
    init = static_imu_init(ts, gyro, accel)
    # La ventana cae entera dentro del tramo quieto [0, 2.5).
    assert init.t_end < 2.5 + 1e-9, f"ventana invade el vuelo: {init.t_end}"
    # b_g al nivel del ruido de la media (0.02/sqrt(400) ~ 1e-3 por eje).
    e_bg = np.linalg.norm(init.gyro_bias - b_g)
    assert e_bg < 4e-3, f"b_g err {e_bg:.2e}"
    # dir(g): cruda desplazada por b_a (aqui |b_a_perp|/g ~ 0.35 deg);
    # corrigiendo con el b_a verdadero queda el puro ruido.
    g_body_true = R_wb.T @ G_W
    assert _angle_deg(init.gravity_body, g_body_true) < 1.0
    assert _angle_deg(-(init.accel_mean - b_a), g_body_true) < 0.3
    # R_wb alinea el arriba MEDIDO con +z exacto (consistencia interna).
    assert _angle_deg(init.R_wb @ init.gravity_body, G_W) < 1e-6


def test_no_static_window_raises():
    ts, gyro, accel, *_ = _synthetic(static_s=0.0)   # todo vuelo
    assert find_static_window(ts, gyro, accel) is None
    try:
        static_imu_init(ts, gyro, accel)
    except RuntimeError:
        return
    raise AssertionError("static_imu_init debio lanzar sin ventana quieta")


def test_attitude_from_gravity_degenerate():
    # Ya nivelado: identidad. Boca abajo: media vuelta (det +1, g a -z).
    assert np.allclose(attitude_from_gravity(np.array([0.0, 0.0, 9.81])),
                       np.eye(3))
    R = attitude_from_gravity(np.array([0.0, 0.0, -9.81]))
    assert np.allclose(R @ np.array([0.0, 0.0, -1.0]), [0.0, 0.0, 1.0])
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_euroc_real():
    from vslam.io.dataset import (_quat_wxyz_to_R, read_euroc_imu,
                                  read_euroc_state)
    seqs = [d for d in ("V1_01_easy", "V1_02_medium", "V1_03_difficult")
            if (DATA / d).is_dir()]
    if not seqs:
        print("  (data/euroc/ sin secuencias V1: test real saltado)")
        return
    for seq in seqs:
        root = DATA / seq
        ts, gyro, acc = read_euroc_imu(root)
        ts_g, _p, q, _v, bg_gt, ba_gt = read_euroc_state(root)
        init = static_imu_init(ts, gyro, acc)
        assert init.t_start - ts[0] < 5.0, "ventana sospechosamente tardia"
        i = min(int(np.searchsorted(ts_g, 0.5 * (init.t_start + init.t_end))),
                len(ts_g) - 1)
        g_body_gt = _quat_wxyz_to_R(q[i]).T @ G_W
        e_bg = np.linalg.norm(init.gyro_bias - bg_gt[i])
        e_raw = _angle_deg(init.gravity_body, g_body_gt)
        e_corr = _angle_deg(-(init.accel_mean - ba_gt[i]), g_body_gt)
        print(f"  {seq}: ventana [{init.t_start - ts[0]:.1f}, "
              f"{init.t_end - ts[0]:.1f}] s | b_g err {e_bg:.2e} rad/s | "
              f"dir(g) cruda {e_raw:.2f} deg, con b_a GT {e_corr:.2f} deg")
        # Medido (jul 2026): b_g 1.9-2.3e-3; cruda 0.44-2.60 (V1_01 = b_a
        # gordo); corregida 0.35-0.63. Margen ~1.5x: esto protege convenciones.
        assert e_bg < 4e-3, f"{seq}: b_g err {e_bg:.2e}"
        assert e_raw < 3.5, f"{seq}: dir(g) cruda {e_raw:.2f} deg"
        assert e_corr < 1.0, f"{seq}: dir(g) con b_a {e_corr:.2f} deg"


def main() -> int:
    test_static_synthetic()
    print("OK: init estatica sintetica (ventana, b_g, dir(g), R_wb)")
    test_no_static_window_raises()
    print("OK: sin reposo -> None / RuntimeError")
    test_attitude_from_gravity_degenerate()
    print("OK: attitude_from_gravity casos degenerados")
    test_euroc_real()
    print("OK: init estatica sobre EuRoC real")
    print("OK: los 4 tests de init VI (v1.1 hito 2) pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
