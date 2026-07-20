#!/usr/bin/env python3
"""Tests del prior IMU del frontend (v1.1 hito 4).

El matching guiado (lección 22) navega con un prior de pose; hasta ahora era
velocidad constante (lección 24), que falla exactamente bajo ACELERACIÓN — el
vuelo de V1_02/V1_03. Aquí se verifica el CABLEADO de la predicción IMU del
tracker (_imu_advance/_imu_anchor), no la matemática de la preintegración
(esa la cubre test_imu_preintegration, hito 1):

1. Encadenar predicciones frame a frame SIN visión (coast) reproduce el
   dead-reckoning de referencia EXACTO (misma integración de Euler).
2. Con movimiento agresivo (4-5 m/s², 1-2 rad/s variables), el prior IMU
   predice el siguiente frame ÓRDENES mejor que velocidad constante.
3. El anclaje refresca v del grafo SOLO con eslabón nuevo (tras un reset de
   época el tail es None y NO pisa la velocidad propagada).
4. Sin segmento (driver sin esos frames) devuelve None: el llamador cae a
   velocidad constante — nunca crashea.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera
from vslam.core.lie import so3_exp, so3_log

try:
    import gtsam  # noqa: F401
    HAS_GTSAM = True
except ImportError:
    HAS_GTSAM = False

if HAS_GTSAM:
    from vslam.frontend.tracker import PnPTracker
from vslam.backend.imu_preintegration import ImuNoiseParams

G_MAP = np.array([0.0, 0.0, -9.81])
RATE, DT = 200.0, 1.0 / 200.0
FRAME_EVERY = 10                               # cámara a 20 Hz
N_FRAMES = 40
BG_TRUE = np.array([0.01, -0.02, 0.015])
BA_TRUE = np.array([0.10, -0.05, 0.20])
NOISE = ImuNoiseParams(gyro_noise_density=1.7e-4, accel_noise_density=2.0e-3,
                       gyro_random_walk=1.9e-5, accel_random_walk=3.0e-3)
CAM = PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0,
                    width=640, height=480)


def _simulate():
    """Vuelo AGRESIVO (aquí no hay matching que retener): aceleraciones de
    ~4-5 m/s² y tasas angulares de 1-2 rad/s VARIABLES — el régimen donde la
    velocidad constante se rompe. IMU perfecto con sesgos (integración de
    Euler directa, la misma retención de orden cero que la preintegración)."""
    n = FRAME_EVERY * N_FRAMES + 1
    t = np.arange(n) * DT
    omega = np.stack([1.2 * np.sin(4.0 * t), 0.9 * np.cos(3.0 * t),
                      0.7 * np.sin(2.0 * t + 1.0)], 1)
    amp = np.array([0.80, 0.70, 0.50])
    frq = np.array([6.0, 5.0, 4.0])
    ph = np.array([0.0, 1.0, 2.0])
    a_w = amp * frq * np.cos(frq * t[:, None] + ph)
    R = np.eye(3); v = amp * np.sin(ph); p = np.zeros(3)
    Rs, vs, ps, gyr, acc = [R.copy()], [v.copy()], [p.copy()], [], []
    for k in range(n - 1):
        gyr.append(omega[k] + BG_TRUE)
        acc.append(R.T @ (a_w[k] - G_MAP) + BA_TRUE)
        p = p + v * DT + 0.5 * a_w[k] * DT * DT
        v = v + a_w[k] * DT
        R = R @ so3_exp(omega[k] * DT)
        Rs.append(R.copy()); vs.append(v.copy()); ps.append(p.copy())
    return t, Rs, vs, ps, np.array(gyr), np.array(acc)


def _pose(Rs, ps, frame):
    T = np.eye(4)
    T[:3, :3] = Rs[frame * FRAME_EVERY]
    T[:3, 3] = ps[frame * FRAME_EVERY]
    return T


def _make_tracker(t, gyr, acc, vs):
    """Tracker VI con cámara = cuerpo (T_cam_imu = I) y proveedor de
    segmentos por frame-id, como el del driver de examples/06."""
    def segment(a, b):
        i0, i1 = a * FRAME_EVERY, b * FRAME_EVERY
        return t[i0:i1 + 1], gyr[i0:i1], acc[i0:i1]

    tracker = PnPTracker(CAM, ba_backend="isam2")
    tracker.enable_imu(NOISE, G_MAP, init_gyro_bias=BG_TRUE,
                       init_accel_bias=BA_TRUE, init_velocity=vs[0],
                       segment_provider=segment)
    return tracker


def test_chained_prediction_is_dead_reckoning():
    # Coast puro: nadie re-ancla → el estado encadena preintegraciones. Debe
    # reproducir la integración directa EXACTA (hito 1: predict ==
    # dead-reckoning a 1e-13; encadenado, deja redondeo, no modelo).
    t, Rs, vs, ps, gyr, acc = _simulate()
    tracker = _make_tracker(t, gyr, acc, vs)
    tracker._T_prev = _pose(Rs, ps, 0)
    tracker._initialized = True
    worst_p = worst_r = 0.0
    for f in range(1, N_FRAMES + 1):
        tracker._frame_idx = f
        T_pred = tracker._imu_advance()
        assert T_pred is not None
        T_true = _pose(Rs, ps, f)
        worst_p = max(worst_p, float(np.linalg.norm(T_pred[:3, 3] - T_true[:3, 3])))
        worst_r = max(worst_r, float(np.linalg.norm(
            so3_log(T_pred[:3, :3].T @ T_true[:3, :3]))))
    assert worst_p < 1e-9, f"deriva posicion {worst_p:.2e} m"
    assert worst_r < 1e-9, f"deriva rotacion {worst_r:.2e} rad"


def test_imu_prior_beats_constant_velocity():
    # Visión perfecta hasta el frame k-1 (re-anclando cada frame, como hace
    # _track_step); predicción del frame k: IMU vs velocidad constante.
    t, Rs, vs, ps, gyr, acc = _simulate()
    tracker = _make_tracker(t, gyr, acc, vs)
    tracker._T_prev = _pose(Rs, ps, 0)
    tracker._initialized = True
    ratios_p, ratios_r = [], []
    for f in range(1, N_FRAMES + 1):
        tracker._frame_idx = f
        T_imu = tracker._imu_advance()
        T_true = _pose(Rs, ps, f)
        if f >= 2:
            T_cv = _pose(Rs, ps, f - 1) @ (
                np.linalg.inv(_pose(Rs, ps, f - 2)) @ _pose(Rs, ps, f - 1))
            e_imu = np.linalg.norm(T_imu[:3, 3] - T_true[:3, 3])
            e_cv = np.linalg.norm(T_cv[:3, 3] - T_true[:3, 3])
            r_imu = np.linalg.norm(so3_log(T_imu[:3, :3].T @ T_true[:3, :3]))
            r_cv = np.linalg.norm(so3_log(T_cv[:3, :3].T @ T_true[:3, :3]))
            ratios_p.append((e_imu, e_cv)); ratios_r.append((r_imu, r_cv))
        tracker._imu_anchor(T_true)        # la "visión" acepta la pose real
        tracker._T_prev = T_true
    med_imu_p = float(np.median([a for a, _ in ratios_p]))
    med_cv_p = float(np.median([b for _, b in ratios_p]))
    med_imu_r = float(np.median([a for a, _ in ratios_r]))
    med_cv_r = float(np.median([b for _, b in ratios_r]))
    assert med_imu_p < 1e-6, f"IMU pos {med_imu_p:.2e} m"
    assert med_imu_r < 1e-6, f"IMU rot {med_imu_r:.2e} rad"
    assert med_cv_p > 20 * med_imu_p, \
        f"CV pos {med_cv_p:.2e} no domina a IMU {med_imu_p:.2e}"
    assert med_cv_r > 20 * med_imu_r, \
        f"CV rot {med_cv_r:.2e} no domina a IMU {med_imu_r:.2e}"
    return med_imu_p, med_cv_p, med_imu_r, med_cv_r


def test_anchor_refreshes_velocity_only_on_new_tail():
    t, Rs, vs, ps, gyr, acc = _simulate()
    tracker = _make_tracker(t, gyr, acc, vs)
    tracker._T_prev = _pose(Rs, ps, 0)
    tracker._initialized = True
    tracker._frame_idx = 1
    tracker._imu_advance()
    v_prop = tracker._imu_pred["v"].copy()
    # Sin eslabón nuevo (tail None, época recién nacida): v NO se pisa.
    tracker._imu_anchor(_pose(Rs, ps, 1))
    assert np.allclose(tracker._imu_pred["v"], v_prop)
    # El grafo procesa un eslabón (caja blanca: tail avanza y publica v).
    v_graph = np.array([9.0, 9.0, 9.0])
    tracker._isam2._imu_prev = 1
    tracker._isam2._vel_last = v_graph
    tracker._imu_anchor(_pose(Rs, ps, 1))
    assert np.allclose(tracker._imu_pred["v"], v_graph)
    # Mismo tail otra vez: conserva la propagación, no re-pisa.
    tracker._imu_pred["v"] = v_prop.copy()
    tracker._imu_anchor(_pose(Rs, ps, 1))
    assert np.allclose(tracker._imu_pred["v"], v_prop)


def test_no_segment_falls_back_to_none():
    t, Rs, vs, ps, gyr, acc = _simulate()
    tracker = _make_tracker(t, gyr, acc, vs)

    def broken(a, b):
        raise KeyError(a)

    tracker._imu_provider = broken
    tracker._T_prev = _pose(Rs, ps, 0)
    tracker._frame_idx = 1
    assert tracker._imu_advance() is None


def main() -> int:
    if not HAS_GTSAM:
        print("OK (saltado): gtsam no esta instalado.")
        return 0
    test_chained_prediction_is_dead_reckoning()
    print("OK: la prediccion encadenada == dead-reckoning de referencia")
    ep, ecv, rp, rcv = test_imu_prior_beats_constant_velocity()
    print(f"OK: prior IMU {ep:.1e} m / {rp:.1e} rad vs velocidad constante "
          f"{ecv:.1e} m / {rcv:.1e} rad (mediana por frame)")
    test_anchor_refreshes_velocity_only_on_new_tail()
    print("OK: el anclaje refresca v solo con eslabon nuevo del grafo")
    test_no_segment_falls_back_to_none()
    print("OK: sin segmento devuelve None (fallback a velocidad constante)")
    print("OK: los 4 tests del prior IMU del frontend (v1.1 hito 4) pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
