#!/usr/bin/env python3
"""Tests del modo VISUAL-INERCIAL de ISAM2LocalBA (v1.1 hito 3).

El par nulo/observable de la casa (lecciones 36/38), ahora con IMU: un grafo
MONOCULAR con valores iniciales corrompidos por escala ×1.3 es CIEGO a la
escala (gauge: el optimizador ni la toca) — la cadena de CombinedImuFactor,
cuyos Δv/Δp vienen en METROS, la recupera a ~1.0. Además: los sesgos
convergen (incluido b_a, que la init estática NO puede ver — lección 48) y la
cadena se RE-ANCLA tras un reset de época sin fallos.

Todo a través de la CLASE (process_keyframe con imu_data), con un mapper de
mentira: el mismo camino que recorrerá el tracker en el hito 3b.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera
from vslam.core.lie import so3_exp

try:
    import gtsam  # noqa: F401
    HAS_GTSAM = True
except ImportError:
    HAS_GTSAM = False

if HAS_GTSAM:
    from vslam.backend.gtsam_isam2 import ISAM2LocalBA
from vslam.backend.imu_preintegration import ImuNoiseParams

G_MAP = np.array([0.0, 0.0, -9.81])
RATE, DT = 200.0, 1.0 / 200.0
KF_EVERY, N_KF = 100, 10                       # KFs cada 0.5 s
BG_TRUE = np.array([0.01, -0.02, 0.015])
BA_TRUE = np.array([0.10, -0.05, 0.20])
SCALE = 1.3
NOISE = ImuNoiseParams(gyro_noise_density=1.7e-4, accel_noise_density=2.0e-3,
                       gyro_random_walk=1.9e-5, accel_random_walk=3.0e-3)
CAM = PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0,
                    width=640, height=480)


def _simulate():
    """Trayectoria suave + medidas IMU con sesgo (integración directa exacta,
    como en test_imu_preintegration)."""
    n = KF_EVERY * (N_KF - 1) + 1
    t = np.arange(n) * DT
    # Suave a propósito: la cámara debe RETENER el campo de landmarks los 10
    # KFs. La velocidad se define ANALÍTICA y acotada (v = A·sin(ωt+φ)) y
    # a_w = dv/dt — así la posición no tiene deriva secular (∫sin con fase
    # tiene término DC: la 1ª versión derivaba 4.6 m y perdía el campo).
    # La señal del IMU viene de a_w (~0.2-0.4 m/s²), no de girar mucho.
    omega = np.stack([0.06 * np.sin(0.8 * t), 0.08 * np.cos(0.5 * t),
                      0.05 * np.sin(0.3 * t + 1.0)], 1)
    amp = np.array([0.40, 0.35, 0.25])
    frq = np.array([1.1, 0.9, 0.7])
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
    rng = np.random.default_rng(3)
    lms = np.column_stack([rng.uniform(-4, 4, 40), rng.uniform(-3, 3, 40),
                           rng.uniform(5.0, 10.0, 40)])
    return t, Rs, vs, ps, np.array(gyr), np.array(acc), lms, rng


def _project(R_wc, p_wc, pt):
    pc = R_wc.T @ (pt - p_wc)
    if pc[2] < 0.5:
        return None
    u = CAM.fx * pc[0] / pc[2] + CAM.cx
    v = CAM.fy * pc[1] / pc[2] + CAM.cy
    return np.array([u, v]) if (0 <= u < 640 and 0 <= v < 480) else None


class _StubMapper:
    """Lo mínimo que process_keyframe consume del SparsePointMapper."""

    def __init__(self, poses, points, obs):
        self._poses, self._points, self._obs = poses, points, obs

    def keyframe_pose(self, kf):
        return self._poses[kf].copy()

    def point_positions(self, pids):
        return {p: self._points[p].copy() for p in pids}


def _run_sequence(with_imu: bool, scale: float, do_reset_at: int = -1):
    t, Rs, vs, ps, gyr, acc, lms, rng = _simulate()
    kf_idx = list(range(0, len(Rs), KF_EVERY))
    # Solo landmarks visibles desde TODOS los KFs: un punto que parpadea en el
    # borde del frustum puede entrar con 2 rayos casi paralelos (una obs vieja
    # pendiente + una re-aparicion) → sistema indeterminado. El tracker real
    # lo impide con los filtros de triangulacion (leccion 7); la fixture, asi.
    lms = lms[[j for j in range(len(lms))
               if all(_project(Rs[k], ps[k], lms[j]) is not None
                      for k in kf_idx)]]
    assert len(lms) >= 25, f"fixture pobre: {len(lms)} landmarks"
    poses, obs = {}, {}
    for i, k in enumerate(kf_idx):
        T = np.eye(4); T[:3, :3] = Rs[k]; T[:3, 3] = ps[k] * scale
        poses[i] = T
        kf_obs = []
        for j, lm in enumerate(lms):
            uv = _project(Rs[k], ps[k], lm)
            if uv is not None:
                kf_obs.append((j, uv + rng.normal(0, 0.5, 2)))
        obs[i] = kf_obs
    mapper = _StubMapper(poses, {j: lm * scale for j, lm in enumerate(lms)}, obs)

    backend = ISAM2LocalBA(CAM)
    if with_imu:
        backend.configure_imu(NOISE, G_MAP, T_cam_imu=np.eye(4),
                              init_gyro_bias=BG_TRUE + 2e-3,
                              init_velocity=vs[0] * scale,
                              vel_prior_sigma=0.2)
    got = {}
    for i, k in enumerate(kf_idx):
        new_obs = [(i, j, uv) for j, uv in obs[i]]
        window = list(range(max(0, i - 4), i + 1))
        seg = None
        if with_imu and i > 0:
            k0, k1 = kf_idx[i - 1], kf_idx[i]
            seg = (t[k0:k1 + 1], gyr[k0:k1], acc[k0:k1])
        out = backend.process_keyframe(mapper, window, new_obs,
                                       imu_data=seg)
        assert out is not None, f"update fallo en KF {i}"
        got.update(out[0])
        if i == do_reset_at:
            backend.reset()
    # Escala recuperada: mediana de distancias entre KFs consecutivos vs verdad.
    ratios = [np.linalg.norm(got[i][:3, 3] - got[i - 1][:3, 3])
              / np.linalg.norm(ps[kf_idx[i]] - ps[kf_idx[i - 1]])
              for i in range(1, N_KF) if i in got and i - 1 in got]
    return backend, float(np.median(ratios))


def test_imu_makes_scale_observable():
    backend_no, s_no = _run_sequence(with_imu=False, scale=SCALE)
    backend_si, s_si = _run_sequence(with_imu=True, scale=SCALE)
    assert backend_no.n_failures == 0 and backend_si.n_failures == 0
    # Sin IMU la escala corrupta es gauge: se queda arriba (medido 1.29).
    assert s_no > 1.2, f"sin IMU deberia quedarse ~1.3, dio {s_no:.3f}"
    # Con IMU los delta_v/delta_p en metros la recuperan (medido ~0.99).
    assert abs(s_si - 1.0) < 0.05, f"con IMU deberia ~1.0, dio {s_si:.3f}"
    print(f"  escala: sin IMU {s_no:.3f} | con IMU {s_si:.3f}")


def test_imu_recovers_biases():
    backend, _ = _run_sequence(with_imu=True, scale=SCALE)
    bg, ba = backend.last_bias
    e_bg = np.linalg.norm(bg - BG_TRUE)
    e_ba = np.linalg.norm(ba - BA_TRUE)
    print(f"  sesgos: b_g err {e_bg:.2e} rad/s | b_a err {e_ba:.3f} m/s2")
    # b_a arranca en 0 (no observable estatico, leccion 48): el grafo lo
    # RECUPERA. Umbrales con margen sobre lo medido (batch: 7e-4 / 0.010).
    assert e_bg < 5e-3, f"b_g err {e_bg:.2e}"
    assert e_ba < 0.15, f"b_a err {e_ba:.3f}"


def test_chain_reanchors_after_reset():
    # Mapa SIN corromper (tras un bucle real el mapper ya esta corregido):
    # el reset a mitad de secuencia debe re-anclar V/B sin fallos y las
    # poses posteriores deben seguir cerca de la verdad.
    backend, s = _run_sequence(with_imu=True, scale=1.0, do_reset_at=5)
    assert backend.n_failures == 0, f"{backend.n_failures} updates fallidos"
    assert abs(s - 1.0) < 0.05, f"escala tras reset {s:.3f}"
    assert np.all(np.isfinite(backend.last_velocity))


def main() -> int:
    if not HAS_GTSAM:
        print("OK (saltado): gtsam no esta instalado.")
        return 0
    test_imu_makes_scale_observable()
    print("OK: el IMU hace observable la escala (nulo/observable)")
    test_imu_recovers_biases()
    print("OK: el grafo recupera los sesgos (incluido b_a)")
    test_chain_reanchors_after_reset()
    print("OK: la cadena re-ancla tras un reset de epoca")
    print("OK: los 3 tests del modo VI de iSAM2 (v1.1 hito 3) pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
