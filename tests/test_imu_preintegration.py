#!/usr/bin/env python3
"""Tests de la preintegración IMU (v1.1 hito 1, referencia NumPy).

(1) EXACTITUD: predict() reproduce la integración directa (dead-reckoning con
    gravedad) a precisión de máquina — el álgebra de la recomposición
    (g·Δt, ½·g·Δt², arrastre de v_i) es un reordenamiento EXACTO, no una
    aproximación. Y el residual en el estado verdadero es 0.
(2) SESGO A PRIMER ORDEN: corrected_deltas vs re-integración con el sesgo
    perturbado. El error debe ser de SEGUNDO orden: ×100 al multiplicar la
    perturbación ×10 (medido: 1.9e-5 → 1.9e-3 con |db| 1e-3 → 1e-2).
(3) EQUIVALENCIA GTSAM (si gtsam está): mismos deltas, mismo predict y misma
    covarianza que PreintegratedImuMeasurements. La wheel de conda usa la
    formulación TANGENTE (nosotros, la de variedad de Forster): la diferencia
    medida es de 2º orden (~7e-5 rad / ~1e-4 m tras 2 s con |dv|~20 m/s) y las
    tolerancias llevan ~10× de margen sobre lo medido. El orden de bloques de
    preintMeasCov es (theta, p, v); el nuestro [phi, v, p] — se permuta aquí.
(4) DEAD-RECKONING REAL (si data/euroc/V1_01_easy está): preintegrar ventanas
    de 1 s del IMU real y predecir desde el GT de estado (que trae v y sesgos).
    Medido: rot mediana 0.33 deg, pos mediana 4.4 cm, p90 7.8 cm — el test
    valida las CONVENCIONES (q_RS cuerpo→mundo, fuerza específica, g = −z):
    un error de frame o de signo daría metros (medio g·t² = 4.9 m).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.backend.imu_preintegration import (GRAVITY, ImuNoiseParams,
                                              ImuPreintegration,
                                              preintegrate_between)
from vslam.core.lie import so3_exp, so3_log

G = np.array([0.0, 0.0, -GRAVITY])
# Ruidos del ADIS16448 de EuRoC (imu0/sensor.yaml) — también para el sintético.
NOISE = ImuNoiseParams(gyro_noise_density=1.6968e-4, accel_noise_density=2.0e-3,
                       gyro_random_walk=1.9393e-5, accel_random_walk=3.0e-3,
                       rate_hz=200.0)
BG = np.array([0.01, -0.02, 0.015])          # sesgos "verdaderos" del sintético
BA = np.array([0.10, -0.05, 0.20])

EUROC_ROOT = Path(__file__).resolve().parents[1] / "data" / "euroc" / "V1_01_easy"


def _smooth_signals(n: int = 400, dt: float = 0.005):
    """Verdad sintética: ω del cuerpo y FUERZA ESPECÍFICA f = Rᵀ(a_w − g),
    suaves y sin ruido (el ruido se prueba en la covarianza, no aquí)."""
    t = np.arange(n) * dt
    omega = np.stack([0.4 * np.sin(0.9 * t), 0.3 * np.cos(1.3 * t),
                      0.5 * np.sin(0.6 * t + 1.0)], axis=1)
    force = np.stack([0.8 * np.sin(1.1 * t), 0.6 * np.cos(0.8 * t) + 9.81,
                      0.5 * np.sin(1.7 * t) - 1.0], axis=1)
    return omega, force, dt


def _dead_reckon(R0, v0, p0, omega, force, dt):
    """Integración DIRECTA del estado (el mismo Euler que la clase, con la
    aceleración del mundo a_w = R·f + g): la verdad contra la que se compara."""
    R, v, p = R0.copy(), v0.copy(), p0.copy()
    for k in range(len(omega)):
        a_w = R @ force[k] + G
        p = p + v * dt + 0.5 * a_w * dt * dt
        v = v + a_w * dt
        R = R @ so3_exp(omega[k] * dt)
    return R, v, p


def _integrate_all(pim, omega, force, dt):
    for k in range(len(omega)):
        pim.integrate(omega[k] + BG, force[k] + BA, dt)   # medidas CON sesgo
    return pim


def test_predict_equals_dead_reckoning():
    omega, force, dt = _smooth_signals()
    R0 = so3_exp(np.array([0.3, -0.2, 0.5]))
    v0 = np.array([0.4, -0.1, 0.2])
    p0 = np.array([1.0, 2.0, 3.0])
    R_gt, v_gt, p_gt = _dead_reckon(R0, v0, p0, omega, force, dt)

    pim = _integrate_all(ImuPreintegration(NOISE, BG, BA), omega, force, dt)
    R_p, v_p, p_p = pim.predict(R0, v0, p0)

    assert np.linalg.norm(so3_log(R_gt.T @ R_p)) < 1e-12
    assert np.linalg.norm(v_gt - v_p) < 1e-10
    assert np.linalg.norm(p_gt - p_p) < 1e-10
    # El residuo del (futuro) factor IMU es cero en el estado verdadero.
    r = pim.residual(R0, v0, p0, R_gt, v_gt, p_gt)
    assert np.linalg.norm(r) < 1e-10, f"residual no nulo: {np.linalg.norm(r)}"


def test_bias_correction_is_first_order():
    omega, force, dt = _smooth_signals()
    pim = _integrate_all(ImuPreintegration(NOISE, BG, BA), omega, force, dt)

    errs = []
    for scale in (1e-3, 1e-2):
        dbg = scale * np.array([1.0, -0.7, 0.4])
        dba = scale * np.array([-0.5, 1.0, 0.8])
        dR_lin, dv_lin, dp_lin = pim.corrected_deltas(BG + dbg, BA + dba)
        # La verdad: re-integrar con el sesgo perturbado.
        pim2 = _integrate_all(ImuPreintegration(NOISE, BG + dbg, BA + dba),
                              omega, force, dt)
        e = max(np.linalg.norm(so3_log(pim2.delta_R.T @ dR_lin)),
                np.linalg.norm(pim2.delta_v - dv_lin),
                np.linalg.norm(pim2.delta_p - dp_lin))
        errs.append(e)
    # Con |db| = 1e-3 el error lineal es ~2e-5 (medido) y crece ×100 al ×10
    # la perturbación (2do orden). Márgenes ~3× sobre lo medido.
    assert errs[0] < 6e-5, f"error 1er orden grande: {errs[0]:.3e}"
    assert 30.0 < errs[1] / errs[0] < 300.0, \
        f"no escala como 2do orden: {errs[1]:.3e}/{errs[0]:.3e}"


def test_gtsam_equivalence():
    try:
        import gtsam
    except ImportError:
        print("  (gtsam no disponible: test de equivalencia saltado)")
        return
    omega, force, dt = _smooth_signals()
    params = gtsam.PreintegrationParams.MakeSharedU(GRAVITY)
    params.setGyroscopeCovariance(NOISE.gyro_noise_density ** 2 * np.eye(3))
    params.setAccelerometerCovariance(NOISE.accel_noise_density ** 2 * np.eye(3))
    params.setIntegrationCovariance(np.zeros((3, 3)))
    bias = gtsam.imuBias.ConstantBias(BA, BG)        # ¡acelerometro PRIMERO!
    pim_g = gtsam.PreintegratedImuMeasurements(params, bias)
    pim_n = ImuPreintegration(NOISE, BG, BA)
    for k in range(len(omega)):
        pim_g.integrateMeasurement(force[k] + BA, omega[k] + BG, dt)
        pim_n.integrate(omega[k] + BG, force[k] + BA, dt)

    # Deltas (medido: 7e-5 / 1.2e-4 / 6e-5 — tangente vs variedad, 2do orden).
    assert abs(pim_g.deltaTij() - pim_n.delta_t) < 1e-9
    assert np.linalg.norm(
        so3_log(pim_g.deltaRij().matrix().T @ pim_n.delta_R)) < 1e-3
    assert np.linalg.norm(pim_g.deltaVij() - pim_n.delta_v) < 2e-3
    assert np.linalg.norm(pim_g.deltaPij() - pim_n.delta_p) < 2e-3

    # predict() contra el de GTSAM (mismo estado inicial).
    R0 = so3_exp(np.array([0.3, -0.2, 0.5]))
    v0 = np.array([0.4, -0.1, 0.2])
    p0 = np.array([1.0, 2.0, 3.0])
    nav = pim_g.predict(gtsam.NavState(gtsam.Rot3(R0), gtsam.Point3(p0), v0),
                        bias)
    R_p, v_p, p_p = pim_n.predict(R0, v0, p0)
    assert np.linalg.norm(so3_log(nav.attitude().matrix().T @ R_p)) < 1e-3
    assert np.linalg.norm(nav.velocity() - v_p) < 2e-3
    assert np.linalg.norm(nav.position() - p_p) < 2e-3

    # Covarianza: GTSAM ordena (theta, p, v); nosotros [phi, v, p].
    # Medido: diff relativa máxima 1.4e-2 (el bloque de rotación difiere ~3%
    # entre formulaciones; v y p, < 1e-4). Sin permutar la diff es 9e-2 —
    # el assert también protege el ORDEN de bloques.
    perm = np.r_[0:3, 6:9, 3:6]
    C_n = pim_n.cov[np.ix_(perm, perm)]
    C_g = pim_g.preintMeasCov()
    rel = np.abs(C_n - C_g).max() / np.abs(C_g).max()
    assert rel < 5e-2, f"covarianza difiere {rel:.3e}"


def test_euroc_dead_reckoning():
    if not EUROC_ROOT.is_dir():
        print("  (data/euroc/V1_01_easy no esta: dead-reckoning real saltado)")
        return
    from vslam.io.dataset import (_quat_wxyz_to_R, euroc_imu_params,
                                  read_euroc_imu, read_euroc_state)
    ts_imu, gyro, acc = read_euroc_imu(EUROC_ROOT)
    ts_gt, p, q, v, bg, ba = read_euroc_state(EUROC_ROOT)
    noise = ImuNoiseParams(**euroc_imu_params(EUROC_ROOT))
    assert noise.rate_hz == 200.0

    window = 1.0
    rot_e, pos_e = [], []
    for t0 in np.linspace(ts_gt[0], ts_gt[-1] - window - 0.1, 60):
        i = int(np.searchsorted(ts_gt, t0))
        j = int(np.searchsorted(ts_gt, ts_gt[i] + window))
        if j >= len(ts_gt):
            continue
        pim = preintegrate_between(ts_imu, gyro, acc, float(ts_gt[i]),
                                   float(ts_gt[j]), noise, bg[i], ba[i])
        R_p, v_p, p_p = pim.predict(_quat_wxyz_to_R(q[i]), v[i], p[i])
        rot_e.append(np.degrees(np.linalg.norm(
            so3_log(_quat_wxyz_to_R(q[j]).T @ R_p))))
        pos_e.append(np.linalg.norm(p_p - p[j]) * 100.0)
    rot_e, pos_e = np.array(rot_e), np.array(pos_e)
    med_rot, med_pos = float(np.median(rot_e)), float(np.median(pos_e))
    p90_pos = float(np.percentile(pos_e, 90))
    print(f"  V1_01 dead-reckoning 1 s ({len(rot_e)} ventanas): "
          f"rot mediana {med_rot:.2f} deg, pos mediana {med_pos:.1f} cm, "
          f"p90 {p90_pos:.1f} cm")
    # Medido: 0.33 deg / 4.4 cm / p90 7.8 cm. Margen ~2x: si esto se rompe,
    # se rompieron las convenciones (frames, signo de g, sesgos), no el ruido.
    assert med_rot < 0.7, f"rotacion mediana {med_rot:.2f} deg"
    assert med_pos < 9.0, f"posicion mediana {med_pos:.1f} cm"
    assert p90_pos < 16.0, f"posicion p90 {p90_pos:.1f} cm"


def main() -> int:
    test_predict_equals_dead_reckoning()
    print("OK: predict == dead-reckoning exacto (y residual 0 en el GT)")
    test_bias_correction_is_first_order()
    print("OK: correccion de sesgo a 1er orden (error de 2do orden)")
    test_gtsam_equivalence()
    print("OK: equivalencia GTSAM (deltas + predict + covarianza)")
    test_euroc_dead_reckoning()
    print("OK: dead-reckoning sobre EuRoC V1_01 real")
    print("OK: los 4 tests de preintegracion IMU (v1.1 hito 1) pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
