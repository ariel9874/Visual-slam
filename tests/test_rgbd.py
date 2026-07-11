#!/usr/bin/env python3
"""Tests del modo RGB-D (v0.6): alineación métrica, init instantánea y loader.

(1) Umeyama RÍGIDO (with_scale=False): la métrica de un sistema métrico no
    regala la escala al alineador — si el estimado está a otra escala, el ATE
    rígido DEBE verlo (y el de similitud, ocultarlo). Se verifica ambos.
(2) INIT RGB-D: con un extractor inyectado y una profundidad sintética, el
    tracker debe inicializar en el PRIMER frame (sin danza de dos vistas) y el
    mapa debe ser la retro-proyección métrica exacta.
(3) Loader: fixture TUM en miniatura (PNG uint16, factor 5000) → metros y
    asociación rgb↔depth por timestamp.
(4) Bucle MÉTRICO = SE(3) (lección 35): un cierre de bucle sobre un mapa
    métrico NO puede re-escalarlo. Se fabrica un bucle con puntos duplicados
    a escala 1.2 (el cebo): la rama monocular DEBE medir ese s_rel y
    re-escalar (Strasdat, v0.4); la métrica DEBE ignorarlo — la escala del
    sensor no se negocia. Medido en fr2_xyz (77 bucles): Sim(3) compone el
    error hasta escala 2.09 / ATE 22 cm; SE(3) lo evita.
(5) Residuo de profundidad en el BA (v0.6 hito 2): con UNA cámara fija el
    gauge monocular deja la escala en el espacio nulo (bundle_adjustment.py)
    — el BA 2D DEBE dejar un mapa mal escalado donde está; el MISMO BA con
    u_R = u − bf/z DEBE devolverlo a la escala métrica. Es el par de tests
    (nulo/observable) que discrimina el mecanismo, no solo el resultado.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera
from vslam.evaluation import ate, umeyama_alignment
from vslam.frontend.tracker import PnPTracker
from vslam.io.dataset import TUMRGBDLoader

CAM = PinholeCamera(fx=450.0, fy=450.0, cx=320.0, cy=240.0, width=640, height=480)


def test_rigid_umeyama_detects_scale():
    rng = np.random.default_rng(0)
    gt = rng.uniform(-2, 2, (200, 3))
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    est_rigid = (R @ gt.T).T + np.array([0.3, -0.1, 0.5])   # rígido puro
    est_scaled = 1.15 * est_rigid                            # +15% de escala

    # Rígido puro: ambos alineadores lo clavan.
    assert ate(est_rigid, gt, with_scale=False)["rmse"] < 1e-9
    # Con escala: la similitud la ABSORBE (rmse ~0 — mentira piadosa)...
    assert ate(est_scaled, gt)["rmse"] < 1e-9
    # ...pero el rígido la DELATA (la vara métrica honesta).
    assert ate(est_scaled, gt, with_scale=False)["rmse"] > 0.05
    s, _, _ = umeyama_alignment(est_scaled, gt, with_scale=False)
    assert s == 1.0


class _StubExtractor:
    """Devuelve keypoints fijos en una rejilla (descriptores aleatorios)."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.kps = [cv2.KeyPoint(float(u), float(v), 8.0)
                    for u in range(40, 640, 40) for v in range(40, 480, 40)]
        self.desc = rng.integers(0, 256, (len(self.kps), 32), dtype=np.uint8)

    def detect_and_compute(self, gray):
        return self.kps, self.desc


def test_rgbd_instant_metric_init():
    # Profundidad sintética: plano inclinado entre 1 y 3 m (todo en rango).
    v, u = np.mgrid[0:480, 0:640]
    depth = (1.0 + 2.0 * u / 640.0).astype(np.float32)
    ext = _StubExtractor()
    tracker = PnPTracker(CAM, extractor=ext)
    gray = np.zeros((480, 640), np.uint8)

    T, info = tracker.process_frame(gray, depth)
    assert info["state"] == "INIT-OK", f"no inicializó: {info['state']}"
    assert tracker._initialized and np.allclose(T, np.eye(4))

    # El mapa debe ser la retro-proyección métrica EXACTA de los keypoints.
    ids, pts, _ = tracker.mapper.snapshot()
    assert len(pts) == len(ext.kps)
    for kp, X in zip(ext.kps, pts):
        z = depth[int(kp.pt[1]), int(kp.pt[0])]
        expected = np.array([(kp.pt[0] - 320.0) / 450.0 * z,
                             (kp.pt[1] - 240.0) / 450.0 * z, z])
        assert np.allclose(X, expected, atol=1e-9), (kp.pt, X, expected)


class _IdentityMatcher:
    """Match 1:1 por índice: el frame actual ES el keyframe antiguo."""

    def match(self, qdesc, tdesc, qkps, tkps, shape):
        n = min(len(qdesc), len(tdesc))
        return [cv2.DMatch(i, i, 0.0) for i in range(n)]


def _make_loop_scenario():
    """Tracker RGB-D inicializado + bucle fabricado contra el KF 0.

    El frame 'actual' (KF 100) ve los MISMOS keypoints que el KF 0, pero su
    mapa local son puntos DUPLICADOS a 1.2× (el cebo de escala): la nube
    compartida del bucle mide s_rel = 1/1.2. La pose PnP del bucle es la
    identidad, así que la ÚNICA corrección posible es la de escala — el
    discriminador perfecto entre el grafo Sim(3) y el SE(3).
    """
    # Rejilla densa: LOOP_MIN_MATCHES=200 exige >200 correspondencias.
    ext = _StubExtractor()
    ext.kps = [cv2.KeyPoint(float(u), float(v), 8.0)
               for u in range(30, 640, 30) for v in range(30, 480, 30)]
    rng = np.random.default_rng(1)
    ext.desc = rng.integers(0, 256, (len(ext.kps), 32), dtype=np.uint8)

    v, u = np.mgrid[0:480, 0:640]
    depth = (1.0 + 2.0 * u / 640.0).astype(np.float32)
    tracker = PnPTracker(CAM, extractor=ext, matcher=_IdentityMatcher(),
                         loop_closure=True)
    gray = np.zeros((480, 640), np.uint8)
    _, info = tracker.process_frame(gray, depth)
    assert info["state"] == "INIT-OK" and tracker._metric

    ids, pts, _ = tracker.mapper.snapshot()
    px = np.float64([kp.pt for kp in ext.kps])

    # KF 100 (cumple LOOP_TEMPORAL_GAP=60) con duplicados a 1.2×.
    from vslam.core.frame import Frame
    tracker.mapper.integrate_keyframe(Frame(frame_id=100, timestamp=0.0,
                                            T_w_c=np.eye(4), is_keyframe=True))
    dup_ids = tracker.mapper.add_points(1.2 * pts, ext.desc, anchor_kf_id=100)
    tracker.mapper.add_observations(100, dup_ids, px)
    cur_mp = dict(enumerate(dup_ids))
    cur = {"id": 100, "kps": ext.kps, "desc": ext.desc, "mp": cur_mp,
           "T": np.eye(4)}
    tracker._kf = cur
    tracker._kf_ids = [tracker._kf_ids[0], 100]
    tracker._kf_db = [tracker._kf_db[0], cur]
    return tracker, ext, gray, dup_ids, pts


def test_metric_loop_is_rigid():
    # (a) MÉTRICO: el bucle debe cerrarse SIN tocar la escala del mapa.
    tracker, ext, gray, dup_ids, pts = _make_loop_scenario()
    info = {"state": ""}
    tracker._try_close_loop(gray, ext.kps, ext.desc, info)
    assert info["state"] == "TRACK+KF+LOOP", f"no cerró: {info['state']}"
    after = tracker.mapper.point_positions(dup_ids)
    dup_after = np.array([after[i] for i in dup_ids])
    assert np.allclose(dup_after, 1.2 * pts, atol=1e-6), \
        "el bucle métrico re-escaló el mapa (debe ser rígido)"
    R = tracker.mapper.keyframe_pose(100)[:3, :3]
    assert abs(np.linalg.det(R) - 1.0) < 1e-9

    # (b) El CEBO funciona: la rama monocular con el MISMO escenario re-escala
    # (es su deber: ahí la escala es gauge y el bucle la mide). Si esto dejara
    # de re-escalar, el test ya no discriminaría nada.
    tracker, ext, gray, dup_ids, pts = _make_loop_scenario()
    tracker._metric = False
    tracker._try_close_loop(gray, ext.kps, ext.desc, {"state": ""})
    after = tracker.mapper.point_positions(dup_ids)
    dup_after = np.array([after[i] for i in dup_ids])
    ratio = float(np.linalg.norm(dup_after, axis=1).mean()
                  / np.linalg.norm(1.2 * pts, axis=1).mean())
    assert ratio < 0.95, f"el cebo no re-escaló en monocular (ratio {ratio:.3f})"


def test_ba_depth_residual_makes_scale_observable():
    """El estéreo virtual (v0.6 hito 2) hace OBSERVABLE la escala en el BA.

    Escena de dos cámaras con UNA sola fija, inicializada con la familia de
    gauge X' = 1.15·X, C' = 1.15·C (residuo 2D EXACTAMENTE cero — el BA
    monocular no tiene gradiente hacia la escala verdadera). Las u_R medidas
    con la profundidad VERDADERA rompen la familia: solo el BA con stereo_bf
    debe recuperar escala 1 y el baseline métrico de la cámara libre.
    """
    from vslam.backend.bundle_adjustment import local_bundle_adjustment

    rng = np.random.default_rng(3)
    n = 60
    pts_true = np.column_stack([rng.uniform(-0.5, 0.5, n),
                                rng.uniform(-0.4, 0.4, n),
                                rng.uniform(1.2, 3.0, n)])
    centers = {0: np.zeros(3), 1: np.array([0.25, 0.0, 0.0])}
    bf = 40.0

    obs = []
    for k, c in centers.items():
        rel = pts_true - c                       # rotaciones identidad
        u = 450.0 * rel[:, 0] / rel[:, 2] + 320.0
        v = 450.0 * rel[:, 1] / rel[:, 2] + 240.0
        u_r = u - bf / rel[:, 2]
        obs += [(k, p, np.array([u[p], v[p], u_r[p]])) for p in range(n)]

    def scaled_init(s):
        poses = {k: np.eye(4) for k in centers}
        for k, c in centers.items():
            poses[k][:3, 3] = s * c
        return poses, {p: s * pts_true[p] for p in range(n)}

    def median_scale(opt):
        return float(np.median([np.linalg.norm(opt[p])
                                / np.linalg.norm(pts_true[p])
                                for p in range(n)]))

    # (a) Monocular (bf=0): la escala es espacio nulo — el 15% DEBE quedarse.
    poses, pts = scaled_init(1.15)
    _, opt = local_bundle_adjustment(CAM, poses, pts, obs, fixed_kfs={0},
                                     iterations=15)
    ratio = median_scale(opt)
    assert ratio > 1.10, f"el gauge se corrigió sin medición ({ratio:.3f})"

    # (b) El MISMO problema con el residuo de profundidad: escala recuperada.
    poses, pts = scaled_init(1.15)
    opt_poses, opt = local_bundle_adjustment(CAM, poses, pts, obs,
                                             fixed_kfs={0}, iterations=15,
                                             stereo_bf=bf)
    ratio = median_scale(opt)
    assert abs(ratio - 1.0) < 0.02, f"no recuperó la escala métrica ({ratio:.3f})"
    assert np.allclose(opt_poses[1][:3, 3], centers[1], atol=0.02), \
        "el baseline de la cámara libre no volvió a su valor métrico"


def test_depth_loader_fixture():
    root = Path(tempfile.mkdtemp()) / "seq"
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    gray = np.full((480, 640), 100, np.uint8)
    dep = np.full((480, 640), 10000, np.uint16)      # 10000/5000 = 2.0 m
    dep[0, 0] = 0                                    # sin dato
    rgb_lines, dep_lines = ["# ts fn"], ["# ts fn"]
    for i, ts in enumerate([1.00, 1.05]):
        cv2.imwrite(str(root / "rgb" / f"{i}.png"), gray)
        rgb_lines.append(f"{ts:.6f} rgb/{i}.png")
        cv2.imwrite(str(root / "depth" / f"{i}.png"), dep)
        dep_lines.append(f"{ts + 0.01:.6f} depth/{i}.png")   # desfase realista
    (root / "rgb.txt").write_text("\n".join(rgb_lines), encoding="utf-8")
    (root / "depth.txt").write_text("\n".join(dep_lines), encoding="utf-8")

    items = list(TUMRGBDLoader(root, with_depth=True))
    assert len(items) == 2
    ts, gray_out, depth = items[0]
    assert depth is not None and depth.dtype == np.float32
    assert abs(float(depth[240, 320]) - 2.0) < 1e-6, "factor 5000 mal aplicado"
    assert float(depth[0, 0]) == 0.0, "el 0 (sin dato) debe conservarse"


def main() -> int:
    test_rigid_umeyama_detects_scale()
    test_rgbd_instant_metric_init()
    test_metric_loop_is_rigid()
    test_ba_depth_residual_makes_scale_observable()
    test_depth_loader_fixture()
    print("OK: los 5 tests de RGB-D (v0.6) pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
