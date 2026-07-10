#!/usr/bin/env python3
"""Tests del BA incremental iSAM2 (v0.5, vslam/backend/gtsam_isam2.py).

Valida el contrato del backend sobre un SparsePointMapper REAL alimentado
incrementalmente (el patrón exacto del tracker):
(1) con observaciones exactas y arranque perturbado, las poses de la ventana
    vuelven a la verdad;
(2) tras un RESET (cierre de bucle) el backend sobrevive: re-siembra poses y
    puntos antiguos con priors y sigue optimizando sin excepciones.
Los puntos entran siempre con sus 2 observaciones (lección del probe: con una
sola, iSAM2 lanza IndeterminantLinearSystem). Se salta limpio sin gtsam.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera
from vslam.core.frame import Frame
from vslam.core.geometry import invert_se3
from vslam.mapping.sparse import SparsePointMapper


def _has_gtsam() -> bool:
    try:
        import gtsam  # noqa: F401
        return True
    except ImportError:
        return False


CAM = PinholeCamera(fx=450.0, fy=450.0, cx=320.0, cy=240.0, width=640, height=480)


def _pose(k):
    T = np.eye(4)
    T[:3, 3] = [0.25 * k, 0.0, 0.02 * k]
    return T


def _project(T, X):
    Tcw = invert_se3(T)
    Xc = Tcw[:3, :3] @ X + Tcw[:3, 3]
    if Xc[2] < 0.2:
        return None
    uv = CAM.project(Xc[None, :])[0]
    if not (0 <= uv[0] < 640 and 0 <= uv[1] < 480):
        return None
    return uv


class _Feeder:
    """Reproduce el patrón del tracker: KFs que crean puntos (con 2 obs) y
    re-observan los de KFs previos; cursores como en _run_local_ba."""

    def __init__(self, seed=0, pose_noise=0.02, point_noise=0.03):
        from vslam.backend.gtsam_isam2 import ISAM2LocalBA
        self.rng = np.random.default_rng(seed)
        self.mapper = SparsePointMapper()
        self.backend = ISAM2LocalBA(CAM)
        self.cursor = {}
        self.kfs = []
        self.pts_by_kf = {}
        self.gt_points = {}
        self.pose_noise = pose_noise
        self.point_noise = point_noise

    def add_kf(self, k, window_size=5):
        T_true = _pose(k)
        T_init = T_true.copy()
        if k >= 2:                                    # los 2 primeros anclan
            T_init[:3, 3] += self.rng.normal(0, self.pose_noise, 3)
        self.mapper.integrate_keyframe(Frame(frame_id=k, timestamp=0.0,
                                             T_w_c=T_init, is_keyframe=True))
        self.kfs.append(k)
        new = []
        if k >= 1:                                    # puntos nacen con 2 obs
            xs, uvs, uvs_prev = [], [], []
            for _ in range(40):
                X = np.array([0.25 * k + self.rng.uniform(-2, 2),
                              self.rng.uniform(-1.5, 1.5),
                              0.02 * k + self.rng.uniform(4, 9)])
                uv, uv_p = _project(_pose(k), X), _project(_pose(k - 1), X)
                if uv is None or uv_p is None:
                    continue
                xs.append(X + self.rng.normal(0, self.point_noise, 3))
                uvs.append(uv)
                uvs_prev.append(uv_p)
            if xs:
                ids = self.mapper.add_points(np.array(xs),
                                             np.zeros((len(xs), 32), np.uint8),
                                             anchor_kf_id=k)
                self.mapper.add_observations(k, ids, np.array(uvs))
                self.mapper.add_observations(k - 1, ids, np.array(uvs_prev))
                new = list(zip(ids, xs))
        self.pts_by_kf[k] = new
        # re-observaciones de los 3 KFs previos
        for kk in range(max(0, k - 3), k):
            pids, uvs = [], []
            for pid, _ in self.pts_by_kf.get(kk, []):
                X = self.mapper.point_positions([pid])[pid]
                uv = _project(_pose(k), X)
                if uv is not None:
                    pids.append(pid)
                    uvs.append(uv)
            if pids:
                self.mapper.add_observations(k, pids, np.array(uvs))
        # alimentar al backend (cursores, como el tracker)
        new_obs = []
        for kf, entries in self.mapper._obs.items():
            start = self.cursor.get(kf, 0)
            if start < len(entries):
                new_obs.extend((kf, pid, uv) for pid, uv in entries[start:])
                self.cursor[kf] = len(entries)
        window = self.kfs[-window_size:]
        result = self.backend.process_keyframe(self.mapper, window, new_obs)
        if result is not None:
            opt_poses, opt_points = result
            for kk, T in opt_poses.items():
                self.mapper.set_keyframe_pose(kk, T)
            self.mapper.set_point_positions(opt_points)
        return result


def test_incremental_recovers_poses():
    f = _Feeder(seed=0)
    for k in range(12):
        f.add_kf(k)
    assert f.backend.n_failures == 0
    for k in f.kfs[-5:]:                              # la ventana refinada
        err = np.linalg.norm(f.mapper.keyframe_pose(k)[:3, 3] - _pose(k)[:3, 3])
        assert err < 0.01, f"KF {k}: {err:.4f} (el BA no recuperó la pose)"


def test_reset_and_reseed_survive():
    f = _Feeder(seed=1)
    for k in range(8):
        f.add_kf(k)
    f.backend.reset()                                 # "cierre de bucle"
    for k in range(8, 14):                            # re-observa puntos viejos
        f.add_kf(k)
    assert f.backend.n_failures == 0, "el reset/re-siembra provocó fallos"
    for k in f.kfs[-3:]:
        err = np.linalg.norm(f.mapper.keyframe_pose(k)[:3, 3] - _pose(k)[:3, 3])
        assert err < 0.02, f"KF {k} tras reset: {err:.4f}"


def main() -> int:
    if not _has_gtsam():
        print("SKIP: gtsam no instalado.")
        return 0
    test_incremental_recovers_poses()
    test_reset_and_reseed_survive()
    print("OK: los 2 tests del BA incremental iSAM2 pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
