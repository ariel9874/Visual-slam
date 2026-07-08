"""Bundle Adjustment con GTSAM — la ruta de RENDIMIENTO del BA (v0.5).

El perfilado (docs/05 §v0.5) señaló al BA local como el 57% del tiempo del
tracker: `local_bundle_adjustment` (referencia NumPy didáctica) tarda ~2 s por
keyframe. Este adaptador resuelve EL MISMO problema con GTSAM, cuyo motor de
factores en C++ (eliminación dispersa + LM) lo hace en órdenes de magnitud
menos. La referencia NumPy queda como implementación educativa y de respaldo
(y como oráculo del test de equivalencia): MISMA interfaz, mismos números.

─── El mapeo repo ↔ GTSAM ─────────────────────────────────────────────────────
- Pose `T_w_c` (cámara→mundo, ejes OpenCV) ↔ `gtsam.Pose3` (pose de la cámara en
  el mundo): coinciden — el `GenericProjectionFactor` proyecta un punto del
  MUNDO por la cámara en esa pose (usa T_w_c⁻¹ internamente).
- Intrínsecos ↔ `gtsam.Cal3_S2(fx, fy, 0, cx, cy)` (sin skew).
- Ruido de píxel σ=1 + kernel de Huber (mismo `huber_px` que la referencia; el
  umbral de Huber de GTSAM está en unidades del error blanqueado = píxeles con
  σ=1, así que 2.5 ≡ 2.5 px, igual que la referencia).
- GAUGE: la referencia FIJA 2 keyframes (7 gdl monocular, ver bundle_adjustment).
  GTSAM optimiza todas las variables → se ancla con un `PriorFactorPose3` de
  ruido diminuto (σ=1e-6) sobre esos KFs: fijado a efectos prácticos.
- Puntos con < 2 observaciones se EXCLUYEN (sub-determinados: se deslizan por su
  rayo — la misma regla de la referencia y del tracker).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np

from vslam.core.camera import PinholeCamera

Observation = Tuple[int, int, np.ndarray]   # (kf_id, point_id, píxel (2,))

_INSTALL_MSG = ("El backend GTSAM requiere `gtsam` (conda-forge win-64/linux):\n"
                "    conda install -c conda-forge gtsam   (exige numpy<2)")


def gtsam_bundle_adjustment(
    camera: PinholeCamera,
    kf_poses: Dict[int, np.ndarray],
    points: Dict[int, np.ndarray],
    observations: List[Observation],
    fixed_kfs: Set[int],
    iterations: int = 8,
    huber_px: float = 2.5,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """BA con GTSAM. Firma IDÉNTICA a `local_bundle_adjustment` (intercambiables).

    Devuelve ({kf_id: T_w_c}, {point_id: posición}); las variables sin
    observaciones (o puntos con < 2 obs) salen sin tocar, como en la referencia.
    """
    try:
        import gtsam
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(_INSTALL_MSG) from exc
    from gtsam import symbol

    poses = {k: np.asarray(T, float) for k, T in kf_poses.items()}
    pts = {p: np.asarray(x, float) for p, x in points.items()}
    obs = [(k, p, np.asarray(uv, float)) for k, p, uv in observations
           if k in poses and p in pts]
    # Solo puntos con ≥ 2 observaciones (los demás quedan sub-determinados).
    counts = Counter(p for _, p, _ in obs)
    used_pts = {p for p, c in counts.items() if c >= 2}
    obs = [(k, p, uv) for k, p, uv in obs if p in used_pts]
    if not obs or not used_pts:
        return poses, pts
    used_kfs = {k for k, _, _ in obs}

    X = lambda i: symbol("x", i)                     # noqa: E731
    L = lambda j: symbol("l", j)                     # noqa: E731
    K = gtsam.Cal3_S2(camera.fx, camera.fy, 0.0, camera.cx, camera.cy)
    px_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    robust = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(huber_px), px_noise)

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    for k in used_kfs:
        initial.insert(X(k), gtsam.Pose3(poses[k]))
    for p in used_pts:
        x = pts[p]
        initial.insert(L(p), gtsam.Point3(float(x[0]), float(x[1]), float(x[2])))
    for k, p, uv in obs:
        graph.add(gtsam.GenericProjectionFactorCal3_S2(uv, robust, X(k), L(p), K))
    # Anclar el gauge: prior fuerte sobre los KFs fijos (≈ fijarlos).
    prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)
    for k in fixed_kfs & used_kfs:
        graph.add(gtsam.PriorFactorPose3(X(k), gtsam.Pose3(poses[k]), prior_noise))

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(iterations)
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()

    opt_poses = dict(poses)
    opt_points = dict(pts)
    for k in used_kfs:
        opt_poses[k] = result.atPose3(X(k)).matrix()
    for p in used_pts:
        opt_points[p] = np.asarray(result.atPoint3(L(p)), float)
    return opt_poses, opt_points
