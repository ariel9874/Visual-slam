"""Bundle Adjustment local: refina poses de keyframes Y puntos 3D a la vez.

Es el refinador de oro de todo el SLAM geométrico: el PnP del tracker estima
cada pose contra un mapa que da por bueno; el BA admite que TODO tiene ruido
(poses y puntos) y busca la configuración conjunta que mejor explica todas
las observaciones. Sin él, el ruido de triangulación se acumula keyframe a
keyframe (lo medimos en v0.2: ORB quedaba en ~7 cm por esto).

─── La matemática: el problema ───────────────────────────────────────────────
    argmin_{T_k, X_p}  Σ_(k,p)  ρ( ‖ π(K, T_k⁻¹·X_p) − u_kp ‖² )

Observación (k, p): el keyframe k vio el punto p en el píxel u_kp. π es la
proyección pinhole y ρ el kernel de Huber (outliers empujan linealmente).

─── La matemática: jacobianos analíticos ─────────────────────────────────────
El residuo r = π(X_c) − u depende de la pose y del punto vía X_c = T_c_w·X_w.
Con la perturbación por la derecha del repo, T_w_c ← T_w_c·Exp(δ), se tiene
T_c_w ← Exp(−δ)·T_c_w, y a primer orden X_c(δ) ≈ X_c − ρ − ω×X_c, es decir:

    ∂X_c/∂δ  = [ −I₃ | [X_c]_× ]              (3×6, orden [ρ, ω] del repo)
    ∂X_c/∂X_w = R_c_w                          (3×3)

    ∂π/∂X_c  = [[ fx/z,    0,  −fx·x/z² ],     (2×3; derivar u = fx·x/z + cx)
                [    0, fy/z,  −fy·y/z² ]]

y por regla de la cadena:  J_pose = ∂π·∂X_c/∂δ  (2×6),  J_punto = ∂π·R_c_w (2×3).
(A diferencia del grafo de poses, aquí los jacobianos analíticos compensan:
hay miles de observaciones y la estructura es la que habilita el Schur.)

─── La matemática: complemento de Schur ──────────────────────────────────────
Las ecuaciones normales tienen estructura de flecha:

    H·δ = g,   H = [[B, E], [Eᵀ, C]],  δ = [δ_cámaras, δ_puntos]

B (6K×6K) acopla cámaras entre sí; C es DIAGONAL POR BLOQUES 3×3 porque dos
puntos nunca comparten un residuo — los puntos solo se hablan a través de las
cámaras. Eso permite eliminar (marginalizar) los puntos por casi nada:

    (B − E·C⁻¹·Eᵀ)·δ_c = g_c − E·C⁻¹·g_p       ← sistema reducido, ¡6K×6K!
    δ_p = C_p⁻¹·(g_p − E_pᵀ·δ_c)               ← retro-sustitución por punto

Con K=5 keyframes el sistema grande (~miles de variables) colapsa a uno de
30×30. Este truco ES la razón de que el BA escale; g2o/Ceres/GTSAM viven de
él (y del mismo Schur nace la "marginalización" de la ventana de DSO).

─── La matemática: RGB-D como estéreo virtual (v0.6) ────────────────────────
El sensor mide z en cada píxel, pero meter z directo al costo mezcla unidades
(metros vs píxeles) y exige un σ_z aparte. El truco de ORB-SLAM2: convertir la
profundidad en la coordenada que MEDIRÍA una cámara derecha a baseline b:

    u_R = u − fx·b/z          (fx·b ≡ bf; disparidad d = fx·b/z)

y extender el residuo de [u, v] a [u, v, u_R]. Todo queda en píxeles (misma
Huber, mismo Schur — solo crecen las filas de los jacobianos) y el peso de la
profundidad decae solo con la distancia: ∂u_R/∂z = fx·b/z², exactamente el
inverso del ruido del sensor (que crece ~z² en el Kinect) — la física y la
geometría se cancelan en la dirección correcta. El residuo extra ancla la
ESTRUCTURA del mapa a la medición métrica en cada observación: sin él, el BA
solo re-teje reproyecciones y la deriva métrica de fr1_desk (error REPARTIDO,
p50 7.8 cm, no un episodio) no tiene de dónde corregirse. Una observación (3,)
con u_R = NaN significa "este píxel no tenía z válida": residuo 2D normal.

─── La matemática: el gauge monocular tiene 7 grados, no 6 ──────────────────
Fijar UNA cámara ancla la rotación y traslación globales (6 gdl), pero en
monocular queda un séptimo: la ESCALA. La familia  X′ = s·X,  C′_k = s·C_k
(escalar la escena alrededor de la cámara fija) deja todos los residuos
intactos → H tiene un espacio nulo de dimensión 1 y el optimizador se para
en cualquier miembro de la familia (lo descubrimos empíricamente: el error
relativo residual era idéntico en poses y puntos — la firma de un offset de
escala). Solución estándar: fijar ≥ 2 cámaras con baseline entre ellas
(ORB-SLAM fija todas las cámaras fuera de la ventana local por esta razón).
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

from vslam.core.camera import PinholeCamera
from vslam.core.geometry import invert_se3
from vslam.core.lie import hat, se3_exp

Observation = Tuple[int, int, np.ndarray]   # (kf_id, point_id, píxel (2,) o
                                             #  (3,) = [u, v, u_R] RGB-D v0.6)


def _residual_and_jacobians(camera, T_c_w, X_w, uv, bf=0.0):
    """Residuo y jacobianos de UNA observación: 2D monocular o 3D con la
    coordenada derecha virtual u_R (RGB-D, teoría arriba; bf = fx·b).
    Devuelve None si el punto cae detrás de la cámara (sin proyección válida)."""
    X_c = T_c_w[:3, :3] @ X_w + T_c_w[:3, 3]
    x, y, z = X_c
    if z < 1e-3:
        return None
    r = [camera.fx * x / z + camera.cx - uv[0],
         camera.fy * y / z + camera.cy - uv[1]]
    d_pi = [[camera.fx / z, 0.0, -camera.fx * x / z ** 2],
            [0.0, camera.fy / z, -camera.fy * y / z ** 2]]
    if bf > 0.0 and len(uv) == 3 and np.isfinite(uv[2]):
        # Fila estéreo virtual: u_R = fx·x/z + cx − bf/z; su derivada respecto
        # de z es la de u MÁS bf/z² (el término que hace pesar la profundidad).
        r.append(camera.fx * x / z + camera.cx - bf / z - uv[2])
        d_pi.append([camera.fx / z, 0.0, (bf - camera.fx * x) / z ** 2])
    r, d_pi = np.asarray(r), np.asarray(d_pi)
    J_pose = d_pi @ np.hstack([-np.eye(3), hat(X_c)])   # ∂X_c/∂δ = [−I | [X_c]ₓ]
    J_point = d_pi @ T_c_w[:3, :3]                      # ∂X_c/∂X_w = R_c_w
    return r, J_pose, J_point


def local_bundle_adjustment(
    camera: PinholeCamera,
    kf_poses: Dict[int, np.ndarray],
    points: Dict[int, np.ndarray],
    observations: List[Observation],
    fixed_kfs: Set[int],
    iterations: int = 8,
    huber_px: float = 2.5,
    stereo_bf: float = 0.0,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Refina poses y puntos minimizando reproyección (LM + Schur, teoría arriba).

    Args:
        kf_poses: {kf_id: T_w_c} — TODAS las poses de la ventana (las fijas
            no se optimizan pero sus observaciones SÍ restringen los puntos).
        points: {point_id: (3,)} posiciones iniciales.
        observations: (kf_id, point_id, píxel). Se ignoran las que refieran
            a poses/puntos fuera de los diccionarios.
        fixed_kfs: ids anclados. En monocular deben ser ≥ 2 con baseline
            (ver el bloque de gauge en el docstring del módulo): con solo 1,
            la escala queda libre y la solución deriva dentro de esa familia.
        stereo_bf: fx·b de la cámara derecha virtual (RGB-D/estéreo, teoría
            arriba). 0 = apagado; con bf > 0 las observaciones (3,) con u_R
            finita aportan el residuo de profundidad. Debe ser el MISMO bf
            con el que el tracker fabricó las u_R.
    Returns:
        ({kf_id: T_w_c optimizada}, {point_id: posición optimizada})
    """
    poses = {k: np.asarray(T, float).copy() for k, T in kf_poses.items()}
    pts = {p: np.asarray(x, float).copy() for p, x in points.items()}
    obs = [(k, p, np.asarray(uv, float)) for k, p, uv in observations
           if k in poses and p in pts]
    free_cams = sorted(k for k in poses if k not in fixed_kfs)
    cam_idx = {k: 6 * n for n, k in enumerate(free_cams)}
    pt_list = sorted(pts)
    if not obs or not pt_list:
        return poses, pts

    # AGUJERO DE COSTO (bug real encontrado verificando este módulo, en dos
    # actos): (1) si las observaciones de un punto detrás de la cámara se
    # OMITEN del costo, el optimizador aprende la trampa — empujar un punto
    # conflictivo detrás de las cámaras borra sus residuos y "baja" el costo
    # (medimos puntos volando a 15000 unidades). (2) Con una penalización
    # tímida (100 px-equivalentes) la trampa persiste refinada: un outlier de
    # 113 px cuesta 280 > 247, y sale rentable esconder el punto detrás de LA
    # cámara del outlier. Moraleja: la penalización debe superar el residuo
    # físicamente posible más grande — el orden de la diagonal de la imagen.
    # (GTSAM/Ceres hacen lo análogo con evaluación "safe" de la proyección.)
    BEHIND_PENALTY = huber_px * (2000.0 - 0.5 * huber_px)

    def total_cost(poses_, pts_) -> float:
        c = 0.0
        for k, p, uv in obs:
            out = _residual_and_jacobians(camera, invert_se3(poses_[k]), pts_[p],
                                          uv, stereo_bf)
            if out is None:
                c += BEHIND_PENALTY
                continue
            e = np.linalg.norm(out[0])
            # Costo de Huber: cuadrático hasta δ, lineal después.
            c += 0.5 * e ** 2 if e <= huber_px else huber_px * (e - 0.5 * huber_px)
        return c

    lam = 1e-4
    cost = total_cost(poses, pts)
    n_c = 6 * len(free_cams)

    for _ in range(iterations):
        B = np.zeros((n_c, n_c))
        g_c = np.zeros(n_c)
        C: Dict[int, np.ndarray] = {p: np.zeros((3, 3)) for p in pt_list}
        g_p: Dict[int, np.ndarray] = {p: np.zeros(3) for p in pt_list}
        # Bloques E por punto: {point: {cam: 6×3}} (solo cámaras que lo ven).
        E: Dict[int, Dict[int, np.ndarray]] = {p: {} for p in pt_list}

        T_c_w_cache = {k: invert_se3(T) for k, T in poses.items()}
        for k, p, uv in obs:
            out = _residual_and_jacobians(camera, T_c_w_cache[k], pts[p], uv,
                                          stereo_bf)
            if out is None:
                continue
            r, J_pose, J_point = out
            # Peso IRLS de Huber sobre el residuo en píxeles.
            e = np.linalg.norm(r)
            w = 1.0 if e <= huber_px else huber_px / e

            C[p] += w * (J_point.T @ J_point)
            g_p[p] -= w * (J_point.T @ r)
            if k in cam_idx:
                i = cam_idx[k]
                B[i:i + 6, i:i + 6] += w * (J_pose.T @ J_pose)
                g_c[i:i + 6] -= w * (J_pose.T @ r)
                blk = E[p].setdefault(k, np.zeros((6, 3)))
                blk += w * (J_pose.T @ J_point)

        # Amortiguación LM sobre ambas diagonales.
        B[np.diag_indices(n_c)] += lam * np.diag(B) + 1e-9
        C_inv = {}
        for p in pt_list:
            Cp = C[p] + (lam * np.diag(np.diag(C[p])) + 1e-9 * np.eye(3))
            C_inv[p] = np.linalg.inv(Cp)

        # Schur: S = B − Σ_p E_p·C_p⁻¹·E_pᵀ  (solo pares de cámaras que
        # comparten el punto p — la covisibilidad ES la estructura de S).
        S = B.copy()
        rhs = g_c.copy()
        for p in pt_list:
            cams = list(E[p])
            for ka in cams:
                ia = cam_idx[ka]
                ECi = E[p][ka] @ C_inv[p]
                rhs[ia:ia + 6] -= ECi @ g_p[p]
                for kb in cams:
                    ib = cam_idx[kb]
                    S[ia:ia + 6, ib:ib + 6] -= ECi @ E[p][kb].T

        try:
            delta_c = np.linalg.solve(S, rhs) if n_c else np.zeros(0)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        # Retro-sustitución de los puntos y actualización de prueba.
        trial_poses = {k: T.copy() for k, T in poses.items()}
        for k, i in cam_idx.items():
            trial_poses[k] = trial_poses[k] @ se3_exp(delta_c[i:i + 6])
        trial_pts = {}
        for p in pt_list:
            acc = g_p[p].copy()
            for k, blk in E[p].items():
                i = cam_idx[k]
                acc -= blk.T @ delta_c[i:i + 6]
            trial_pts[p] = pts[p] + C_inv[p] @ acc

        trial_cost = total_cost(trial_poses, trial_pts)
        if trial_cost < cost:
            poses, pts, cost, lam = trial_poses, trial_pts, trial_cost, max(lam / 3, 1e-9)
        else:
            lam *= 5.0

    return poses, pts
