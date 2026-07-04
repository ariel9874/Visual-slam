"""Utilidades geométricas compartidas: SE(3), triangulación y PnP.

Son las primitivas sobre las que se construye el tracking 3D-2D (v0.2):
el tracker y el mapper las usan; los tests las verifican con geometría
sintética exacta (tests/test_triangulation_pnp.py).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from vslam.core.camera import PinholeCamera


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Inversa cerrada en SE(3):  T⁻¹ = [[Rᵀ, −Rᵀ·t], [0, 1]].

    (Derivación y notación de subíndices: examples/01_monocular_vo.py.)
    """
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _reprojection_errors(camera: PinholeCamera, T_w_c: np.ndarray,
                         points_w: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Error de reproyección (px) de cada punto 3D en una vista; inf si queda
    detrás de la cámara (la proyección pinhole no está definida para Z ≤ 0)."""
    T_c_w = invert_se3(T_w_c)
    pts_c = (T_c_w[:3, :3] @ points_w.T).T + T_c_w[:3, 3]
    err = np.full(len(points_w), np.inf)
    front = pts_c[:, 2] > 1e-6
    if front.any():
        uv = camera.project(pts_c[front])
        err[front] = np.linalg.norm(uv - pixels[front], axis=1)
    return err


def triangulate_two_views(
    camera: PinholeCamera,
    T_w_c0: np.ndarray,
    T_w_c1: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    reproj_thresh_px: float = 2.0,
    min_parallax_deg: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangula correspondencias entre dos vistas con poses conocidas.

    ─── La matemática: DLT + filtros de calidad ───
    Cada observación λ·x̂ = P·X̄ (con P = K·[R|t] = K·T_c_w[:3]) aporta, tras
    eliminar λ con el producto vectorial ([x̂]_×·P·X̄ = 0), dos ecuaciones
    lineales en X̄. Con las dos vistas se apila A·X̄ = 0 y la solución es el
    vector singular de menor valor singular (eso hace cv2.triangulatePoints).

    La solución algebraica hay que FILTRARLA, porque el sistema lineal acepta
    cualquier basura numérica:
      1. Quiralidad: profundidad positiva en AMBAS cámaras.
      2. Reproyección: el punto debe explicar sus dos observaciones (< umbral
         en píxeles) — el error algebraico de la DLT no es el geométrico.
      3. Paralaje: ángulo entre los rayos de observación
             θ = arccos( r̂₀ · r̂₁ ),   r_i = X − C_i
         Con rayos casi paralelos (baseline pequeño frente a la profundidad),
         la intersección está mal condicionada: un error de ε px en la imagen
         mueve el punto ~ profundidad²/(baseline·f) — puntos "en el infinito"
         que envenenarían el PnP posterior.

    Returns:
        (points_w, valid): puntos 3D (N, 3) en el frame del mundo y máscara
        booleana (N,) con los que pasan los tres filtros.
    """
    pts0 = np.ascontiguousarray(pts0, dtype=np.float64)
    pts1 = np.ascontiguousarray(pts1, dtype=np.float64)

    P0 = camera.K @ invert_se3(T_w_c0)[:3]
    P1 = camera.K @ invert_se3(T_w_c1)[:3]
    X_h = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)          # (4, N) homogéneo
    w = X_h[3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)                    # evita división por ~0
    points_w = (X_h[:3] / w).T

    # Filtro 1+2: quiralidad implícita (err = inf si Z ≤ 0) y reproyección.
    err0 = _reprojection_errors(camera, T_w_c0, points_w, pts0)
    err1 = _reprojection_errors(camera, T_w_c1, points_w, pts1)
    valid = (err0 < reproj_thresh_px) & (err1 < reproj_thresh_px)

    # Filtro 3: paralaje entre los rayos de observación.
    C0, C1 = T_w_c0[:3, 3], T_w_c1[:3, 3]
    r0 = points_w - C0
    r1 = points_w - C1
    cos = np.einsum("ij,ij->i", r0, r1) / (
        np.linalg.norm(r0, axis=1) * np.linalg.norm(r1, axis=1) + 1e-12)
    parallax = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    valid &= parallax > min_parallax_deg

    return points_w, valid


def solve_pnp(
    camera: PinholeCamera,
    points_w: np.ndarray,
    pixels: np.ndarray,
    reproj_thresh_px: float = 3.0,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Pose de la cámara desde correspondencias 3D-2D (PnP robusto + refinamiento).

    ─── La matemática ───
    PnP (Perspective-n-Point): dados puntos del mapa {X_i} y sus píxeles
    {u_i}, encontrar T que minimice el error de reproyección
        T* = argmin_T Σ ‖ π(K, T_c_w·X_i) − u_i ‖²
    Pipeline: RANSAC con EPnP como solver interno (expresa los X_i como
    combinación de 4 puntos de control → solución lineal O(n)) para separar
    inliers, y refinamiento final Levenberg-Marquardt SOLO con los inliers
    (el óptimo geométrico de verdad).

    OpenCV parametriza la rotación como rvec (eje-ángulo): R se recupera con
    la fórmula de Rodrigues  R = I + sin θ·[k]_× + (1 − cos θ)·[k]_×²  con
    θ = ‖rvec‖, k = rvec/θ. Y devuelve la pose MUNDO→CÁMARA (T_c_w): hay que
    invertirla para nuestra convención T_w_c.

    Returns:
        (T_w_c, inlier_mask); T_w_c es None si no hay solución fiable.
    """
    n = len(points_w)
    no_inliers = np.zeros(n, dtype=bool)
    if n < 6:
        return None, no_inliers

    obj = np.ascontiguousarray(points_w, dtype=np.float64)
    img = np.ascontiguousarray(pixels, dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img, camera.K, None,
        iterationsCount=200, reprojectionError=reproj_thresh_px,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 6:
        return None, no_inliers

    mask = no_inliers.copy()
    mask[inliers.ravel()] = True
    rvec, tvec = cv2.solvePnPRefineLM(obj[mask], img[mask], camera.K, None, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)
    T_c_w = np.eye(4)
    T_c_w[:3, :3] = R
    T_c_w[:3, 3] = tvec.ravel()
    return invert_se3(T_c_w), mask
