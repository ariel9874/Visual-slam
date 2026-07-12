"""Conversión de convenciones en la FRONTERA núcleo ↔ ROS (regla de oro:
nunca dentro del núcleo — ros2/README.md, docs/02 §6).

─── La matemática: cambio de convención de ejes ──────────────────────────────
El núcleo usa los ejes ÓPTICOS de OpenCV: +Z delante (eje óptico), +X derecha,
+Y abajo. ROS usa REP-103 para el cuerpo: +X delante, +Y izquierda, +Z arriba.
El cambio de base es la rotación fija

        R_bo = [[ 0,  0,  1],      x_body  =  z_opt   (delante)
                [-1,  0,  0],      y_body  = -x_opt   (izquierda)
                [ 0, -1,  0]]      z_body  = -y_opt   (arriba)

Una pose T_w_c (óptica en ambos lados: mundo óptico ← cámara óptica) se
convierte CONJUGANDO: hay que rotar el frame del mundo Y el del cuerpo,

        T_map_base = R̃_bo · T_w_c · R̃_bo⁻¹ ,   R̃_bo = [[R_bo, 0], [0, 1]]

(conjugar preserva la estructura de grupo: composiciones y deltas se convierten
igual — si solo se rotara un lado, los ejes del mundo y del cuerpo quedarían
inconsistentes y RViz mostraría la trayectoria "de lado").
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Pose, Transform

# Óptico (OpenCV) → cuerpo (REP-103), embebida en SE(3).
R_BO = np.array([[0.0, 0.0, 1.0],
                 [-1.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0]])
_T_BO = np.eye(4)
_T_BO[:3, :3] = R_BO
_T_OB = _T_BO.T                          # inversa (rotación pura)


def optical_to_rep103(T_w_c: np.ndarray) -> np.ndarray:
    """Pose óptica (núcleo) → REP-103 (ROS), por conjugación (teoría arriba)."""
    return _T_BO @ T_w_c @ _T_OB


def rotmat_to_quat_xyzw(R: np.ndarray):
    """Rotación → cuaternión (x, y, z, w) — el ORDEN de geometry_msgs.
    Fórmula de la traza con las 4 ramas numéricamente estables (la misma de
    vslam/mapping/gaussian.py, aquí en NumPy y orden xyzw)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return float(x), float(y), float(z), float(w)


def quat_xyzw_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def T_to_pose(T: np.ndarray) -> Pose:
    p = Pose()
    p.position.x, p.position.y, p.position.z = map(float, T[:3, 3])
    (p.orientation.x, p.orientation.y,
     p.orientation.z, p.orientation.w) = rotmat_to_quat_xyzw(T[:3, :3])
    return p


def pose_to_T(p: Pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_rotmat(p.orientation.x, p.orientation.y,
                                    p.orientation.z, p.orientation.w)
    T[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return T


def T_to_transform(T: np.ndarray) -> Transform:
    t = Transform()
    t.translation.x, t.translation.y, t.translation.z = map(float, T[:3, 3])
    (t.rotation.x, t.rotation.y,
     t.rotation.z, t.rotation.w) = rotmat_to_quat_xyzw(T[:3, :3])
    return t
