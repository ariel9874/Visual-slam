"""Trayectoria estimada + exportación en formato TUM para evaluación.

El formato TUM (`timestamp tx ty tz qx qy qz qw`, uno por línea) es el estándar
de facto para comparar trayectorias con herramientas como `evo`:
    evo_ape tum groundtruth.txt trajectory.txt -a   # error absoluto (alineado)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convierte una matriz de rotación 3x3 a cuaternión (qx, qy, qz, qw).

    ─── La matemática ───
    Un cuaternión unitario q = (v·sin(θ/2), cos(θ/2)) codifica la rotación de
    ángulo θ alrededor del eje unitario v. Ojo: q y −q representan la MISMA
    rotación (doble cobertura de SO(3)). Los formatos de trayectoria (TUM) lo
    prefieren a la matriz porque son 4 números con una sola restricción
    (‖q‖ = 1), no sufre gimbal lock y se interpola bien (slerp).

    La conversión invierte dos identidades de la fórmula de Rodrigues:
        tr(R) = 1 + 2·cos(θ)          (la traza fija el ángulo)
        R − R^T = 2·sin(θ)·[v]_x      (la parte antisimétrica fija el eje)
    Método de Shepperd: cada rama calcula primero la componente de q de mayor
    magnitud (según el elemento diagonal dominante) y deriva las demás de
    ella, para no dividir nunca por un número pequeño — estabilidad numérica
    incluso cerca de θ = 0 y θ = π, donde las fórmulas ingenuas cancelan.
    """
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw])


class Trajectory:
    """Secuencia de poses con marca de tiempo (T_w_c, convención del repo)."""

    def __init__(self) -> None:
        self._items: List[Tuple[float, np.ndarray]] = []

    def append(self, timestamp: float, T_w_c: np.ndarray) -> None:
        self._items.append((timestamp, np.asarray(T_w_c, dtype=np.float64).copy()))

    def __len__(self) -> int:
        return len(self._items)

    @property
    def positions(self) -> np.ndarray:
        """Posiciones de cámara (N, 3) en el frame del mundo."""
        if not self._items:
            return np.zeros((0, 3))
        return np.stack([T[:3, 3] for _, T in self._items])

    def save_tum(self, path: str | Path) -> None:
        """Guarda la trayectoria en formato TUM (compatible con `evo`)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for t, T in self._items:
            q = rotation_to_quaternion(T[:3, :3])
            tx, ty, tz = T[:3, 3]
            lines.append(f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
