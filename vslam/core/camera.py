"""Modelo de cámara pinhole (proyectiva ideal, sin distorsión).

─── La matemática: proyección central ───────────────────────────────────────
Un punto X_c = (X, Y, Z) expresado en el frame de la cámara se proyecta al
plano imagen dividiendo por su profundidad:

    x_n = X/Z ,   y_n = Y/Z        (coordenadas "normalizadas": plano Z = 1)

y los intrínsecos convierten esas coordenadas métricas a píxeles:

    u = fx·x_n + cx ,   v = fy·y_n + cy

En forma matricial homogénea (λ absorbe la división por Z):

    λ·[u, v, 1]^T = K · X_c ,  λ = Z ,  K = [[fx,  0, cx],
                                             [ 0, fy, cy],
                                             [ 0,  0,  1]]

  - fx, fy: focal en píxeles (focal física / tamaño del píxel del sensor).
  - (cx, cy): punto principal — donde el eje óptico atraviesa el sensor.
  - La inversa K^{-1}·[u, v, 1]^T devuelve el RAYO que pasa por el píxel
    (dirección, sin profundidad).

La proyección DESTRUYE la profundidad: todos los puntos de la semirrecta
λ·(x_n, y_n, 1) con λ > 0 caen en el mismo píxel. Esa pérdida es el origen de
(a) la necesidad de triangular desde ≥ 2 vistas para tener 3D, y
(b) la ambigüedad de escala de TODO el SLAM monocular.
──────────────────────────────────────────────────────────────────────────────

Convenciones (fijadas para todo el repo, ver docs/02_arquitectura.md §4):
  - Ejes de cámara estilo OpenCV: +Z hacia delante, +X derecha, +Y abajo.

En v0.1 asumimos imágenes ya rectificadas (sin distorsión radial/tangencial).
Un modelo con distorsión (Brown-Conrady, Kannala-Brandt para fisheye) entrará
como subclase cuando soportemos datasets crudos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 0   # 0 = desconocido (opcional para la geometría)
    height: int = 0

    @property
    def K(self) -> np.ndarray:
        """Matriz de intrínsecos 3x3 (float64)."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """Proyecta puntos 3D expresados en el frame de la cámara a píxeles.

        Args:
            points_cam: array (N, 3) con Z > 0 (delante de la cámara).
        Returns:
            array (N, 2) de coordenadas de píxel (u, v).
        """
        pts = np.asarray(points_cam, dtype=np.float64)
        z = pts[:, 2]
        u = self.fx * pts[:, 0] / z + self.cx
        v = self.fy * pts[:, 1] / z + self.cy
        return np.stack([u, v], axis=1)

    def backproject(self, pixels: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Retro-proyecta píxeles (u, v) con profundidad Z al frame de la cámara."""
        px = np.asarray(pixels, dtype=np.float64)
        z = np.asarray(depth, dtype=np.float64)
        x = (px[:, 0] - self.cx) / self.fx * z
        y = (px[:, 1] - self.cy) / self.fy * z
        return np.stack([x, y, z], axis=1)

    @classmethod
    def from_file(cls, path: str | Path) -> "PinholeCamera":
        """Carga calibración desde un .txt con una línea: ``fx fy cx cy [width height]``.

        Las líneas que empiezan con '#' se ignoran (comentarios).
        """
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(v) for v in line.split()]
            if len(vals) < 4:
                raise ValueError(f"Calibración inválida en {path}: se esperan al menos fx fy cx cy")
            w, h = (int(vals[4]), int(vals[5])) if len(vals) >= 6 else (0, 0)
            return cls(fx=vals[0], fy=vals[1], cx=vals[2], cy=vals[3], width=w, height=h)
        raise ValueError(f"Archivo de calibración vacío: {path}")
