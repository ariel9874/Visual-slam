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

─── La matemática: distorsión de lente (Brown-Conrady) ───────────────────────
La proyección ideal de arriba asume una lente perfecta. Una lente real desvía
el rayo: la radial (barril/cojín) crece con el radio, la tangencial nace de que
el sensor no es perfectamente paralelo a la lente. El modelo estándar (el que
usan TUM, KITTI, OpenCV) actúa sobre las coordenadas NORMALIZADAS antes de K:

    r² = x_n² + y_n²
    radial      = 1 + k1·r² + k2·r⁴ + k3·r⁶
    x_d = x_n·radial + 2·p1·x_n·y_n + p2·(r² + 2·x_n²)      (tangencial)
    y_d = y_n·radial + p1·(r² + 2·y_n²) + 2·p2·x_n·y_n
    (u, v) = (fx·x_d + cx,  fy·y_d + cy)

El SLAM geométrico (E, PnP, triangulación) vive en el modelo IDEAL: hay que
DES-distorsionar los píxeles ANTES de tocar la geometría. Como la relación
directa (x_n → x_d) no tiene inversa cerrada, `undistort_points` la resuelve
numéricamente (Newton, vía cv2). Con distorsión nula es la identidad, así que
el sintético (dist=0) pasa sin cambios — la geometría de v0.1-0.4 no se entera.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 0   # 0 = desconocido (opcional para la geometría)
    height: int = 0
    # Distorsión Brown-Conrady (k1, k2, p1, p2, k3); tupla para que el
    # dataclass frozen siga siendo hashable (un ndarray no lo es). (0,)*5 = sin
    # distorsión = identidad: el pipeline sintético no cambia.
    distortion: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def K(self) -> np.ndarray:
        """Matriz de intrínsecos 3x3 (float64)."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def dist(self) -> np.ndarray:
        """Coeficientes de distorsión (5,) en el orden de OpenCV: k1 k2 p1 p2 k3."""
        return np.array(self.distortion, dtype=np.float64)

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(self.dist != 0.0))

    def undistort_points(self, pixels: np.ndarray) -> np.ndarray:
        """Lleva píxeles DISTORSIONADOS al plano imagen ideal (píxeles sin
        distorsión, en los mismos intrínsecos K). Es el puente entre un dataset
        crudo y la geometría ideal del repo: llamarlo sobre los keypoints ANTES
        de E/PnP/triangulación. Con dist=0 devuelve los píxeles intactos.
        """
        px = np.asarray(pixels, dtype=np.float64)
        if not self.has_distortion or px.size == 0:
            return px.reshape(-1, 2)
        import cv2  # local: mantiene el módulo importable sin cv2 (docs/tests)
        # P=K re-proyecta a píxeles (sin P, cv2 devolvería coords normalizadas).
        und = cv2.undistortPoints(px.reshape(-1, 1, 2), self.K, self.dist, P=self.K)
        return und.reshape(-1, 2)

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
        """Carga calibración desde un .txt con una línea:
        ``fx fy cx cy [width height [k1 k2 p1 p2 k3]]``.

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
            dist = tuple(vals[6:11]) if len(vals) >= 11 else (0.0,) * 5
            return cls(fx=vals[0], fy=vals[1], cx=vals[2], cy=vals[3], width=w, height=h,
                       distortion=dist)  # type: ignore[arg-type]
        raise ValueError(f"Archivo de calibración vacío: {path}")
