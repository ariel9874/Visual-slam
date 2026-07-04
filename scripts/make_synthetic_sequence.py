#!/usr/bin/env python3
"""Genera una secuencia sintética de imágenes con ground truth exacto.

Escena: tres planos frontales texturizados con ruido multi-escala, colocados a
distintas profundidades (composición lejano→cercano para ocluir correctamente).
Cada plano se renderiza con la homografía EXACTA inducida por la pose de la
cámara, así que la geometría de la secuencia es perfecta y el ground truth
también. Tres profundidades distintas = escena no plana = matriz esencial bien
condicionada; textura de ruido = miles de características ORB discriminativas.

Salida (en --output):
    images/000000.png ...   la secuencia
    calib.txt               fx fy cx cy width height
    groundtruth.txt         poses reales en formato TUM (comparables con evo)

Uso:
    python scripts/make_synthetic_sequence.py --output data/synthetic
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera
from vslam.core.trajectory import Trajectory


def rot_y(theta: float) -> np.ndarray:
    """Rotación alrededor del eje Y (guiñada, con ejes de cámara estilo OpenCV)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def make_texture(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Textura de ruido multi-escala (suma de octavas): rica en esquinas a
    varias frecuencias espaciales, ideal para detectores tipo FAST/ORB."""
    tex = np.zeros((height, width), dtype=np.float64)
    for cell, weight in [(160, 1.0), (60, 0.8), (24, 0.6), (8, 0.4)]:
        octave = rng.uniform(0, 1, (height // cell + 2, width // cell + 2))
        tex += weight * cv2.resize(octave, (width, height), interpolation=cv2.INTER_CUBIC)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    return (30 + tex * 195).astype(np.uint8)  # rango [30, 225]


@dataclass
class TexturedPlane:
    """Plano frontal (paralelo al plano imagen inicial) en z = depth.

    Corners en mundo: (cx ± hx, cy ± hy, depth). La textura se mapea
    linealmente sobre el quad, por lo que la imagen del plano bajo cualquier
    cámara pinhole es una homografía exacta de la textura.
    """
    center_x: float
    center_y: float
    depth: float
    half_x: float
    half_y: float
    texture: np.ndarray

    def world_corners(self) -> np.ndarray:
        cx, cy, z, hx, hy = self.center_x, self.center_y, self.depth, self.half_x, self.half_y
        # Orden: (izq-arriba, der-arriba, der-abajo, izq-abajo) — debe coincidir
        # con el orden de las esquinas de la textura en render_plane().
        return np.array([
            [cx - hx, cy - hy, z],
            [cx + hx, cy - hy, z],
            [cx + hx, cy + hy, z],
            [cx - hx, cy + hy, z],
        ])


def render_plane(canvas: np.ndarray, plane: TexturedPlane,
                 camera: PinholeCamera, R_w_c: np.ndarray, C: np.ndarray) -> None:
    """Renderiza el plano sobre el canvas con la homografía exacta textura→imagen."""
    # Esquinas del plano en el frame de la cámara: X_c = R_w_c^T (X_w - C).
    corners_cam = (R_w_c.T @ (plane.world_corners() - C).T).T
    if np.any(corners_cam[:, 2] < 0.2):  # plano (parcialmente) detrás de la cámara
        return
    img_pts = camera.project(corners_cam).astype(np.float32)

    th, tw = plane.texture.shape
    tex_pts = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)

    H_mat = cv2.getPerspectiveTransform(tex_pts, img_pts)
    size = (canvas.shape[1], canvas.shape[0])
    warped = cv2.warpPerspective(plane.texture, H_mat, size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(
        np.full((th, tw), 255, np.uint8), H_mat, size, flags=cv2.INTER_NEAREST
    )
    canvas[mask > 127] = warped[mask > 127]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    W, H = 640, 480
    camera = PinholeCamera(fx=450.0, fy=450.0, cx=W / 2, cy=H / 2, width=W, height=H)

    # Tres planos a profundidades distintas (escena NO plana). El fondo es
    # enorme para cubrir todo el recorrido; los cercanos dan paralaje fuerte.
    planes = [  # de lejano a cercano (el orden de render resuelve la oclusión)
        TexturedPlane(6.0, 0.0, 14.0, 18.0, 9.0, make_texture(2200, 1100, rng)),
        TexturedPlane(1.5, 1.2, 8.0, 4.0, 2.6, make_texture(900, 600, rng)),
        TexturedPlane(4.5, -1.0, 5.5, 2.2, 1.5, make_texture(640, 440, rng)),
    ]

    out = Path(args.output)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    gt = Trajectory()
    step = 0.05  # velocidad ~constante => la VO monocular (||t||=1) conserva la forma

    for k in range(args.frames):
        # Pose real: avanza en +X con leve deriva en Z y giro de guiñada suave.
        yaw = 0.003 * k
        R_w_c = rot_y(yaw)
        C = np.array([step * k, 0.0, 0.012 * k])
        T_w_c = np.eye(4)
        T_w_c[:3, :3] = R_w_c
        T_w_c[:3, 3] = C
        gt.append(k / 30.0, T_w_c)

        canvas = np.full((H, W), 15, np.uint8)  # "cielo" casi negro y sin textura
        for plane in planes:
            render_plane(canvas, plane, camera, R_w_c, C)

        cv2.imwrite(str(img_dir / f"{k:06d}.png"), canvas)

    (out / "calib.txt").write_text(
        f"# fx fy cx cy width height\n{camera.fx} {camera.fy} {camera.cx} {camera.cy} {W} {H}\n",
        encoding="utf-8",
    )
    gt.save_tum(out / "groundtruth.txt")
    print(f"OK: {args.frames} frames en {img_dir}")
    print(f"    calibración: {out / 'calib.txt'}")
    print(f"    ground truth (TUM): {out / 'groundtruth.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
