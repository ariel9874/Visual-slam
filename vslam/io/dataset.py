"""Carga de secuencias de imágenes desde disco.

v0.1: un loader genérico que sirve para KITTI (carpeta image_0/), secuencias
sintéticas de este repo, o frames exportados de un video con ffmpeg:
    ffmpeg -i video.mp4 -vf "fps=15" frames/%06d.png

TODO(v0.2): subclases con las convenciones de cada dataset (timestamps reales,
ground truth, calibración): TUMLoader, KITTILoader, EuRoCLoader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


class ImageSequenceLoader:
    """Itera (timestamp, imagen_gris) sobre una carpeta de imágenes ordenadas.

    Args:
        directory: carpeta con las imágenes (se ordenan por nombre de archivo,
            por eso los datasets numeran con ceros a la izquierda: 000001.png).
        fps: si no hay archivo de timestamps, se asigna t = índice / fps.
    """

    def __init__(self, directory: str | Path, fps: float = 30.0) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"No existe el directorio de imágenes: {self.directory}")
        self.paths = sorted(
            p for p in self.directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No se encontraron imágenes en: {self.directory}")
        self.fps = fps

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[Tuple[float, np.ndarray]]:
        for i, path in enumerate(self.paths):
            # IMREAD_GRAYSCALE: la geometría de v0.1 solo necesita intensidades.
            # (Los mappers fotométricos de v0.5 querrán el color: se añadirá
            # un flag `grayscale=False` llegado el momento.)
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise IOError(f"No se pudo leer la imagen: {path}")
            yield i / self.fps, image
