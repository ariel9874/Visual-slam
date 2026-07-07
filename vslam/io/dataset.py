"""Carga de secuencias de imágenes desde disco.

v0.1: un loader genérico que sirve para KITTI (carpeta image_0/), secuencias
sintéticas de este repo, o frames exportados de un video con ffmpeg:
    ffmpeg -i video.mp4 -vf "fps=15" frames/%06d.png

v0.45: loaders de datasets reales con sus convenciones (timestamps, ground
truth, calibración). Aquí: TUM RGB-D. KITTI y EuRoC vendrán después.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from vslam.core.camera import PinholeCamera

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# Intrínsecos + distorsión Brown-Conrady de las cámaras de TUM RGB-D
# (https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect).
# Formato: (fx, fy, cx, cy, k1, k2, p1, p2, k3). fr3 se distribuye ya rectificada.
TUM_INTRINSICS = {
    "freiburg1": (517.306408, 516.469215, 318.643040, 255.313989,
                  0.262383, -0.953104, -0.005358, 0.002628, 1.163314),
    "freiburg2": (520.908620, 521.007327, 325.141442, 249.701764,
                  0.231222, -0.784899, -0.003257, -0.000105, 0.917205),
    "freiburg3": (535.4, 539.2, 320.1, 247.6, 0.0, 0.0, 0.0, 0.0, 0.0),
}


def tum_camera(sequence: str) -> PinholeCamera:
    """Cámara de una secuencia TUM a partir de su nombre (busca 'freiburgN')."""
    for key, p in TUM_INTRINSICS.items():
        if key in sequence or key.replace("freiburg", "fr") in sequence:
            return PinholeCamera(fx=p[0], fy=p[1], cx=p[2], cy=p[3],
                                 width=640, height=480, distortion=p[4:9])
    raise ValueError(f"No reconozco la cámara TUM de '{sequence}' "
                     f"(esperaba freiburg1/2/3 en el nombre)")


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


def _read_tum_index(path: Path) -> List[Tuple[float, str]]:
    """Lee un archivo 'timestamp ruta' de TUM (rgb.txt / depth.txt). Ignora #."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ts, rel = line.split()
        out.append((float(ts), rel))
    return out


class TUMRGBDLoader:
    """Itera (timestamp, imagen_gris) sobre una secuencia TUM RGB-D.

    A diferencia de ImageSequenceLoader, respeta los TIMESTAMPS REALES del
    archivo rgb.txt (la cámara no captura a fps constante) — imprescindible para
    asociar el ground truth de la mocap, que corre a otra frecuencia. Solo se
    usa el canal RGB (monocular); la profundidad queda para v0.6 (RGB-D).

    Args:
        root: carpeta de la secuencia (contiene rgb.txt, rgb/, groundtruth.txt).
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        rgb_txt = self.root / "rgb.txt"
        if not rgb_txt.is_file():
            raise FileNotFoundError(f"No existe {rgb_txt} (¿es una secuencia TUM?)")
        self.entries = _read_tum_index(rgb_txt)
        if not self.entries:
            raise FileNotFoundError(f"rgb.txt sin entradas en {self.root}")

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([t for t, _ in self.entries], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[Tuple[float, np.ndarray]]:
        for ts, rel in self.entries:
            image = cv2.imread(str(self.root / rel), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise IOError(f"No se pudo leer la imagen: {self.root / rel}")
            yield ts, image


def read_tum_trajectory(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Lee una trayectoria TUM (t tx ty tz qx qy qz qw). Devuelve (timestamps
    (M,), posiciones (M, 3))."""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0].astype(np.float64), data[:, 1:4].astype(np.float64)


def associate_by_timestamp(query_ts: np.ndarray, ref_ts: np.ndarray,
                           max_dt: float = 0.02) -> np.ndarray:
    """Para cada timestamp de `query_ts`, el índice del más cercano en `ref_ts`
    (o -1 si el más cercano dista más de `max_dt` s).

    Es la asociación estándar de TUM (rgb ↔ mocap): la mocap corre a ~100-200 Hz
    y el RGB a ~30 Hz, así que casi todo frame RGB tiene un GT a < 20 ms. Los
    frames sin GT cercano se excluyen SOLO de la evaluación (no del tracking).
    """
    ref = np.asarray(ref_ts, dtype=np.float64)
    order = np.argsort(ref)
    ref_sorted = ref[order]
    idx = np.searchsorted(ref_sorted, query_ts)
    out = np.full(len(query_ts), -1, dtype=int)
    for i, (q, j) in enumerate(zip(query_ts, idx)):
        cands = [k for k in (j - 1, j) if 0 <= k < len(ref_sorted)]
        best = min(cands, key=lambda k: abs(ref_sorted[k] - q), default=None)
        if best is not None and abs(ref_sorted[best] - q) <= max_dt:
            out[i] = order[best]
    return out
