"""Carga de secuencias de imágenes desde disco.

v0.1: un loader genérico que sirve para KITTI (carpeta image_0/), secuencias
sintéticas de este repo, o frames exportados de un video con ffmpeg:
    ffmpeg -i video.mp4 -vf "fps=15" frames/%06d.png

v0.45: loaders de datasets reales con sus convenciones (timestamps, ground
truth, calibración). Aquí: TUM RGB-D. KITTI y EuRoC vendrán después.
"""

from __future__ import annotations

import re
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


# ── EuRoC MAV (formato ASL) ──────────────────────────────────────────────────
# Estructura: <sec>/mav0/cam0/{data.csv, data/*.png, sensor.yaml} +
#             <sec>/mav0/state_groundtruth_estimate0/data.csv (GT en frame del
#             CUERPO/IMU — hay que llevarlo a la cámara con el extrínseco T_BS).
# Dron con movimiento agresivo 6-DoF: buen estrés para reloc/gate, distinto a TUM.

def _yaml_list(text: str, key: str) -> List[float]:
    """Extrae la lista `key: [a, b, c, ...]` de un YAML de EuRoC (parser mínimo,
    sin dependencia de PyYAML). Soporta listas multilínea (T_BS.data) e ignora
    comentarios tras el cierre `]`. Es cuadrado para lo que necesitamos —
    intrinsics, distortion_coefficients, resolution y la matriz T_BS."""
    m = re.search(rf"(?m)^\s*{re.escape(key)}\s*:\s*\[(.*?)\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"clave '{key}' no encontrada o sin lista en el sensor.yaml")
    return [float(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """Rotación (3×3) desde un cuaternión [w, x, y, z] (la convención del GT de
    EuRoC: q_RS = orientación del cuerpo en el frame de referencia)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def euroc_camera(root: str | Path, cam: str = "cam0") -> PinholeCamera:
    """Cámara EuRoC desde `mav0/<cam>/sensor.yaml` (pinhole + radial-tangencial).
    EuRoC da 4 coeficientes (k1, k2, p1, p2); k3 = 0 en Brown-Conrady."""
    text = (Path(root) / "mav0" / cam / "sensor.yaml").read_text(encoding="utf-8")
    fx, fy, cx, cy = _yaml_list(text, "intrinsics")
    k1, k2, p1, p2 = _yaml_list(text, "distortion_coefficients")[:4]
    w, h = (int(round(v)) for v in _yaml_list(text, "resolution"))
    return PinholeCamera(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h,
                         distortion=(k1, k2, p1, p2, 0.0))


class EuRoCLoader:
    """Itera (timestamp_seg, imagen_gris) sobre una secuencia EuRoC MAV.

    Los timestamps del CSV están en NANOsegundos → se pasan a segundos (÷1e9)
    para casar con el GT y con el resto del repo (formato TUM en segundos).
    """

    def __init__(self, root: str | Path, cam: str = "cam0") -> None:
        self.root = Path(root)
        self.cam_dir = self.root / "mav0" / cam
        csv = self.cam_dir / "data.csv"
        if not csv.is_file():
            raise FileNotFoundError(f"No existe {csv} (¿es una secuencia EuRoC?)")
        self.entries: List[Tuple[float, str]] = []
        for line in csv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ts, fn = line.split(",")[:2]
            self.entries.append((int(ts) * 1e-9, fn.strip()))
        if not self.entries:
            raise FileNotFoundError(f"data.csv sin entradas en {self.cam_dir}")

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([t for t, _ in self.entries], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[Tuple[float, np.ndarray]]:
        for ts, fn in self.entries:
            image = cv2.imread(str(self.cam_dir / "data" / fn), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise IOError(f"No se pudo leer la imagen: {self.cam_dir / 'data' / fn}")
            yield ts, image


def read_euroc_groundtruth(root: str | Path, cam: str = "cam0"
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """(timestamps_seg, posiciones de la CÁMARA en el mundo) del GT de EuRoC.

    ─── La matemática: el GT vive en el frame del CUERPO (IMU) ───
    EuRoC entrega T_world_body (posición p_RS_R + cuaternión q_RS del cuerpo).
    La cámara está desplazada del cuerpo por el extrínseco T_BS (cuerpo←cámara,
    en cam0/sensor.yaml). La posición de la cámara en el mundo es:

        p_cam_world = T_world_body · T_BS · [0,0,0,1]ᵀ
                    = R_world_body · t_BS + p_world_body

    Sin esta corrección, comparar la trayectoria de la cámara (estimada) con la
    del cuerpo (GT) mete un error de brazo de palanca que ROTA con la pose (no
    lo absorbe la alineación de similitud del ATE). El brazo t_BS en EuRoC es de
    ~7 cm: pequeño pero medible, y es la trampa que advierte docs/05 §7.
    """
    gt_csv = Path(root) / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    data = np.loadtxt(gt_csv, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    ts = data[:, 0] * 1e-9
    p_body = data[:, 1:4]
    q = data[:, 4:8]                                  # [w, x, y, z]
    text = (Path(root) / "mav0" / cam / "sensor.yaml").read_text(encoding="utf-8")
    t_bs = np.array(_yaml_list(text, "data")).reshape(4, 4)[:3, 3]
    pos = np.array([_quat_wxyz_to_R(q[i]) @ t_bs + p_body[i]
                    for i in range(len(ts))])
    return ts, pos
