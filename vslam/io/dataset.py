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
from vslam.core.geometry import invert_se3

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
    """Itera sobre una secuencia TUM RGB-D: (timestamp, gris) o, con
    `with_depth=True` (v0.6), (timestamp, gris, profundidad_en_metros).

    A diferencia de ImageSequenceLoader, respeta los TIMESTAMPS REALES del
    archivo rgb.txt (la cámara no captura a fps constante) — imprescindible para
    asociar el ground truth de la mocap, que corre a otra frecuencia.

    Profundidad (v0.6): depth.txt se asocia al RGB por timestamp más cercano
    (Kinect: RGB y profundidad son sensores distintos, no sincronizados). Los
    PNG son uint16 con FACTOR 5000 (convención TUM: 5000 = 1 m); 0 = SIN DATO
    (sombras del proyector IR, superficies especulares, fuera de rango). Se
    devuelve float32 en METROS con NaN→0.0 implícito (0 sigue siendo "sin dato").

    Args:
        root: carpeta de la secuencia (rgb.txt, rgb/, [depth.txt, depth/]).
        with_depth: emitir también el mapa de profundidad asociado.
        max_depth_dt: descartar la profundidad si el par más cercano dista más
            de esto (s) — el frame se emite con depth=None.
    """

    DEPTH_FACTOR = 5000.0        # convención TUM: valor uint16 / 5000 = metros

    def __init__(self, root: str | Path, with_depth: bool = False,
                 max_depth_dt: float = 0.05) -> None:
        self.root = Path(root)
        rgb_txt = self.root / "rgb.txt"
        if not rgb_txt.is_file():
            raise FileNotFoundError(f"No existe {rgb_txt} (¿es una secuencia TUM?)")
        self.entries = _read_tum_index(rgb_txt)
        if not self.entries:
            raise FileNotFoundError(f"rgb.txt sin entradas en {self.root}")
        self.with_depth = with_depth
        self._depth_of: List[Optional[str]] = [None] * len(self.entries)
        if with_depth:
            depth_entries = _read_tum_index(self.root / "depth.txt")
            d_ts = np.array([t for t, _ in depth_entries])
            assoc = associate_by_timestamp(self.timestamps, d_ts,
                                           max_dt=max_depth_dt)
            for i, j in enumerate(assoc):
                if j >= 0:
                    self._depth_of[i] = depth_entries[j][1]

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([t for t, _ in self.entries], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[Tuple]:
        for i, (ts, rel) in enumerate(self.entries):
            image = cv2.imread(str(self.root / rel), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise IOError(f"No se pudo leer la imagen: {self.root / rel}")
            if not self.with_depth:
                yield ts, image
                continue
            depth = None
            if self._depth_of[i] is not None:
                raw = cv2.imread(str(self.root / self._depth_of[i]),
                                 cv2.IMREAD_UNCHANGED)
                if raw is not None:
                    depth = raw.astype(np.float32) / self.DEPTH_FACTOR
            yield ts, image, depth


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
    t_bs = _euroc_T_BS(root, cam)[:3, 3]
    pos = np.array([_quat_wxyz_to_R(q[i]) @ t_bs + p_body[i]
                    for i in range(len(ts))])
    return ts, pos


def _euroc_T_BS(root: str | Path, cam: str) -> np.ndarray:
    """Extrínseco T_BS (body←sensor, 4×4) de `mav0/<cam>/sensor.yaml`."""
    text = (Path(root) / "mav0" / cam / "sensor.yaml").read_text(encoding="utf-8")
    return np.array(_yaml_list(text, "data")).reshape(4, 4)


class EuRoCStereoRig:
    """Rectificación estéreo de un par EuRoC (cam0 izquierda, cam1 derecha).

    ─── La matemática: por qué RECTIFICAR ─────────────────────────────────────
    Dos cámaras cualesquiera ven un punto sobre su recta epipolar; buscar la
    correspondencia es una búsqueda 2D. La RECTIFICACIÓN reproyecta ambas
    imágenes a un par virtual con los ejes ópticos paralelos y los planos de
    imagen coplanares → las rectas epipolares se vuelven FILAS horizontales y la
    correspondencia colapsa a una búsqueda 1D: el mismo punto aparece en
    (u_L, v) y (u_R, v) con la MISMA v. La diferencia d = u_L − u_R es la
    DISPARIDAD, y la geometría del par da la profundidad:

        z = fx · b / d          (b = baseline; fx · b ≡ bf)

    `cv2.stereoRectify` calcula las homografías R1, R2 y las nuevas matrices de
    proyección P1, P2 desde los intrínsecos + la pose relativa cam0←cam1. Tras
    rectificar, la cámara izquierda es un pinhole SIN distorsión (P1) y
    bf = −P2[0,3] (P2 codifica el baseline como −fx·b en su columna de traslación).

    ─── La conexión con el hito 2 (RGB-D) ─────────────────────────────────────
    En RGB-D la cámara derecha era VIRTUAL: sintetizábamos u_R = u − bf/z desde
    la profundidad del sensor. Aquí la cámara derecha es REAL y u_R se MIDE
    (u_R = u_L − d, el match en la imagen derecha). El residuo del BA es
    idéntico ([u, v, u_R], teoría en bundle_adjustment.py) — solo cambia la
    PROCEDENCIA de z: sensor de profundidad vs. triangulación estéreo.
    """

    def __init__(self, root: str | Path, left: str = "cam0",
                 right: str = "cam1") -> None:
        camL, camR = euroc_camera(root, left), euroc_camera(root, right)
        # Pose relativa: X_right = T_R_L · X_left, con T_R_L = T_B_R⁻¹ · T_B_L
        # (ambos sensor.yaml dan body←sensor). stereoRectify quiere justo la
        # transformación de la 1ª cámara (izq) a la 2ª (der).
        T_R_L = invert_se3(_euroc_T_BS(root, right)) @ _euroc_T_BS(root, left)
        size = (camL.width, camL.height)
        R1, R2, P1, P2, self.Q, _, _ = cv2.stereoRectify(
            camL.K, camL.dist, camR.K, camR.dist, size,
            T_R_L[:3, :3], T_R_L[:3, 3],
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        # Cámara izquierda RECTIFICADA (sin distorsión: la rectificación la quita).
        self.camera = PinholeCamera(fx=P1[0, 0], fy=P1[1, 1],
                                    cx=P1[0, 2], cy=P1[1, 2],
                                    width=size[0], height=size[1])
        self.baseline = float(-P2[0, 3] / P2[0, 0])   # metros
        self.bf = float(-P2[0, 3])                    # fx · baseline (px·m)
        self.map_left = cv2.initUndistortRectifyMap(
            camL.K, camL.dist, R1, P1, size, cv2.CV_32FC1)
        self.map_right = cv2.initUndistortRectifyMap(
            camR.K, camR.dist, R2, P2, size, cv2.CV_32FC1)

    def rectify(self, gray_left: np.ndarray, gray_right: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        """(izq, der) rectificadas: filas = rectas epipolares (búsqueda 1D)."""
        return (cv2.remap(gray_left, *self.map_left, cv2.INTER_LINEAR),
                cv2.remap(gray_right, *self.map_right, cv2.INTER_LINEAR))


class EuRoCStereoLoader:
    """Itera (ts, izquierda_rectificada, profundidad) — MISMA interfaz que
    `TUMRGBDLoader(with_depth=True)`, para que el tracker RGB-D métrico (v0.6)
    funcione sin cambios. La profundidad NO viene de un sensor: se triangula por
    DISPARIDAD estéreo densa (`cv2.StereoSGBM`) sobre el par rectificado.

    depth = bf / disparidad; = 0 donde la disparidad es inválida o cae fuera de
    [min_depth, max_depth] (mismo convenio '0 = sin dato' que RGB-D). El ruido
    de la profundidad crece con z² (∂z/∂d = −bf/d²): por eso el residuo del BA
    pesa u_R = u − bf/z, cuyo peso decae exactamente con z² — la geometría
    compensa el ruido (ver bundle_adjustment.py).
    """

    def __init__(self, root: str | Path, rig: Optional[EuRoCStereoRig] = None,
                 num_disparities: int = 96, block_size: int = 7,
                 min_depth: float = 0.5, max_depth: float = 40.0) -> None:
        self.rig = rig or EuRoCStereoRig(root)
        self._left = EuRoCLoader(root, "cam0")
        self._right = EuRoCLoader(root, "cam1")
        self.min_depth, self.max_depth = min_depth, max_depth
        # SGBM: robusto y estándar. P1/P2 = suavidad (penalización a saltos de
        # disparidad), escalados con el tamaño de bloque (recomendación de OpenCV).
        self._sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disparities, blockSize=block_size,
            P1=8 * block_size ** 2, P2=32 * block_size ** 2,
            uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
            disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    @property
    def camera(self) -> PinholeCamera:
        return self.rig.camera            # la izquierda RECTIFICADA

    @property
    def stereo_bf(self) -> float:
        return self.rig.bf

    def __len__(self) -> int:
        return min(len(self._left), len(self._right))

    def __iter__(self) -> Iterator[Tuple[float, np.ndarray, np.ndarray]]:
        for (ts, gl), (_, gr) in zip(self._left, self._right):
            L, R = self.rig.rectify(gl, gr)
            disp = self._sgbm.compute(L, R).astype(np.float32) / 16.0
            depth = np.zeros_like(disp)
            valid = disp > 0.0
            depth[valid] = self.rig.bf / disp[valid]
            depth[(depth < self.min_depth) | (depth > self.max_depth)] = 0.0
            yield ts, L, depth
