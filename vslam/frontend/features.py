"""Extracción de características — pieza intercambiable del frontend.

Este módulo es un REGISTRO de extractores tras una interfaz común
(`FeatureExtractorBase`): el pipeline pide `create_extractor("akaze")` y no
sabe nada más. Análisis comparativo de cada técnica (idea matemática,
beneficios, cuellos de botella, cuándo usarla): docs/03_detectores_y_matchers.md.

Los extractores aprendidos (SuperPoint, DISK) viven en learned.py y requieren
el extra opcional `pip install -e ".[deep]"` — si faltan las dependencias, el
registro da un error claro en lugar de romper la instalación base.

─── La matemática (del default, ORB) ─────────────────────────────────────────
FAST (detector). El píxel p es "esquina" si, en el círculo de Bresenham de
radio 3 que lo rodea (16 píxeles), existe un arco de n ≥ 9 contiguos TODOS más
claros que I(p) + τ o todos más oscuros que I(p) − τ (τ = fast_threshold).
Solo comparaciones de enteros → corre a cientos de Hz. La intuición: en un
borde el arco de un mismo signo mide ~8 (media circunferencia); solo una
esquina "encierra" al centro con un arco largo.

Orientación (la O de ORB). Momentos de intensidad del parche alrededor de p:
    m_pq = Σ_{x,y} x^p · y^q · I(x, y)
El centroide C = (m10/m00, m01/m00) define θ = atan2(m01, m10): el vector
p→C "apunta hacia lo brillante" y gira solidario con la imagen — un marco de
referencia reproducible que hace al descriptor invariante a rotación.

BRIEF (descriptor). 256 comparaciones de pares de píxeles (p_k, q_k), fijados
de antemano y rotados por θ, sobre el parche suavizado:
    bit_k = 1  si  I(p_k) < I(q_k),   0 en caso contrario
→ d ∈ {0,1}^256 (32 bytes). La información no está en las intensidades sino
en su ORDEN relativo: por eso tolera cambios de iluminación monótonos.

Pirámide de escalas. La imagen se re-muestrea n_levels veces por factor
1/scale_factor: el mismo punto visto desde más cerca/lejos se detecta en otro
nivel pero produce un descriptor comparable → invarianza a escala.

(Las matemáticas de AKAZE/BRISK/SIFT/KAZE están resumidas en el docstring de
cada clase y desarrolladas en docs/03.)
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np


class FeatureExtractorBase(ABC):
    """Contrato de todo extractor: imagen gris → (keypoints, descriptores).

    Attributes:
        name: clave del registro (la que acepta --detector en el ejemplo 01).
        descriptor_type: "binary" (uint8, métrica Hamming) o "float"
            (float32, métrica L2). El matcher elige su métrica leyendo esto
            (o el dtype), así que extractor y matcher se combinan libremente.
    """

    name: str = "?"
    descriptor_type: str = "binary"

    @abstractmethod
    def detect_and_compute(self, gray: np.ndarray) -> Tuple[Sequence, np.ndarray]:
        """Detecta keypoints y calcula descriptores.

        Args:
            gray: imagen en escala de grises (H, W) uint8.
        Returns:
            (keypoints, descriptors): lista de cv2.KeyPoint y array (N, D).
            Si no se detecta nada, descriptors es un array vacío (0, D).
        """

    def _empty(self, dim: int) -> Tuple[list, np.ndarray]:
        dtype = np.uint8 if self.descriptor_type == "binary" else np.float32
        return [], np.empty((0, dim), dtype=dtype)


class ORBExtractor(FeatureExtractorBase):
    """ORB: FAST multi-escala + BRIEF orientado (matemática arriba).

    El default del repo: binario, tiempo real en CPU, el estándar de los
    SLAM clásicos (ORB-SLAM).
    """

    name = "orb"
    descriptor_type = "binary"

    def __init__(self, n_features: int = 2000, scale_factor: float = 1.2,
                 n_levels: int = 8, fast_threshold: int = 20) -> None:
        self._orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=scale_factor,
                                   nlevels=n_levels, fastThreshold=fast_threshold)

    def detect_and_compute(self, gray):
        kps, desc = self._orb.detectAndCompute(gray, None)
        return (kps, desc) if desc is not None else self._empty(32)


class AKAZEExtractor(FeatureExtractorBase):
    """AKAZE: espacio de escalas NO lineal + descriptor binario M-LDB.

    ─── La matemática ───
    En lugar de la pirámide gaussiana (que difumina todo por igual), difunde
    la imagen resolviendo   ∂L/∂t = div( g(|∇L|)·∇L ),  donde la conductividad
    g(·) se anula en gradientes fuertes: el suavizado progresa DENTRO de las
    regiones pero se detiene en los bordes. Resultado: keypoints (máximos del
    det. del hessiano en ese espacio) mucho más repetibles con blur y cambios
    de iluminación que los de una pirámide lineal. Cuesta 2-4× ORB.
    """

    name = "akaze"
    descriptor_type = "binary"

    def __init__(self, threshold: float = 0.001) -> None:
        self._akaze = cv2.AKAZE_create(threshold=threshold)

    def detect_and_compute(self, gray):
        kps, desc = self._akaze.detectAndCompute(gray, None)
        return (kps, desc) if desc is not None else self._empty(61)


class BRISKExtractor(FeatureExtractorBase):
    """BRISK: anillos concéntricos; pares largos orientan, cortos describen.

    ─── La matemática ───
    Muestrea el parche en anillos concéntricos (suavizado ∝ radio). Los pares
    de LARGA distancia estiman la orientación global del parche:
        g = (1/L) · Σ (I(p_j) − I(p_i)) · (p_j − p_i)/‖p_j − p_i‖²
    y los de CORTA distancia generan los 512 bits del descriptor. La escala
    se interpola de forma continua entre niveles de la pirámide (más fina
    que la escala discreta de ORB).
    """

    name = "brisk"
    descriptor_type = "binary"

    def __init__(self, threshold: int = 30) -> None:
        self._brisk = cv2.BRISK_create(thresh=threshold)

    def detect_and_compute(self, gray):
        kps, desc = self._brisk.detectAndCompute(gray, None)
        return (kps, desc) if desc is not None else self._empty(64)


class SIFTExtractor(FeatureExtractorBase):
    """SIFT: extremos de DoG en espacio-escala + histogramas de gradiente.

    ─── La matemática ───
    Detecta extremos locales en (x, y, σ) de la Diferencia de Gaussianas
        DoG(x, σ) = L(x, k·σ) − L(x, σ)  ≈  (k−1)·σ²·∇²G
    (aproximación del laplaciano NORMALIZADO por σ², que es lo que hace la
    respuesta comparable entre escalas → invarianza a escala de verdad).
    Descriptor: rejilla 4×4 de histogramas de 8 orientaciones de gradiente
    → 128 floats, normalizados (invarianza afín a iluminación). Preciso y
    robusto; ~10× ORB en CPU y matching L2 más caro.
    """

    name = "sift"
    descriptor_type = "float"

    def __init__(self, n_features: int = 2000) -> None:
        self._sift = cv2.SIFT_create(nfeatures=n_features)

    def detect_and_compute(self, gray):
        kps, desc = self._sift.detectAndCompute(gray, None)
        return (kps, desc) if desc is not None else self._empty(128)


class KAZEExtractor(FeatureExtractorBase):
    """KAZE: el espacio no lineal de AKAZE con descriptor float (M-SURF).

    Algo más preciso que AKAZE y bastante más caro (el más lento del catálogo
    clásico): útil como techo de calidad en benchmarks, raro en producción.
    """

    name = "kaze"
    descriptor_type = "float"

    def __init__(self, threshold: float = 0.001) -> None:
        self._kaze = cv2.KAZE_create(threshold=threshold)

    def detect_and_compute(self, gray):
        kps, desc = self._kaze.detectAndCompute(gray, None)
        return (kps, desc) if desc is not None else self._empty(64)


class GFTTORBExtractor(FeatureExtractorBase):
    """Combo didáctico: detector Shi-Tomasi (GFTT) + descriptor de ORB.

    ─── La matemática (Shi-Tomasi) ───
    Tensor de estructura del parche:  M = Σ_w ∇I·∇Iᵀ  (2×2). El parche es
    "esquina" si su MENOR autovalor cumple min(λ₁, λ₂) > umbral: hay gradiente
    fuerte en dos direcciones independientes y la posición queda determinada
    sin ambigüedad. Es exactamente la matriz que invierte el tracker KLT, por
    eso estas esquinas son las "good features to TRACK".

    El propósito del combo es pedagógico: demuestra que detector y descriptor
    son piezas separables (y que emparejarlas mal tiene costos — GFTT no
    aporta escala, así que el descriptor pierde invarianza a escala).
    """

    name = "gftt-orb"
    descriptor_type = "binary"

    def __init__(self, n_features: int = 2000) -> None:
        self._gftt = cv2.GFTTDetector_create(maxCorners=n_features,
                                             qualityLevel=0.01, minDistance=7)
        self._orb = cv2.ORB_create(nfeatures=n_features)

    def detect_and_compute(self, gray):
        kps = self._gftt.detect(gray, None)
        if not kps:
            return self._empty(32)
        kps, desc = self._orb.compute(gray, kps)
        return (kps, desc) if desc is not None else self._empty(32)


# ─────────────────────────────── registro ────────────────────────────────────

def _superpoint_factory(**kwargs) -> FeatureExtractorBase:
    from vslam.frontend.learned import SuperPointExtractor
    return SuperPointExtractor(**kwargs)


def _disk_factory(**kwargs) -> FeatureExtractorBase:
    from vslam.frontend.learned import DISKExtractor
    return DISKExtractor(**kwargs)


_EXTRACTORS: Dict[str, Callable[..., FeatureExtractorBase]] = {
    "orb": ORBExtractor,
    "akaze": AKAZEExtractor,
    "brisk": BRISKExtractor,
    "sift": SIFTExtractor,
    "kaze": KAZEExtractor,
    "gftt-orb": GFTTORBExtractor,
    # Aprendidos (requieren `pip install -e ".[deep]"`, ver learned.py):
    "superpoint": _superpoint_factory,
    "disk": _disk_factory,
}


def available_extractors() -> List[str]:
    """Nombres registrados (los aprendidos pueden fallar al CREARLOS si
    faltan sus dependencias — el registro lista, la creación valida)."""
    return sorted(_EXTRACTORS)


def create_extractor(name: str, **kwargs) -> FeatureExtractorBase:
    """Instancia un extractor del registro por nombre (ver docs/03)."""
    try:
        factory = _EXTRACTORS[name]
    except KeyError:
        raise ValueError(f"Extractor desconocido: {name!r}. "
                         f"Disponibles: {', '.join(available_extractors())}") from None
    return factory(**kwargs)


# Alias retrocompatible (v0.1 exponía la clase ORB con este nombre).
FeatureExtractor = ORBExtractor
