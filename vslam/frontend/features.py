"""Extracción de características — pieza intercambiable del frontend.

Hoy: ORB (Rublee et al., ICCV 2011) = detector FAST multi-escala + descriptor
binario BRIEF con orientación. Es la elección clásica (ORB-SLAM) porque es
gratis en CPU y sus descriptores se comparan con distancia de Hamming (XOR).

─── La matemática ────────────────────────────────────────────────────────────
FAST (detector). El píxel p es "esquina" si, en el círculo de Bresenham de
radio 3 que lo rodea (16 píxeles), existe un arco de n ≥ 9 contiguos TODOS más
claros que I(p) + τ o todos más oscuros que I(p) − τ (τ = fast_threshold).
Solo comparaciones de enteros → corre a cientos de Hz. La intuición: en un
borde el arco brillante mide ~8 (media circunferencia); solo una esquina
"encierra" al centro con un arco largo de un mismo signo.

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
──────────────────────────────────────────────────────────────────────────────

Mañana: para robustez en textura pobre/iluminación difícil, esta misma interfaz
admite un extractor aprendido (SuperPoint) sin tocar el resto del pipeline —
esa es exactamente la lección de los híbridos modernos (docs/01 §2.2, nota).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np


class FeatureExtractor:
    """Envoltorio fino sobre cv2.ORB con la interfaz estándar del repo.

    Cualquier extractor alternativo debe exponer el mismo método
    ``detect_and_compute(gray) -> (keypoints, descriptors)``.
    """

    def __init__(
        self,
        n_features: int = 2000,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        fast_threshold: int = 20,
    ) -> None:
        # La pirámide de escalas (n_levels, scale_factor) da invarianza a escala:
        # sin ella, acercarse a un objeto rompería el matching.
        self._orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            fastThreshold=fast_threshold,
        )

    def detect_and_compute(self, gray: np.ndarray) -> Tuple[Sequence, np.ndarray]:
        """Detecta keypoints y calcula sus descriptores.

        Args:
            gray: imagen en escala de grises (H, W) uint8.
        Returns:
            (keypoints, descriptors): lista de cv2.KeyPoint y array (N, 32) uint8.
            Si no se detecta nada, descriptors es un array vacío (0, 32).
        """
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        if descriptors is None:
            keypoints, descriptors = [], np.empty((0, 32), dtype=np.uint8)
        return keypoints, descriptors
