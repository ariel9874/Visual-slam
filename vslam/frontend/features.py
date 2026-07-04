"""Extracción de características — pieza intercambiable del frontend.

Hoy: ORB (Rublee et al., ICCV 2011) = detector FAST multi-escala + descriptor
binario BRIEF con orientación. Es la elección clásica (ORB-SLAM) porque es
gratis en CPU y sus descriptores se comparan con distancia de Hamming (XOR).

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
