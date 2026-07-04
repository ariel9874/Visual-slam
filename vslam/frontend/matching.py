"""Emparejamiento de descriptores entre dos frames.

Estrategia v0.1: fuerza bruta con distancia de Hamming (descriptores binarios)
+ *ratio test* de Lowe. El ratio test descarta emparejamientos ambiguos: si el
mejor candidato no es claramente mejor que el segundo (ratio < umbral), el
descriptor no es discriminativo ahí (texturas repetitivas) y es mejor tirarlo.
Menos matches buenos >> muchos matches dudosos: RANSAC lo agradece después.

Alternativas que esta interfaz admite sin cambiar el pipeline:
  - cross-check (consistencia ida-vuelta) en lugar de ratio test,
  - FLANN/LSH para miles de características,
  - emparejadores aprendidos (SuperGlue/LightGlue) para condiciones extremas.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


def match_descriptors(
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    ratio: float = 0.75,
) -> List[cv2.DMatch]:
    """Empareja descriptores binarios de A contra B con knn(2) + ratio test.

    Args:
        desc_a: descriptores del frame A (N, 32) uint8 — serán los `queryIdx`.
        desc_b: descriptores del frame B (M, 32) uint8 — serán los `trainIdx`.
        ratio: umbral de Lowe (0.7-0.8 típico; más bajo = más estricto).
    Returns:
        Lista de cv2.DMatch que superan el test, ordenada por distancia.
    """
    if len(desc_a) < 2 or len(desc_b) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    return sorted(good, key=lambda m: m.distance)
