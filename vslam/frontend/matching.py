"""Emparejamiento de descriptores entre dos frames.

Estrategia v0.1: fuerza bruta con distancia de Hamming (descriptores binarios)
+ *ratio test* de Lowe. El ratio test descarta emparejamientos ambiguos: si el
mejor candidato no es claramente mejor que el segundo (ratio < umbral), el
descriptor no es discriminativo ahí (texturas repetitivas) y es mejor tirarlo.
Menos matches buenos >> muchos matches dudosos: RANSAC lo agradece después.

─── La matemática ────────────────────────────────────────────────────────────
Distancia de Hamming:  d_H(a, b) = popcount(a XOR b)  = nº de bits en que
difieren los dos descriptores de 256 bits. Es la métrica natural del espacio
{0,1}^256, y el motivo de usar descriptores binarios: XOR + popcount son
instrucciones de CPU (64 bits por ciclo), así que comparar 2000×2000
candidatos por frame es trivial.

Ratio test (Lowe, 2004). Para cada descriptor consultado, sean d1 ≤ d2 las
distancias a sus DOS vecinos más cercanos en el otro frame. Se acepta solo si

    d1 < r · d2        (r ≈ 0.75)

Lectura estadística: si el match es espurio, d1 y d2 son dos muestras de la
misma distribución de "parecido casual" → d1/d2 ≈ 1. Un match correcto no
tiene rival plausible: d1 proviene de otra distribución (mismo punto físico)
y el cociente se hunde. El test elimina exactamente los casos ambiguos
(texturas repetitivas: ventanas, ladrillos), que son los que más daño le
hacen a RANSAC porque forman consensos falsos coherentes.
──────────────────────────────────────────────────────────────────────────────

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
