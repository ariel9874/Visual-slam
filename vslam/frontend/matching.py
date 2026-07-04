"""Emparejamiento de descriptores — registro de matchers intercambiables.

Igual que features.py: una interfaz común (`MatcherBase`) y un registro
(`create_matcher("flann")`). Análisis de cada técnica en docs/03. El matcher
elige la métrica automáticamente según el dtype del descriptor: uint8 →
Hamming (binarios), float32 → L2. Así cualquier matcher clásico funciona con
cualquier extractor del registro.

Nota de diseño: la firma `match(desc_a, desc_b, kps_a, kps_b, image_shape)`
acepta keypoints porque los matchers APRENDIDOS (LightGlue/SuperGlue) razonan
sobre las POSICIONES además de los descriptores (atención espacial). Los
clásicos simplemente los ignoran — el costo de una interfaz común es cargar
con los argumentos del miembro más exigente.

─── La matemática ────────────────────────────────────────────────────────────
Distancia de Hamming:  d_H(a, b) = popcount(a XOR b)  = nº de bits en que
difieren los dos descriptores. Es la métrica natural del espacio {0,1}^n, y
el motivo de usar descriptores binarios: XOR + popcount son instrucciones de
CPU (64 bits por ciclo), así que comparar 2000×2000 candidatos es trivial.

Ratio test (Lowe, 2004). Para cada descriptor consultado, sean d1 ≤ d2 las
distancias a sus DOS vecinos más cercanos en el otro frame. Se acepta solo si

    d1 < r · d2        (r ≈ 0.75)

Lectura estadística: si el match es espurio, d1 y d2 son dos muestras de la
misma distribución de "parecido casual" → d1/d2 ≈ 1. Un match correcto no
tiene rival plausible: d1 proviene de otra distribución (mismo punto físico)
y el cociente se hunde. Elimina justo los casos ambiguos (texturas
repetitivas), que son los que forman consensos falsos en RANSAC.

Cross-check (mutuo mejor vecino): aceptar (i, j) solo si
    j = argmin_k d(a_i, b_k)   Y   i = argmin_k d(a_k, b_j).
Sin parámetros; más estricto; alternativa (no complemento) del ratio test.

FLANN (vecinos aproximados): para float, KD-trees aleatorizados con
backtracking acotado (`checks`); para binarios, LSH multi-probe — hashes que
muestrean subconjuntos de bits, de modo que colisionar ≈ estar cerca en
Hamming. Sub-lineal: solo compensa con >5k descriptores o mapas grandes.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _norm_for(desc: np.ndarray) -> int:
    """Métrica según el tipo de descriptor: binario → Hamming, float → L2."""
    return cv2.NORM_HAMMING if desc.dtype == np.uint8 else cv2.NORM_L2


def _ratio_filter(knn_pairs, ratio: float) -> List[cv2.DMatch]:
    good = [m for pair in knn_pairs if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]
    return sorted(good, key=lambda m: m.distance)


class MatcherBase(ABC):
    """Contrato de todo matcher: descriptores (± keypoints) → lista de DMatch.

    Los índices siguen la convención OpenCV: queryIdx apunta a A, trainIdx a B.
    """

    name: str = "?"

    @abstractmethod
    def match(
        self,
        desc_a: np.ndarray,
        desc_b: np.ndarray,
        kps_a: Optional[Sequence] = None,
        kps_b: Optional[Sequence] = None,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> List[cv2.DMatch]:
        """Empareja A contra B. kps/image_shape solo los usan los aprendidos."""


class RatioMatcher(MatcherBase):
    """Fuerza bruta + ratio test de Lowe (el default del repo)."""

    name = "ratio"

    def __init__(self, ratio: float = 0.75) -> None:
        self.ratio = ratio

    def match(self, desc_a, desc_b, kps_a=None, kps_b=None, image_shape=None):
        if len(desc_a) < 2 or len(desc_b) < 2:
            return []
        matcher = cv2.BFMatcher(_norm_for(desc_a))
        return _ratio_filter(matcher.knnMatch(desc_a, desc_b, k=2), self.ratio)


class CrossCheckMatcher(MatcherBase):
    """Fuerza bruta + consistencia mutua (mejor vecino en ambas direcciones)."""

    name = "crosscheck"

    def match(self, desc_a, desc_b, kps_a=None, kps_b=None, image_shape=None):
        if len(desc_a) < 1 or len(desc_b) < 1:
            return []
        matcher = cv2.BFMatcher(_norm_for(desc_a), crossCheck=True)
        return sorted(matcher.match(desc_a, desc_b), key=lambda m: m.distance)


class FlannMatcher(MatcherBase):
    """Vecinos aproximados (FLANN) + ratio test.

    Elige el índice según el descriptor: LSH para binarios, KD-trees para
    float. Con ~2k features no gana nada a fuerza bruta; está aquí por
    completitud y para el matching contra mapas grandes de v0.3.
    """

    name = "flann"
    _LSH = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    _KDTREE = dict(algorithm=1, trees=5)

    def __init__(self, ratio: float = 0.75, checks: int = 50) -> None:
        self.ratio = ratio
        self.checks = checks

    def match(self, desc_a, desc_b, kps_a=None, kps_b=None, image_shape=None):
        if len(desc_a) < 2 or len(desc_b) < 2:
            return []
        index = self._LSH if desc_a.dtype == np.uint8 else self._KDTREE
        flann = cv2.FlannBasedMatcher(index, dict(checks=self.checks))
        # FLANN exige float32 en el índice KD-tree; los binarios pasan tal cual.
        a = desc_a if desc_a.dtype == np.uint8 else np.float32(desc_a)
        b = desc_b if desc_b.dtype == np.uint8 else np.float32(desc_b)
        return _ratio_filter(flann.knnMatch(a, b, k=2), self.ratio)


# ─────────────────────────────── registro ────────────────────────────────────

def _lightglue_factory(**kwargs) -> MatcherBase:
    from vslam.frontend.learned import LightGlueMatcher
    return LightGlueMatcher(**kwargs)


_MATCHERS: Dict[str, Callable[..., MatcherBase]] = {
    "ratio": RatioMatcher,
    "crosscheck": CrossCheckMatcher,
    "flann": FlannMatcher,
    # Aprendido (requiere `pip install -e ".[deep]"`, ver learned.py):
    "lightglue": _lightglue_factory,
}


def available_matchers() -> List[str]:
    return sorted(_MATCHERS)


def create_matcher(name: str, **kwargs) -> MatcherBase:
    """Instancia un matcher del registro por nombre (ver docs/03)."""
    try:
        factory = _MATCHERS[name]
    except KeyError:
        raise ValueError(f"Matcher desconocido: {name!r}. "
                         f"Disponibles: {', '.join(available_matchers())}") from None
    return factory(**kwargs)


def match_descriptors(desc_a: np.ndarray, desc_b: np.ndarray,
                      ratio: float = 0.75) -> List[cv2.DMatch]:
    """Función retrocompatible de v0.1 (equivale a RatioMatcher)."""
    return RatioMatcher(ratio=ratio).match(desc_a, desc_b)
