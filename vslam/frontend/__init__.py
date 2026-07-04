"""Capa 1 — Frontend de tracking rápido.

Responsabilidad: estimar la pose de cada frame en tiempo real y decidir qué
frames se promueven a keyframes. Todo lo costoso (BA, mapeo denso) vive en
otras capas.
"""

from vslam.frontend.features import FeatureExtractor
from vslam.frontend.matching import match_descriptors

__all__ = ["FeatureExtractor", "match_descriptors"]
