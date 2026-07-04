"""Capa 1 — Frontend de tracking rápido.

Responsabilidad: estimar la pose de cada frame en tiempo real y decidir qué
frames se promueven a keyframes. Detectores y matchers son intercambiables
por configuración (registros en features.py y matching.py; análisis
comparativo en docs/03_detectores_y_matchers.md).
"""

from vslam.frontend.features import (
    FeatureExtractor,
    FeatureExtractorBase,
    available_extractors,
    create_extractor,
)
from vslam.frontend.matching import (
    MatcherBase,
    available_matchers,
    create_matcher,
    match_descriptors,
)

__all__ = [
    "FeatureExtractor",
    "FeatureExtractorBase",
    "available_extractors",
    "create_extractor",
    "MatcherBase",
    "available_matchers",
    "create_matcher",
    "match_descriptors",
]
