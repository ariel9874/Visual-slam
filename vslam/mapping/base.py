"""Interfaz del mapper: el módulo intercambiable por excelencia del repo.

La tesis de la arquitectura (docs/01 §5) es que el mapa puede cambiar de
representación —nube dispersa, gaussianas 3D, campo neural— sin que el frontend
ni el backend se enteren. Para eso el contrato exige tres cosas:

1. `integrate_keyframe` NO puede bloquear: el mapeo denso corre en su propio
   hilo/proceso con el presupuesto que le sobre al tracking.
2. `update_poses` es obligatorio: tras un cierre de bucle el backend corrige
   las poses pasadas y el mapa DEBE deformarse en consecuencia (con gaussianas:
   transformar submapas rígidamente; con campos implícitos: problema abierto).
3. `get_map` devuelve algo exportable para visualización/evaluación.

Implementaciones previstas:
  - SparsePointMapper (v0.2): triangulación de matches entre keyframes.
  - GaussianSplattingMapper (v0.5): optimiza gaussianas 3D contra los keyframes
    por rasterización diferenciable (estilo MonoGS/Photo-SLAM).
  - NeRFMapper (futuro): campo neural con hash-grid (estilo NICE-SLAM/Co-SLAM).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from vslam.core.frame import Frame


class MapperBase(ABC):
    """Contrato común de todos los mappers."""

    @abstractmethod
    def integrate_keyframe(self, keyframe: Frame) -> None:
        """Incorpora un keyframe (con pose inicial del frontend) al mapa.
        Debe retornar rápido: el trabajo pesado se difiere/encola."""

    @abstractmethod
    def update_poses(self, optimized_poses: Dict[int, np.ndarray]) -> None:
        """El backend corrigió poses (p. ej. tras un cierre de bucle):
        {frame_id: T_w_c}. El mapper debe re-anclar su geometría."""

    @abstractmethod
    def get_map(self) -> Any:
        """Representación exportable del mapa (puntos Nx3, gaussianas, malla...)."""
