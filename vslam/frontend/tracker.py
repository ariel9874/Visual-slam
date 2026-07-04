"""Interfaz del tracker: el contrato de la capa de tracking.

La implementación de referencia (didáctica, 2D-2D con matriz esencial) vive en
examples/01_monocular_vo.py para poder leerse de arriba a abajo en un archivo.
Cuando estabilicemos v0.2 (tracking 3D-2D con PnP contra el mapa local), la
lógica migrará aquí detrás de esta interfaz.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from vslam.core.frame import Frame


class TrackerBase(ABC):
    """Contrato: recibe frames, devuelve poses; decide keyframes.

    Implementaciones previstas:
      - EssentialMatrixTracker (v0.1, referencia en examples/01): 2D-2D puro.
      - PnPTracker (v0.2): triangula puntos y trackea 3D-2D — menos deriva.
      - KLTTracker (v0.4, C++): flujo óptico piramidal, más rápido que re-detectar.
    """

    @abstractmethod
    def process(self, frame: Frame) -> Optional[np.ndarray]:
        """Estima y escribe frame.T_w_c. Devuelve la pose, o None si el
        tracking se perdió (el llamador decide: relocalizar o reiniciar)."""
        raise NotImplementedError
