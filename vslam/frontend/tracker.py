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

    ─── La matemática de cada estrategia ───
    2D-2D (v0.1): sin mapa. La pose relativa sale de la restricción epipolar
    x̂'ᵀ·E·x̂ = 0 (derivada en examples/01). Coste estructural: la escala de t
    es inobservable y TODO se re-estima en cada par de frames.

    3D-2D / PnP (v0.2): con mapa {X_i ↔ u_i}. Se minimiza el ERROR DE
    REPROYECCIÓN — la función objetivo central de todo el SLAM geométrico:

        T* = argmin_T  Σ_i  ρ( ‖ π(K, T⁻¹·X_i) − u_i ‖² )

    donde T = T_w_c (la pose buscada), π es la proyección pinhole del punto
    llevado al frame de cámara, y ρ un kernel robusto (Huber: cuadrático
    cerca de cero, lineal lejos) que evita que un outlier domine la suma.
    Como los X_i ya tienen escala, la pose la HEREDA del mapa: la deriva de
    escala frame a frame desaparece. Esta es la razón de fondo para migrar
    de 2D-2D a 3D-2D en cuanto exista un mapa.

    KLT (v0.4): en lugar de re-detectar y describir, sigue cada parche
    minimizando el error fotométrico local (Lucas-Kanade):
        Δu* = argmin_Δu Σ_parche ( I_t(u + Δu) − I_{t−1}(u) )²
    linealizado con el gradiente de imagen (Gauss-Newton sobre 2 parámetros
    por punto, en pirámide para movimientos grandes). Es la puerta de entrada
    a los métodos DIRECTOS (DSO generaliza esta idea a pose + profundidades).
    """

    @abstractmethod
    def process(self, frame: Frame) -> Optional[np.ndarray]:
        """Estima y escribe frame.T_w_c. Devuelve la pose, o None si el
        tracking se perdió (el llamador decide: relocalizar o reiniciar)."""
        raise NotImplementedError
