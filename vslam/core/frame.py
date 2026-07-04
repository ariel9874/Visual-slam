"""Frame: la unidad de datos que fluye entre las capas del sistema.

Este es el contrato central de la arquitectura (docs/02_arquitectura.md §4):
el frontend lo rellena, el backend refina su pose, el mapper lo consume.
Su equivalente C++ vive en cpp/include/vslam/core/frame.hpp y su equivalente
ROS 2 será vslam_msgs/Keyframe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class Frame:
    """Una imagen con sus observaciones y su pose estimada.

    Attributes:
        frame_id: identificador entero, creciente y único en la sesión.
        timestamp: tiempo de captura en segundos.
        image: imagen en escala de grises (H, W) uint8. Puede liberarse
            (None) una vez extraídas las características, para ahorrar memoria.
        keypoints: lista de cv2.KeyPoint detectados (vacía hasta que el
            frontend procese el frame).
        descriptors: array (N, 32) uint8 para descriptores binarios tipo ORB,
            o (N, D) float32 para descriptores aprendidos. None si no hay.
        T_w_c: pose 4x4 en SE(3) que transforma puntos de cámara a mundo.
            La traslación T_w_c[:3, 3] es la posición de la cámara en el mundo.
        is_keyframe: True si el frontend lo promovió a keyframe (y por tanto
            viaja al backend y al mapper).
    """

    frame_id: int
    timestamp: float
    image: Optional[np.ndarray] = None
    keypoints: Sequence = field(default_factory=list)
    descriptors: Optional[np.ndarray] = None
    T_w_c: np.ndarray = field(default_factory=lambda: np.eye(4))
    is_keyframe: bool = False

    @property
    def position(self) -> np.ndarray:
        """Posición de la cámara en el frame del mundo (3,)."""
        return self.T_w_c[:3, 3].copy()

    def release_image(self) -> None:
        """Libera la imagen cruda (los keyframes densos NO deben hacerlo:
        los mappers fotométricos —3DGS/NeRF— necesitan el color original)."""
        self.image = None
