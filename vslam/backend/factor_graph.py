"""Interfaz del backend de optimización: grafos de factores.

Un grafo de factores expresa el SLAM como inferencia MAP: los nodos son
variables (poses de keyframes, y más adelante puntos 3D, sesgos de IMU...) y
los factores son restricciones probabilísticas entre ellas (odometría relativa,
cierres de bucle, priors). Optimizar = encontrar las variables que maximizan la
verosimilitud conjunta (mínimos cuadrados no lineales: Gauss-Newton/LM).

Esta interfaz aísla al resto del sistema de la librería concreta:
  - GTSAM  (primera opción: iSAM2 permite optimización incremental en línea)
  - g2o    (clásico en la literatura SLAM)
  - Ceres  (general, muy robusto)

v0.1: solo el contrato + un adaptador GTSAM esbozado. Se implementa en v0.3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class FactorGraphBackend(ABC):
    """Contrato mínimo de un backend de grafo de poses (pose graph).

    Nota de diseño: las poses entran/salen como matrices 4x4 (T_w_c, la
    convención del repo). La conversión a los tipos internos de cada librería
    (gtsam.Pose3, g2o.SE3Quat) es responsabilidad exclusiva del adaptador.
    """

    @abstractmethod
    def add_pose(self, node_id: int, T_w_c: np.ndarray, fixed: bool = False) -> None:
        """Añade una variable de pose. `fixed=True` ancla el gauge (primer keyframe)."""

    @abstractmethod
    def add_odometry_factor(
        self, id_from: int, id_to: int, T_rel: np.ndarray, information: np.ndarray
    ) -> None:
        """Factor binario de odometría: T_rel = T_from^-1 · T_to medido por el
        frontend. `information` (6x6) codifica la confianza (inversa de covarianza)."""

    @abstractmethod
    def add_loop_factor(
        self, id_from: int, id_to: int, T_rel: np.ndarray, information: np.ndarray
    ) -> None:
        """Factor de cierre de bucle (misma forma que odometría, pero conecta
        keyframes lejanos en el tiempo — es lo que elimina la deriva global).
        Conviene envolverlo en un kernel robusto (Huber/DCS) por si el
        reconocimiento de lugar se equivoca."""

    @abstractmethod
    def optimize(self, iterations: int = 20) -> Dict[int, np.ndarray]:
        """Resuelve el grafo. Devuelve {node_id: T_w_c optimizada}.

        El llamador debe propagar el resultado: corregir la pose del frontend
        y llamar a MapperBase.update_poses() para deformar el mapa."""


class GTSAMBackend(FactorGraphBackend):
    """Adaptador GTSAM — TODO(v0.3).

    Boceto de la implementación prevista (se deja como guía):
        import gtsam
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        # add_pose      -> values.insert(X(i), gtsam.Pose3(T)); prior si fixed
        # add_*_factor  -> graph.add(gtsam.BetweenFactorPose3(X(i), X(j),
        #                       gtsam.Pose3(T_rel), noise_model))
        # optimize      -> gtsam.LevenbergMarquardtOptimizer(graph, values)
        #                  (o iSAM2 para modo incremental en línea)
    """

    def add_pose(self, node_id: int, T_w_c: np.ndarray, fixed: bool = False) -> None:
        raise NotImplementedError("Planificado para v0.3 — ver docstring de la clase.")

    def add_odometry_factor(self, id_from, id_to, T_rel, information) -> None:
        raise NotImplementedError("Planificado para v0.3 — ver docstring de la clase.")

    def add_loop_factor(self, id_from, id_to, T_rel, information) -> None:
        raise NotImplementedError("Planificado para v0.3 — ver docstring de la clase.")

    def optimize(self, iterations: int = 20) -> Dict[int, np.ndarray]:
        raise NotImplementedError("Planificado para v0.3 — ver docstring de la clase.")
