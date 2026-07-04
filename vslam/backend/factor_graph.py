"""Interfaz del backend de optimización: grafos de factores.

Un grafo de factores expresa el SLAM como inferencia MAP: los nodos son
variables (poses de keyframes, y más adelante puntos 3D, sesgos de IMU...) y
los factores son restricciones probabilísticas entre ellas (odometría relativa,
cierres de bucle, priors). Optimizar = encontrar las variables que maximizan la
verosimilitud conjunta (mínimos cuadrados no lineales: Gauss-Newton/LM).

─── La matemática ────────────────────────────────────────────────────────────
Estimación MAP. Variables Θ = {T_1 … T_n} (poses de keyframes, en SE(3)).
Cada medida z_k tiene un modelo gaussiano p(z_k | Θ) ∝ exp(−½·‖e_k(Θ)‖²_Σk).
Maximizar el producto de factores = (tomando −log) minimizar la suma de
errores de Mahalanobis:

    Θ* = argmin_Θ  Σ_k  e_k(Θ)ᵀ · Λ_k · e_k(Θ) ,      Λ_k = Σ_k⁻¹

Λ es la MATRIZ DE INFORMACIÓN — por eso los métodos de abajo piden
`information`: grande = medida fiable que "tira" fuerte; pequeña = medida
ruidosa que apenas restringe.

Error de un factor de odometría entre las poses i → j, con medida T̂_ij:

    e_ij = Log( T̂_ij⁻¹ · T_i⁻¹ · T_j )  ∈ ℝ⁶

Léelo así: T_i⁻¹·T_j es la transformación relativa que PREDICEN las variables
actuales; se compara con la medida y el residuo se lleva al ESPACIO TANGENTE
se(3) (3 de rotación + 3 de traslación) con el logaritmo de Lie. No se restan
matrices 4×4: SE(3) es una variedad curva, no un espacio vectorial, y el
error/la actualización solo tienen sentido en su tangente.

Optimización (Gauss-Newton / Levenberg-Marquardt): linealizar cada residuo
e(Θ ⊞ δ) ≈ e + J·δ y resolver las ecuaciones normales

    (Jᵀ·Λ·J) · δ = −Jᵀ·Λ·e ,      Θ ← Θ ⊞ δ    (retracción a la variedad)

iterando hasta converger (LM añade amortiguación (Jᵀ Λ J + μI) para dar pasos
prudentes lejos del óptimo). La clave computacional: Jᵀ Λ J es DISPERSA porque
cada factor toca 1-2 variables — GTSAM/g2o explotan esa dispersión con
eliminación de variables, e iSAM2 actualiza la solución incrementalmente al
llegar cada keyframe sin re-resolver el grafo entero.

Cierre de bucle (v0.3): es solo UN factor más, entre poses lejanas en el
tiempo. Al optimizar, el error acumulado en toda la cadena de odometría se
redistribuye a lo largo de ella — así se "cose" la trayectoria. Sin bucles,
este backend únicamente alisa la odometría (y eso ya reduce el zigzag del
ejemplo 01).
──────────────────────────────────────────────────────────────────────────────

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
    """Adaptador GTSAM — TODO(v0.35).

    Nota de plataforma: PyPI no publica wheels de gtsam para Windows
    (verificado en este repo); en Windows usa la referencia NumPy
    (pose_graph.GaussNewtonPoseGraph) o instala GTSAM vía conda-forge/WSL.

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
