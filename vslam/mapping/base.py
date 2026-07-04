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

─── La matemática de cada representación ─────────────────────────────────────
Mapa disperso (v0.2) — triangulación lineal (DLT). Un punto X̄ (homogéneo,
4x1) visto por una cámara con matriz de proyección P = K·[R|t] cumple
λ·x̂ = P·X̄. La escala λ se elimina con el producto vectorial:

    [x̂]_× · P · X̄ = 0     → 2 ecuaciones lineales independientes por vista

Apilando ≥ 2 vistas queda A·X̄ = 0: X̄ es el vector singular asociado al menor
valor singular de A (SVD). Es la solución ALGEBRAICA — el refinamiento que
minimiza el error de reproyección de verdad se hace después, en el bundle
adjustment del backend. La calidad depende del ángulo de paralaje: rayos casi
paralelos (baseline pequeño) → profundidad mal condicionada.

Mapa denso diferenciable (v0.5+) — "renderiza y compara". El mapa es un
conjunto de parámetros G que se ajusta para re-sintetizar los keyframes:

    G* = argmin_G  Σ_kf ‖ I_kf − Render(G, T_kf, K) ‖   (+ regularizadores)

· 3D Gaussian Splatting: G = {(μ_i, Σ_i, α_i, c_i)} — media, covarianza
  (Σ = R·S·Sᵀ·Rᵀ: rotación por escalas, siempre definida positiva), opacidad
  y color de cada gaussiana. Se proyectan a 2D (Σ' = J·W·Σ·Wᵀ·Jᵀ, con J el
  jacobiano de la proyección) y el color de cada píxel es el α-blending de
  las gaussianas ordenadas por profundidad:

      C = Σ_i  c_i · α_i · Π_{j<i} (1 − α_j)

  Todo es diferenciable respecto a G → descenso de gradiente directo sobre el
  mapa, y rasterizar es rápido porque cada gaussiana toca pocos píxeles.

· NeRF: G = pesos de un campo F_G(x, d) → (densidad σ, color c). El color de
  un rayo r(s) = o + s·d es la integral de render volumétrico

      C = ∫ T(s)·σ(r(s))·c(r(s), d) ds ,     T(s) = exp(−∫₀ˢ σ)

  aproximada muestreando decenas de puntos POR RAYO y evaluando la red en
  cada uno — de ahí que NeRF sea órdenes de magnitud más caro de renderizar
  que 3DGS, y que este repo apueste por gaussianas para el mapper denso.
──────────────────────────────────────────────────────────────────────────────
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
