"""Rasterizador 3DGS por TILES + CUDA (v0.7 hito 4): la gemela de RENDIMIENTO.

La referencia (`gaussian_render.render`) es densa —O(N·H·W)— y legible: prueba
que el contrato y la matemática son correctos, pero satura la memoria a partir
de unos pocos miles de gaussianas (lección 39: OOM a 8000 gaussianas / 160×120).
Esta gemela delega el rasterizado en **gsplat** (Nerfstudio), que agrupa las
gaussianas por TILES de píxeles y las mezcla en kernels CUDA a medida: el coste
pasa de O(N·H·W) a O(solapamientos reales), habilitando full-res + decenas de
miles de gaussianas en tiempo real.

Es el MISMO patrón que el C++ del matching o el GTSAM del BA (regla 3 de la hoja
de ruta): una referencia NumPy/PyTorch legible + una gemela rápida con contrato
IDÉNTICO, atada por un test de equivalencia (test_gaussian_gsplat.py).

─── El puente de convenciones (referencia ↔ gsplat) ──────────────────────────
gsplat comparte casi toda nuestra convención, pero con dos diferencias:

  · POSE: nosotros pasamos T_w_c (cámara→mundo); gsplat quiere `viewmats` =
    world→cam = T_w_c⁻¹. Mismos ejes OpenCV (z hacia adelante, y hacia abajo).
  · CUATERNIÓN: ambos usan [w, x, y, z] escalar-primero y los normalizan dentro.

Y una coincidencia afortunada: el desenfoque anti-aliasing de gsplat, `eps2d`,
por defecto vale 0.3 px² — EXACTAMENTE nuestro `DILATION`. Se pasa explícito
para que la equivalencia no dependa de un default de la librería.
"""

from __future__ import annotations

from typing import Tuple

import torch

from vslam.mapping.gaussian_render import DILATION


def render(
    means: torch.Tensor,       # (N, 3) posiciones en el MUNDO
    quats: torch.Tensor,       # (N, 4) [w, x, y, z] (se normalizan dentro)
    scales: torch.Tensor,      # (N, 3) > 0
    opacities: torch.Tensor,   # (N,) en [0, 1]
    colors: torch.Tensor,      # (N, C) en [0, 1] (gris=1 / RGB=3 / N-D)
    T_w_c: torch.Tensor,       # (4, 4) cámara→mundo (ejes OpenCV)
    K: torch.Tensor,           # (3, 3) intrínsecos
    height: int,
    width: int,
    background: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gemela gsplat de `gaussian_render.render`: MISMA firma, mismo resultado
    (imagen (H, W, C), alpha (H, W)), diferenciable de punta a punta. Requiere
    CUDA y gsplat instalado; el import compila los kernels en el primer uso.
    """
    from gsplat import rasterization

    viewmat = torch.inverse(T_w_c).unsqueeze(0)          # world→cam, (1, 4, 4)
    Ks = K.unsqueeze(0)                                   # (1, 3, 3)

    # gsplat rasteriza a (C_cam, H, W, canales) y alpha (C_cam, H, W, 1). Con
    # sh_degree=None trata `colors` como color crudo N-D (no armónicos esféricos);
    # eps2d=DILATION replica nuestro desenfoque; near_plane laxo como la referencia.
    # NO se usa su parámetro `backgrounds` (su forma depende del modo `packed`):
    # el fondo se compone abajo con la MISMA fórmula que la referencia.
    render_colors, render_alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=Ks,
        width=width,
        height=height,
        eps2d=DILATION,
        near_plane=1e-4,
        sh_degree=None,
        render_mode="RGB",
    )
    # Sin `backgrounds`, gsplat devuelve C = Σ c_i·a_i·T_i (sin fondo) y la alpha
    # acumulada: exactamente los dos términos de la referencia. Componer aquí
    # C + (1−α)·bg garantiza la MISMA semántica que gaussian_render.render.
    image = render_colors[0]                              # (H, W, C)
    alpha = render_alphas[0, ..., 0]                      # (H, W)
    image = image + (1.0 - alpha)[..., None] * background
    return image, alpha
