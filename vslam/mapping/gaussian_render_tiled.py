"""Rasterizador 3DGS POR TILES (v0.7 hito 4, referencia escalable en PyTorch).

La referencia densa (`gaussian_render.render`) es legible pero materializa un
tensor (N, H, W, 2) → coste O(N·H·W) → OOM más allá de unos miles de gaussianas
(lección 39: revienta a 8000 gaussianas / 160×120). Esta gemela hace la MISMA
matemática pero acota la memoria como lo hace gsplat: parte la imagen en TILES y
en cada tile mezcla solo las gaussianas cuyo footprint 2D lo solapa. El pico de
memoria baja de O(N·H·W) a O(n_tile·tile²) → full-res + decenas de miles de
gaussianas en la misma GPU.

Es el mismo patrón de la regla 3 (C++ del matching, GTSAM del BA, y la gemela
gsplat que quedó pendiente por el toolchain de Windows): una referencia legible +
una gemela escalable con contrato IDÉNTICO, atada por un test de equivalencia
(test_gaussian_tiled.py). Aquí la gemela sigue siendo PyTorch puro —sin kernels
CUDA a medida— así que corre en cualquier entorno; el salto a tiempo real (kernels
por tile) es el paso gsplat cuando exista rueda binaria compatible.

─── La matemática (idéntica a la referencia, calculada por tiles) ─────────────
Proyección + covarianza EWA (Σ'=J·W·Σ·Wᵀ·Jᵀ) + α-blending front-to-back con
orden GLOBAL de profundidad: todo igual que gaussian_render.render. La única
diferencia es el ORDEN de cómputo: se proyecta y se ordena por z UNA vez (O(N)),
y el blending se hace tile a tile sobre el subconjunto de gaussianas que caen en
cada tile. Como el blending es por-píxel y los tiles particionan la imagen, el
resultado es el mismo. El culling usa un radio de ~`cutoff`·σ (σ = raíz del
autovalor mayor de Σ'); más allá de 3-4σ el peso gaussiano es <1% y se descarta
sin efecto visible (el mismo recorte que hace cualquier rasterizador de splatting).
"""

from __future__ import annotations

from typing import Tuple

import torch

from vslam.mapping.gaussian_render import DILATION, _covariance_3d

# Radio de culling en desviaciones estándar: a 3σ el peso es exp(-4.5)≈1%, a 4σ
# exp(-8)≈0.03% — recortar ahí no cambia la imagen pero acota drásticamente la
# memoria (gaussianas lejanas no entran en el tile).
CUTOFF_SIGMA = 3.5


def render(
    means: torch.Tensor,       # (N, 3) posiciones en el MUNDO
    quats: torch.Tensor,       # (N, 4) [w, x, y, z] (se normalizan)
    scales: torch.Tensor,      # (N, 3) > 0
    opacities: torch.Tensor,   # (N,) en [0, 1]
    colors: torch.Tensor,      # (N, C) en [0, 1] (gris=1 / RGB=3 / N-D)
    T_w_c: torch.Tensor,       # (4, 4) cámara→mundo (ejes OpenCV)
    K: torch.Tensor,           # (3, 3) intrínsecos
    height: int,
    width: int,
    background: float = 0.0,
    tile: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gemela por tiles de `gaussian_render.render`: MISMA firma y resultado
    (imagen (H, W, C), alpha (H, W)), diferenciable, pero con memoria acotada.
    """
    device, dtype = means.device, means.dtype
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    C = colors.shape[-1]

    # 1) Proyección mundo→cámara + pinhole (idéntico a la referencia). O(N).
    R_wc = T_w_c[:3, :3]
    t_wc = T_w_c[:3, 3]
    R_cw = R_wc.transpose(0, 1)
    mu_c = (R_cw @ (means - t_wc).T).T
    x, y, z = mu_c[:, 0], mu_c[:, 1], mu_c[:, 2]
    z_safe = torch.clamp(z, min=1e-4)
    u = fx * x / z_safe + cx
    v = fy * y / z_safe + cy
    mean2d = torch.stack([u, v], dim=-1)                 # (N, 2)

    # 2) Covarianza proyectada Σ' = J·W·Σ·Wᵀ·Jᵀ + dilatación (EWA).
    Sigma = _covariance_3d(quats, scales)
    zero = torch.zeros_like(z_safe)
    J = torch.stack([
        torch.stack([fx / z_safe, zero, -fx * x / z_safe ** 2], dim=-1),
        torch.stack([zero, fy / z_safe, -fy * y / z_safe ** 2], dim=-1),
    ], dim=1)                                            # (N, 2, 3)
    JW = J @ R_cw.unsqueeze(0)
    cov2d = JW @ Sigma @ JW.transpose(1, 2)
    cov2d = cov2d + DILATION * torch.eye(2, device=device, dtype=dtype)

    a, b = cov2d[:, 0, 0], cov2d[:, 0, 1]
    c, d = cov2d[:, 1, 0], cov2d[:, 1, 1]
    det = (a * d - b * c).clamp(min=1e-8)
    inv = torch.stack([
        torch.stack([d, -b], dim=-1),
        torch.stack([-c, a], dim=-1),
    ], dim=1) / det[:, None, None]                       # (N, 2, 2)

    # 2b) Radio de culling: cutoff·√(autovalor mayor de Σ'). λ_max de una 2×2:
    #     (tr + √(tr²−4·det))/2. Barato y diferenciable-agnóstico (solo culling).
    with torch.no_grad():
        tr = a + d
        disc = (tr * tr - 4.0 * det).clamp(min=0.0)
        lam_max = 0.5 * (tr + torch.sqrt(disc))
        radius = CUTOFF_SIGMA * torch.sqrt(lam_max.clamp(min=1e-6))   # (N,)
        visible = z > 1e-4

    # 3) Orden GLOBAL por profundidad (una sola vez). Todo lo demás se reordena.
    order = torch.argsort(z)
    mean2d, inv, opac = mean2d[order], inv[order], opacities[order]
    col = colors[order]
    radius_s, u_s, v_s, vis_s = radius[order], u[order], v[order], visible[order]

    image = torch.zeros(height, width, C, device=device, dtype=dtype)
    alpha = torch.zeros(height, width, device=device, dtype=dtype)

    # 4) Recorrido por tiles: en cada uno, solo las gaussianas que lo solapan.
    for ty in range(0, height, tile):
        y1 = min(ty + tile, height)
        for tx in range(0, width, tile):
            x1 = min(tx + tile, width)
            # Culling: bbox de la gaussiana (±radio) contra el tile.
            m = (vis_s
                 & (u_s + radius_s >= tx) & (u_s - radius_s <= x1 - 1)
                 & (v_s + radius_s >= ty) & (v_s - radius_s <= y1 - 1))
            if not bool(m.any()):
                if background != 0.0:
                    image[ty:y1, tx:x1] = background
                continue
            sm, si = mean2d[m], inv[m]
            sc, so = col[m], opac[m]

            ys, xs = torch.meshgrid(
                torch.arange(ty, y1, device=device, dtype=dtype),
                torch.arange(tx, x1, device=device, dtype=dtype), indexing="ij")
            grid = torch.stack([xs, ys], dim=-1) + 0.5        # (th, tw, 2) centros
            delta = grid[None] - sm[:, None, None, :]         # (n, th, tw, 2)
            power = -0.5 * torch.einsum("nhwi,nij,nhwj->nhw", delta, si, delta)
            g = torch.exp(power.clamp(max=0.0))
            a_i = (so[:, None, None] * g).clamp(0.0, 0.999)   # (n, th, tw)

            # α-blending front-to-back (el subconjunto ya viene en orden global).
            T = torch.cumprod(1.0 - a_i, dim=0)
            T_excl = torch.ones_like(a_i)
            T_excl[1:] = T[:-1]
            w = a_i * T_excl
            img_tile = torch.einsum("nhw,nc->hwc", w, sc)     # (th, tw, C)
            a_tile = w.sum(dim=0)                             # (th, tw)
            image[ty:y1, tx:x1] = img_tile + (1.0 - a_tile)[..., None] * background
            alpha[ty:y1, tx:x1] = a_tile

    return image, alpha
