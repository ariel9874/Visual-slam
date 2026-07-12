#!/usr/bin/env python3
"""Test de EQUIVALENCIA referencia densa ↔ rasterizador por tiles (v0.7 hito 4).

El rasterizador por tiles (`gaussian_render_tiled.render`) hace la MISMA
matemática que la referencia densa (`gaussian_render.render`) pero acota la
memoria procesando la imagen tile a tile. Debe dar el MISMO render — igual que el
BA GTSAM ≡ BA NumPy o el matching C++ ≡ Python. Es el contrato que autoriza a
usar el por-tiles como render de trabajo (rompe el techo O(N·H·W) de la lección 39).

No es bit-idéntico: el por-tiles recorta cada gaussiana a ~3.5σ (culling por tile),
donde el peso gaussiano ya es <0.1%. En una escena normal eso coincide con el
denso a >40 dB de PSNR mutuo.

(1) EQUIVALENCIA color: misma escena → PSNR(denso, tiles) muy alto.
(2) DIFERENCIABILIDAD: el backward por tiles da gradiente finito y no nulo.
(3) GRIS: el render de 1 canal (el del mapper en TUM) también coincide.

Corre en CPU (no requiere CUDA); SKIP solo si falta torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _scene(torch, dev, n=200, channels=3, seed=0):
    torch.manual_seed(seed)
    K = torch.tensor([[80.0, 0, 32], [0, 80.0, 32], [0, 0, 1]],
                     device=dev, dtype=torch.float32)
    T = torch.eye(4, device=dev, dtype=torch.float32)
    z = torch.rand(n, 1, device=dev) * 1.5 + 2.5
    means = torch.cat([(torch.rand(n, 2, device=dev) - 0.5) * 1.0 * z, z], dim=1)
    quats = torch.randn(n, 4, device=dev)
    scales = torch.rand(n, 3, device=dev) * 0.03 + 0.05
    opac = torch.full((n,), 0.9, device=dev)
    colors = torch.rand(n, channels, device=dev)
    return K, T, means, quats, scales, opac, colors


def test_equivalence_color():
    import torch
    from vslam.mapping.gaussian_render import render as render_ref, psnr
    from vslam.mapping.gaussian_render_tiled import render as render_tiled
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K, T, means, quats, scales, opac, colors = _scene(torch, dev, channels=3)
    H = W = 64
    with torch.no_grad():
        img_ref, _ = render_ref(means, quats, scales, opac, colors, T, K, H, W)
        img_t, _ = render_tiled(means, quats, scales, opac, colors, T, K, H, W, tile=16)
    db = psnr(img_t.clamp(0, 1), img_ref.clamp(0, 1))
    assert db > 40.0, f"PSNR mutuo denso<->tiles {db:.1f} dB (esperado > 40)"


def test_tiled_is_differentiable():
    import torch
    from vslam.mapping.gaussian_render_tiled import render as render_tiled
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K, T, means, quats, scales, opac, colors = _scene(torch, dev, channels=3)
    means = means.clone().requires_grad_(True)
    H = W = 64
    img, _ = render_tiled(means, quats, scales, opac, colors, T, K, H, W, tile=16)
    (img ** 2).sum().backward()
    assert means.grad is not None and torch.isfinite(means.grad).all()
    assert means.grad.abs().sum() > 0, "gradiente nulo: el por-tiles no propaga"


def test_equivalence_grayscale():
    import torch
    from vslam.mapping.gaussian_render import render as render_ref, psnr
    from vslam.mapping.gaussian_render_tiled import render as render_tiled
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K, T, means, quats, scales, opac, colors = _scene(torch, dev, channels=1)
    H = W = 64
    with torch.no_grad():
        img_ref, _ = render_ref(means, quats, scales, opac, colors, T, K, H, W)
        img_t, _ = render_tiled(means, quats, scales, opac, colors, T, K, H, W, tile=16)
    db = psnr(img_t.clamp(0, 1), img_ref.clamp(0, 1))
    assert db > 40.0, f"PSNR mutuo (gris) {db:.1f} dB (esperado > 40)"


def main() -> int:
    if not _has_torch():
        print("SKIP: torch no instalado.")
        return 0
    test_equivalence_color()
    test_tiled_is_differentiable()
    test_equivalence_grayscale()
    print("OK: los 3 tests de equivalencia denso<->tiles (v0.7) pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
