#!/usr/bin/env python3
"""Test de EQUIVALENCIA referencia ↔ gsplat (v0.7 hito 4).

La gemela rápida (`gaussian_render_gsplat.render`, tiles + CUDA) debe producir
el MISMO render que la referencia legible (`gaussian_render.render`, densa) —
igual que el BA GTSAM debe coincidir con el BA NumPy, o el matching C++ con el
de Python. Es el contrato que autoriza a usar la gemela en producción.

No son bit-idénticos: gsplat recorta cada gaussiana a ~3σ (poda por tile) y
ordena por profundidad DENTRO de cada tile, mientras la referencia mezcla todas
las gaussianas en todos los píxeles con un orden global. Ese recorte deja un
residuo de ~60 dB de PSNR mutuo (error máximo <0.01) — imperceptible.

OJO, la lección que destapó este test: al principio daba 25 dB, y el error estaba
en el NÚCLEO (alpha alta), no en las colas — luego NO era el recorte. Con una sola
gaussiana el pico de gsplat resultó ser exactamente exp(−0.5·0.5/σ²) veces el
nuestro: la firma de un desplazamiento de MEDIO PÍXEL. Nuestra referencia
muestreaba en la esquina entera (i, j) en vez de en el CENTRO del píxel
(i+0.5, j+0.5), que es la convención estándar (3DGS original, gsplat, OpenGL).
El bug era NUESTRO y lo destapó la gemela: para eso existen los tests de
equivalencia (mismo papel que NumPy↔GTSAM en el BA).

(1) EQUIVALENCIA color: misma escena → PSNR(referencia, gsplat) > 45 dB.
(2) DIFERENCIABILIDAD: el backward de gsplat corre y da gradiente no nulo en la
    media (sin él, el mapper no aprendería con la gemela).
(3) GRIS: el render de 1 canal (el que usa el mapper en TUM) también coincide.

SKIP limpio si no hay torch, CUDA, o gsplat (la gemela es opt-in; en Windows
gsplat NO enlaza —lección 40— y se corre en el contenedor docker/Dockerfile.gsplat).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _available():
    try:
        import torch
        from vslam.mapping import gaussian_render_gsplat  # noqa: F401
        import gsplat  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def _scene(torch, dev, n=120, channels=3, seed=0):
    """Gaussianas coloreadas en un frustum frente a la cámara en el origen."""
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
    from vslam.mapping.gaussian_render_gsplat import render as render_gs
    dev = "cuda"
    K, T, means, quats, scales, opac, colors = _scene(torch, dev, channels=3)
    H = W = 64
    with torch.no_grad():
        img_ref, _ = render_ref(means, quats, scales, opac, colors, T, K, H, W)
        img_gs, _ = render_gs(means, quats, scales, opac, colors, T, K, H, W)
    db = psnr(img_gs.clamp(0, 1), img_ref.clamp(0, 1))
    assert db > 45.0, f"PSNR mutuo referencia<->gsplat {db:.1f} dB (esperado > 45)"


def test_gsplat_is_differentiable():
    import torch
    from vslam.mapping.gaussian_render_gsplat import render as render_gs
    dev = "cuda"
    K, T, means, quats, scales, opac, colors = _scene(torch, dev, channels=3)
    means = means.clone().requires_grad_(True)
    H = W = 64
    img, _ = render_gs(means, quats, scales, opac, colors, T, K, H, W)
    (img ** 2).sum().backward()
    assert means.grad is not None and torch.isfinite(means.grad).all()
    assert means.grad.abs().sum() > 0, "gradiente nulo: la gemela no propaga"


def test_equivalence_grayscale():
    import torch
    from vslam.mapping.gaussian_render import render as render_ref, psnr
    from vslam.mapping.gaussian_render_gsplat import render as render_gs
    dev = "cuda"
    K, T, means, quats, scales, opac, colors = _scene(torch, dev, channels=1)
    H = W = 64
    with torch.no_grad():
        img_ref, _ = render_ref(means, quats, scales, opac, colors, T, K, H, W)
        img_gs, _ = render_gs(means, quats, scales, opac, colors, T, K, H, W)
    db = psnr(img_gs.clamp(0, 1), img_ref.clamp(0, 1))
    assert db > 45.0, f"PSNR mutuo (gris) {db:.1f} dB (esperado > 45)"


def test_pixel_center_convention():
    """El bug que destapó la gemela (ver cabecera): una gaussiana centrada en el
    punto principal, con opacidad y color 1, debe picar en exp(−0.5·0.5/σ²) — NO
    en 1.0. Es la firma de muestrear en el CENTRO del píxel (i+0.5, j+0.5): el
    pico cae entre las cuatro muestras vecinas, a media diagonal (0.5²+0.5²=0.5).
    Si alguien revierte el +0.5 de la rejilla, este test lo caza SIN gsplat."""
    import torch
    from vslam.mapping.gaussian_render import render as render_ref, DILATION
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f, c, s, z = 80.0, 32.0, 0.05, 3.0
    K = torch.tensor([[f, 0, c], [0, f, c], [0, 0, 1]], device=dev)
    T = torch.eye(4, device=dev)
    with torch.no_grad():
        img, _ = render_ref(torch.tensor([[0.0, 0.0, z]], device=dev),
                            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev),
                            torch.full((1, 3), s, device=dev),
                            torch.ones(1, device=dev),
                            torch.ones(1, 1, device=dev), T, K, 64, 64)
    sigma2 = (f * s / z) ** 2 + DILATION          # varianza proyectada + dilatación
    expected = float(torch.exp(torch.tensor(-0.5 * 0.5 / sigma2)))
    got = float(img.max())
    assert abs(got - expected) < 0.01, \
        f"pico {got:.4f} vs esperado {expected:.4f} (¿se perdió el +0.5 de centro de píxel?)"


def main() -> int:
    if not _available():
        print("SKIP: falta torch+CUDA o gsplat (gemela opt-in).")
        return 0
    test_equivalence_color()
    test_gsplat_is_differentiable()
    test_equivalence_grayscale()
    test_pixel_center_convention()
    print("OK: los 4 tests de equivalencia referencia<->gsplat (v0.7) pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
