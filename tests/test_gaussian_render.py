#!/usr/bin/env python3
"""Tests del rasterizador de Gaussian Splatting diferenciable (v0.7).

(1) PROYECCIÓN: una gaussiana en el eje óptico cae en el punto principal.
(2) GRADIENTE: el jacobiano de autograd respecto a la media coincide con
    diferencias finitas — la prueba de que la cadena proyección→covarianza→
    blending está bien derivada (si el render no fuera diferenciable de verdad,
    el mapper no aprendería nada).
(3) SOBREAJUSTE: partiendo de gaussianas aleatorias, el descenso de gradiente
    re-sintetiza una imagen objetivo hasta PSNR > 30 dB — el mismo umbral del
    criterio de v0.7, aquí sobre una sola vista (el rasterizador funciona).
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


def _camera(torch, device, dtype, f=60.0, c=24.0):
    K = torch.tensor([[f, 0, c], [0, f, c], [0, 0, 1]], device=device, dtype=dtype)
    T = torch.eye(4, device=device, dtype=dtype)          # cámara en el origen
    return K, T


def test_projection_center():
    import torch
    from vslam.mapping.gaussian_render import render
    dev, dt = "cpu", torch.float64
    K, T = _camera(torch, dev, dt)
    means = torch.tensor([[0.0, 0.0, 3.0]], device=dev, dtype=dt)   # eje óptico
    quats = torch.tensor([[1.0, 0, 0, 0]], device=dev, dtype=dt)
    scales = torch.full((1, 3), 0.08, device=dev, dtype=dt)
    opac = torch.ones(1, device=dev, dtype=dt)
    col = torch.ones(1, 3, device=dev, dtype=dt)
    img, alpha = render(means, quats, scales, opac, col, T, K, 48, 48)
    # El pico de alpha debe estar en el punto principal (cx, cy) = (24, 24).
    peak = torch.argmax(alpha)
    py, px = int(peak // 48), int(peak % 48)
    assert abs(px - 24) <= 1 and abs(py - 24) <= 1, f"pico en ({px},{py})"
    assert alpha.max() > 0.5


def test_gradient_matches_finite_difference():
    import torch
    from vslam.mapping.gaussian_render import render
    dev, dt = "cpu", torch.float64
    K, T = _camera(torch, dev, dt)
    means = torch.tensor([[0.2, -0.1, 3.0], [-0.3, 0.15, 2.5]],
                         device=dev, dtype=dt, requires_grad=True)
    quats = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], device=dev, dtype=dt)
    scales = torch.full((2, 3), 0.1, device=dev, dtype=dt)
    opac = torch.full((2,), 0.8, device=dev, dtype=dt)
    col = torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.5, 0.9]], device=dev, dtype=dt)

    def loss_of(m):
        img, _ = render(m, quats, scales, opac, col, T, K, 32, 32)
        return (img ** 2).sum()

    loss_of(means).backward()
    grad = means.grad.clone()
    eps = 1e-6
    for i in range(2):
        for j in range(3):
            d = torch.zeros_like(means)
            d[i, j] = eps
            num = (loss_of((means + d).detach()) - loss_of((means - d).detach())) / (2 * eps)
            assert abs(float(num) - float(grad[i, j])) < 1e-3, \
                f"grad[{i},{j}] auto {grad[i,j]:.5f} vs fd {float(num):.5f}"


def test_overfit_reaches_high_psnr():
    import torch
    from vslam.mapping.gaussian_render import render, psnr
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32
    torch.manual_seed(0)
    K, T = _camera(torch, dev, dt)
    H = Wd = 48

    def frustum_gaussians(n):
        z = torch.rand(n, 1, device=dev) * 2.0 + 2.0        # z ∈ [2, 4]
        xy = (torch.rand(n, 2, device=dev) - 0.5) * 1.6      # dentro del cuadro
        return torch.cat([xy * z, z], dim=1)

    # Objetivo: render de un conjunto "verdad" de gaussianas.
    with torch.no_grad():
        n_t = 60
        tg_means = frustum_gaussians(n_t)
        tg_quats = torch.randn(n_t, 4, device=dev)
        tg_scales = torch.rand(n_t, 3, device=dev) * 0.08 + 0.06
        tg_opac = torch.full((n_t,), 0.9, device=dev)
        tg_col = torch.rand(n_t, 3, device=dev)
        target, _ = render(tg_means, tg_quats, tg_scales, tg_opac, tg_col, T, K, H, Wd)
        target = target.clamp(0, 1)

    # Modelo: gaussianas aleatorias, todos los parámetros optimizables.
    n = 300
    means = frustum_gaussians(n).requires_grad_(True)
    quats = torch.randn(n, 4, device=dev).requires_grad_(True)
    log_scales = torch.log(torch.full((n, 3), 0.07, device=dev)).requires_grad_(True)
    opac_logit = torch.full((n,), 1.0, device=dev).requires_grad_(True)
    col_logit = torch.zeros(n, 3, device=dev).requires_grad_(True)
    opt = torch.optim.Adam([
        {"params": [means], "lr": 0.008},
        {"params": [quats], "lr": 0.01},
        {"params": [log_scales], "lr": 0.01},
        {"params": [opac_logit], "lr": 0.05},
        {"params": [col_logit], "lr": 0.03},
    ])
    for _ in range(1500):
        opt.zero_grad()
        img, _ = render(means, quats, torch.exp(log_scales),
                        torch.sigmoid(opac_logit), torch.sigmoid(col_logit),
                        T, K, H, Wd)
        loss = torch.abs(img - target).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        img, _ = render(means, quats, torch.exp(log_scales),
                        torch.sigmoid(opac_logit), torch.sigmoid(col_logit),
                        T, K, H, Wd)
        db = psnr(img.clamp(0, 1), target)
    assert db > 30.0, f"PSNR {db:.1f} dB tras sobreajuste (esperado > 30)"


def main() -> int:
    if not _has_torch():
        print("SKIP: torch no instalado.")
        return 0
    test_projection_center()
    test_gradient_matches_finite_difference()
    test_overfit_reaches_high_psnr()
    print("OK: los 3 tests del rasterizador de gaussianas (v0.7) pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
