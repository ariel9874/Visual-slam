#!/usr/bin/env python3
"""Tests del GaussianSplattingMapper (v0.7), detrás de MapperBase.

(1) MULTI-VISTA: se siembra el mapa con la GEOMETRÍA correcta (como haría la
    nube dispersa del tracker) pero color DESCONOCIDO (gris), y se optimiza
    contra varios keyframes de supervisión → PSNR medio > 30 dB. Es el criterio
    de v0.7 en miniatura: renderiza y compara desde varias vistas.
(2) update_poses RÍGIDO: tras un "cierre de bucle" simulado, las gaussianas
    ancladas a un keyframe se mueven EXACTAMENTE con el delta de su pose
    (la generalización densa del re-anclaje de la nube dispersa).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


CAM = PinholeCamera(fx=80.0, fy=80.0, cx=32.0, cy=32.0, width=64, height=64)


def _poses(torch, device):
    """Cámara 0 en el origen + 4 vistas con traslación/rotación pequeñas."""
    def pose(tx, ty, ang):
        c, s = np.cos(ang), np.sin(ang)
        T = np.eye(4)
        T[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])   # yaw pequeño
        T[:3, 3] = [tx, ty, 0.0]
        return T
    specs = [(0, 0, 0), (0.25, 0, 0.05), (-0.25, 0, -0.05),
             (0, 0.2, 0.03), (0, -0.2, -0.03)]
    return [pose(*s) for s in specs]


def test_multiview_overfit_psnr():
    import torch
    from vslam.mapping.gaussian import GaussianSplattingMapper
    from vslam.mapping.gaussian_render import render
    from vslam.core.frame import Frame

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    K = torch.tensor(CAM.K, dtype=torch.float32, device=dev)
    H = W = 64

    # Escena "verdad": puntos coloreados en un frustum frente a la cámara 0.
    n = 250
    z = torch.rand(n, 1, device=dev) * 1.5 + 2.5
    means = torch.cat([(torch.rand(n, 2, device=dev) - 0.5) * 1.2 * z, z], dim=1)
    quats = torch.randn(n, 4, device=dev)
    scales = torch.rand(n, 3, device=dev) * 0.03 + 0.04
    opac = torch.full((n,), 0.9, device=dev)
    colors = torch.rand(n, 3, device=dev)

    poses = _poses(torch, dev)
    mapper = GaussianSplattingMapper(CAM, device=dev)
    for i, Tnp in enumerate(poses):
        T = torch.tensor(Tnp, dtype=torch.float32, device=dev)
        with torch.no_grad():
            img, _ = render(means, quats, scales, opac, colors, T, K, H, W)
        rgb = (img.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        mapper.integrate_keyframe(Frame(frame_id=i, timestamp=0.0,
                                        image=rgb, T_w_c=Tnp, is_keyframe=True))

    # Siembra: GEOMETRÍA correcta (nube dispersa), color desconocido (gris).
    gray = np.full((n, 3), 0.5, dtype=np.float32)
    mapper.add_points(means.cpu().numpy(), gray, anchor_kf_id=0)

    db0 = mapper.mean_psnr()
    db = mapper.optimize(iters=500)
    assert db > 30.0, f"PSNR medio {db:.1f} dB (partió de {db0:.1f}, esperado > 30)"


def test_update_poses_is_rigid():
    import torch
    from vslam.mapping.gaussian import GaussianSplattingMapper
    from vslam.core.frame import Frame

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mapper = GaussianSplattingMapper(CAM, device=dev)
    T0 = np.eye(4)
    img = np.full((64, 64), 128, np.uint8)
    mapper.integrate_keyframe(Frame(frame_id=0, timestamp=0.0, image=img,
                                    T_w_c=T0, is_keyframe=True))
    pts = np.array([[0.1, -0.2, 3.0], [0.3, 0.1, 2.5]], dtype=np.float32)
    mapper.add_points(pts, np.full((2, 3), 0.6, np.float32), anchor_kf_id=0)

    # "Bucle": el keyframe 0 se mueve con un delta rígido conocido.
    ang = 0.3
    c, s = np.cos(ang), np.sin(ang)
    T_new = np.eye(4)
    T_new[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T_new[:3, 3] = [0.5, -0.3, 0.2]
    mapper.update_poses({0: T_new})

    # D = T_new · T0⁻¹ = T_new (T0 = I). Cada media debe ir con R_D·μ + t_D.
    D = T_new @ np.linalg.inv(T0)
    expected = (D[:3, :3] @ pts.T).T + D[:3, 3]
    got = mapper.get_map()["means"]
    assert np.allclose(got, expected, atol=1e-5), f"{got} vs {expected}"
    # y la pose almacenada del keyframe se actualizó.
    assert np.allclose(mapper._kfs[0]["T"].cpu().numpy(), T_new, atol=1e-5)


def main() -> int:
    if not _has_torch():
        print("SKIP: torch no instalado.")
        return 0
    test_multiview_overfit_psnr()
    test_update_poses_is_rigid()
    print("OK: los 2 tests del GaussianSplattingMapper (v0.7) pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
