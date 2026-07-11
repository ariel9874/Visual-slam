#!/usr/bin/env python3
"""Tests del estéreo EuRoC (v0.6) contra un FIXTURE hecho a mano (sin el dataset
de ~1.1 GB): el rig de rectificación y la profundidad por disparidad.

(1) RIG: dos cámaras rectificadas con baseline horizontal conocido. Verifica que
    `cv2.stereoRectify` recupera el bf (= fx·b), que la cámara izquierda sale
    SIN distorsión y que rectificar un par ya rectificado es (casi) la identidad.
(2) PROFUNDIDAD: se fabrica un par de un PLANO fronto-paralelo a Z conocido
    (imagen derecha = izquierda desplazada por la disparidad entera d = bf/Z).
    StereoSGBM debe recuperar d → profundidad ≈ Z. Es el puente completo
    rig→disparidad→metros que alimenta la ruta RGB-D métrica del tracker.
(3) La identidad que une los dos hitos: la disparidad medida ES u_L − u_R, así
    que u_R = u_L − d = u_L − bf/z — el MISMO residuo que el BA RGB-D sintetiza
    desde profundidad (bundle_adjustment.py). Real vs. virtual, misma ecuación.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.io.dataset import EuRoCStereoLoader, EuRoCStereoRig

FX, FY, CX, CY, W, H = 400.0, 400.0, 320.0, 240.0, 640, 480
BASELINE = 0.10                              # 10 cm → bf = fx·b = 40 px·m


def _sensor_yaml(t_bs_data: str) -> str:
    """sensor.yaml EuRoC RECTIFICADO (sin distorsión) con el extrínseco dado."""
    return (
        "sensor_type: camera\n"
        "T_BS:\n  cols: 4\n  rows: 4\n"
        f"  data: [{t_bs_data}]\n"
        f"resolution: [{W}, {H}]\n"
        "camera_model: pinhole\n"
        f"intrinsics: [{FX}, {FY}, {CX}, {CY}]\n"
        "distortion_model: radial-tangential\n"
        "distortion_coefficients: [0.0, 0.0, 0.0, 0.0]\n"
    )


def _identity_T_BS() -> str:
    return "1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1"


def _right_T_BS() -> str:
    # Cámara derecha desplazada +baseline en X del cuerpo (= frame de la izq).
    return f"1,0,0,{BASELINE}, 0,1,0,0, 0,0,1,0, 0,0,0,1"


def _build_rig_fixture(left_imgs, right_imgs) -> Path:
    """Escribe mav0/cam0 y mav0/cam1 con las imágenes dadas (listas de arrays)."""
    root = Path(tempfile.mkdtemp()) / "V_fixture"
    for cam, tbs, imgs in (("cam0", _identity_T_BS(), left_imgs),
                           ("cam1", _right_T_BS(), right_imgs)):
        d = root / "mav0" / cam
        (d / "data").mkdir(parents=True)
        (d / "sensor.yaml").write_text(_sensor_yaml(tbs), encoding="utf-8")
        lines = ["#timestamp [ns],filename"]
        for i, img in enumerate(imgs):
            ts = 100_000_000 + i * 50_000_000
            cv2.imwrite(str(d / "data" / f"{ts}.png"), img)
            lines.append(f"{ts},{ts}.png")
        (d / "data.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_rig_recovers_baseline_and_rectifies():
    blank = np.full((H, W), 128, np.uint8)
    root = _build_rig_fixture([blank], [blank])
    rig = EuRoCStereoRig(root)

    assert abs(rig.baseline - BASELINE) < 1e-3, f"baseline {rig.baseline}"
    assert abs(rig.bf - FX * BASELINE) < 1e-1, f"bf {rig.bf} != {FX*BASELINE}"
    # La cámara izquierda rectificada es un pinhole sin distorsión, fx intacto.
    assert not rig.camera.has_distortion
    assert abs(rig.camera.fx - FX) < 1.0 and rig.camera.width == W
    # Rectificar un par YA rectificado ~ identidad: un patrón vuelve a sí mismo.
    rng = np.random.default_rng(0)
    pat = rng.integers(0, 256, (H, W), np.uint8)
    L, _ = rig.rectify(pat, pat)
    # Interior (los bordes pueden quedar fuera del mapa): diferencia pequeña.
    core = slice(40, H - 40), slice(40, W - 40)
    assert np.abs(L[core].astype(int) - pat[core].astype(int)).mean() < 2.0


def test_depth_from_disparity():
    # Plano fronto-paralelo a Z: la derecha es la izquierda desplazada d = bf/Z.
    d = 16
    Z = FX * BASELINE / d                     # = 40/16 = 2.5 m
    rng = np.random.default_rng(1)
    left = rng.integers(0, 256, (H, W), np.uint8)
    left = cv2.GaussianBlur(left, (3, 3), 0)  # suaviza el aliasing sub-píxel
    right = np.zeros_like(left)
    right[:, :W - d] = left[:, d:]            # right(c) = left(c + d)  →  disp = d

    root = _build_rig_fixture([left], [right])
    loader = EuRoCStereoLoader(root, num_disparities=32, block_size=7,
                               min_depth=0.5, max_depth=40.0)
    assert abs(loader.stereo_bf - FX * BASELINE) < 1e-1
    ts, L, depth = next(iter(loader))

    valid = depth > 0
    assert valid.mean() > 0.3, f"muy pocos píxeles válidos: {valid.mean():.2f}"
    med = float(np.median(depth[valid]))
    assert abs(med - Z) < 0.25, f"profundidad mediana {med:.2f} != {Z:.2f}"


def main() -> int:
    test_rig_recovers_baseline_and_rectifies()
    test_depth_from_disparity()
    print("OK: los 2 tests de estéreo (v0.6) pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
