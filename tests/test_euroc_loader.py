#!/usr/bin/env python3
"""Tests del loader EuRoC MAV (v0.45) contra un FIXTURE hecho a mano.

Valida sin necesitar el dataset (que son ~1.5 GB): el parser del sensor.yaml
(intrínsecos, distorsión, T_BS multilínea), la conversión de timestamps ns→s,
y —lo delicado— la transformación del GT del frame del CUERPO al de la CÁMARA
con el extrínseco (docs/05 §7 avisa de esta trampa).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.io.dataset import (EuRoCLoader, euroc_camera, read_euroc_groundtruth,
                              _quat_wxyz_to_R)

SENSOR_YAML = """\
sensor_type: camera
T_BS:
  cols: 4
  rows: 4
  data: [0.0148655, -0.999881, 0.00414, -0.0216401,
         0.999557, 0.0149672, 0.0257155, -0.064677,
         -0.0257744, 0.00375619, 0.999661, 0.00981073,
         0.0, 0.0, 0.0, 1.0]
rate_hz: 20
resolution: [752, 480]
camera_model: pinhole
intrinsics: [458.654, 457.296, 367.215, 248.375] #fu, fv, cu, cv
distortion_model: radial-tangential
distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
"""
T_BS_TRANS = np.array([-0.0216401, -0.064677, 0.00981073])   # brazo de palanca


def _build_fixture() -> Path:
    root = Path(tempfile.mkdtemp()) / "MH_fixture"
    cam = root / "mav0" / "cam0"
    (cam / "data").mkdir(parents=True)
    (cam / "sensor.yaml").write_text(SENSOR_YAML, encoding="utf-8")
    # Dos frames a 0.1 y 0.15 s (timestamps en ns en el CSV).
    stamps_ns = [100_000_000, 150_000_000]
    lines = ["#timestamp [ns],filename"]
    for ts in stamps_ns:
        cv2.imwrite(str(cam / "data" / f"{ts}.png"),
                    np.full((480, 752), 128, np.uint8))
        lines.append(f"{ts},{ts}.png")
    (cam / "data.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Ground truth (frame del cuerpo): timestamp, p(3), q_wxyz(4). Frame 0 con
    # orientación identidad, frame 1 rotado 90° en Z (para probar la rotación).
    gt = root / "mav0" / "state_groundtruth_estimate0"
    gt.mkdir(parents=True)
    s = np.sqrt(0.5)
    rows = [
        "#timestamp, p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z",
        f"100000000,1.0,2.0,3.0,1.0,0.0,0.0,0.0",
        f"150000000,1.0,2.0,3.0,{s},0.0,0.0,{s}",
    ]
    (gt / "data.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def test_camera_parsing():
    root = _build_fixture()
    cam = euroc_camera(root)
    assert abs(cam.fx - 458.654) < 1e-6 and abs(cam.cy - 248.375) < 1e-6
    assert cam.width == 752 and cam.height == 480
    assert cam.has_distortion and abs(cam.dist[0] + 0.28340811) < 1e-6
    assert abs(cam.dist[4]) < 1e-12, "k3 debe ser 0 (EuRoC da 4 coeficientes)"


def test_loader_timestamps_and_images():
    root = _build_fixture()
    loader = EuRoCLoader(root)
    assert len(loader) == 2
    items = list(loader)
    ts0, img0 = items[0]
    assert abs(ts0 - 0.1) < 1e-9, "timestamps deben pasar de ns a s"
    assert abs(items[1][0] - 0.15) < 1e-9
    assert img0.shape == (480, 752)


def test_groundtruth_body_to_camera():
    root = _build_fixture()
    ts, pos = read_euroc_groundtruth(root)
    assert abs(ts[0] - 0.1) < 1e-9
    # Frame 0: orientación identidad → p_cam = p_body + t_BS.
    expected0 = np.array([1.0, 2.0, 3.0]) + T_BS_TRANS
    assert np.allclose(pos[0], expected0, atol=1e-6), f"{pos[0]} vs {expected0}"
    # Frame 1: rotado → el brazo de palanca ROTA pero conserva su longitud.
    lever = np.linalg.norm(pos[1] - np.array([1.0, 2.0, 3.0]))
    assert abs(lever - np.linalg.norm(T_BS_TRANS)) < 1e-6
    # y coincide con aplicar la misma rotación al brazo.
    R = _quat_wxyz_to_R(np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)]))
    assert np.allclose(pos[1], R @ T_BS_TRANS + np.array([1.0, 2.0, 3.0]), atol=1e-6)


def main() -> int:
    test_camera_parsing()
    test_loader_timestamps_and_images()
    test_groundtruth_body_to_camera()
    print("OK: los 3 tests del loader EuRoC pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
