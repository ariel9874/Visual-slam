"""Tests del registro de frontends: todo extractor/matcher clásico debe
detectar, describir y emparejar sobre una escena sintética con textura.

Ejecutar:  pytest tests/  (o directamente: python tests/test_frontends.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.frontend.features import available_extractors, create_extractor
from vslam.frontend.matching import available_matchers, create_matcher

CLASSICAL_EXTRACTORS = ["orb", "gftt-orb", "brisk", "akaze", "sift", "kaze"]
CLASSICAL_MATCHERS = ["ratio", "crosscheck", "flann"]


def _textured_image(w: int = 640, h: int = 480, seed: int = 5) -> np.ndarray:
    """Ruido multi-escala: rico en esquinas a varias frecuencias (como el
    generador de secuencias sintéticas)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w))
    # Octavas del generador de secuencias + una de 4 px: sin el warpado en
    # perspectiva (que re-muestrea y afila la textura), los detectores
    # conservadores (BRISK/AKAZE) necesitan detalle fino explícito.
    for cell, weight in [(160, 1.0), (60, 0.8), (24, 0.6), (8, 0.5), (4, 0.4)]:
        octave = rng.uniform(0, 1, (h // cell + 2, w // cell + 2))
        img += weight * cv2.resize(octave, (w, h), interpolation=cv2.INTER_CUBIC)
    img = (img - img.min()) / (img.max() - img.min())
    return (30 + img * 195).astype(np.uint8)


def test_registry_lists_everything():
    for name in CLASSICAL_EXTRACTORS + ["superpoint", "disk"]:
        assert name in available_extractors(), name
    for name in CLASSICAL_MATCHERS + ["lightglue"]:
        assert name in available_matchers(), name


def test_classical_extractors_detect_and_describe():
    img = _textured_image()
    for name in CLASSICAL_EXTRACTORS:
        ext = create_extractor(name)
        kps, desc = ext.detect_and_compute(img)
        assert len(kps) > 50, f"{name}: solo {len(kps)} keypoints"
        assert len(desc) == len(kps), name
        expected = np.uint8 if ext.descriptor_type == "binary" else np.float32
        assert desc.dtype == expected, f"{name}: dtype {desc.dtype}"


def test_classical_matchers_recover_known_shift():
    """Desplazamos la imagen 5 px: todo matcher debe encontrar muchos matches
    y su desplazamiento mediano debe ser ~(5, 0)."""
    img_a = _textured_image()
    M = np.float32([[1, 0, 5.0], [0, 1, 0.0]])
    img_b = cv2.warpAffine(img_a, M, (img_a.shape[1], img_a.shape[0]))

    ext = create_extractor("orb")
    kps_a, desc_a = ext.detect_and_compute(img_a)
    kps_b, desc_b = ext.detect_and_compute(img_b)

    for name in CLASSICAL_MATCHERS:
        matcher = create_matcher(name)
        matches = matcher.match(desc_a, desc_b, kps_a, kps_b, img_a.shape)
        assert len(matches) > 100, f"{name}: solo {len(matches)} matches"
        dx = np.median([kps_b[m.trainIdx].pt[0] - kps_a[m.queryIdx].pt[0]
                        for m in matches])
        dy = np.median([kps_b[m.trainIdx].pt[1] - kps_a[m.queryIdx].pt[1]
                        for m in matches])
        assert abs(dx - 5.0) < 1.0 and abs(dy) < 1.0, f"{name}: ({dx:.2f}, {dy:.2f})"


def test_float_descriptors_match_with_l2():
    """SIFT (float) debe funcionar con los mismos matchers (métrica auto: L2)."""
    img = _textured_image()
    ext = create_extractor("sift")
    kps, desc = ext.detect_and_compute(img)
    matches = create_matcher("ratio").match(desc, desc)
    # Emparejada consigo misma, la gran mayoría debe casar a distancia ~0.
    assert len(matches) > 0.5 * len(kps)


def test_unknown_names_fail_clearly():
    for factory, bad in ((create_extractor, "sirf"), (create_matcher, "rateo")):
        try:
            factory(bad)
            raise AssertionError("debió lanzar ValueError")
        except ValueError as exc:
            assert "Disponibles" in str(exc)


if __name__ == "__main__":
    test_registry_lists_everything()
    test_classical_extractors_detect_and_describe()
    test_classical_matchers_recover_known_shift()
    test_float_descriptors_match_with_l2()
    test_unknown_names_fail_clearly()
    print("OK: los 5 tests del registro de frontends pasan.")
