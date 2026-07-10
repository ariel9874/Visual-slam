#!/usr/bin/env python3
"""Tests del reconocimiento de lugar por BoW (v0.5, place_recognition.py).

Con 'lugares' sintéticos (conjuntos de descriptores base) y keyframes que son
muestras ruidosas de un lugar (bits volteados en ORB / ruido gaussiano en
float), el query debe devolver como top-1 un keyframe DEL MISMO lugar, y
puntuar más alto a los del mismo lugar que a los de lugares ajenos.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.frontend.place_recognition import BagOfVisualWords

N_PLACES, KFS_PER_PLACE, DESC_PER_KF = 6, 4, 400


def _binary_scene(seed=0):
    """Cada lugar tiene 600 descriptores ORB base; cada KF muestrea 400 y les
    voltea ~4% de los bits (el ruido típico entre re-observaciones)."""
    rng = np.random.default_rng(seed)
    places = [rng.integers(0, 256, (600, 32), dtype=np.uint8)
              for _ in range(N_PLACES)]
    kfs = []                                  # (kf_id, place, desc)
    kf_id = 0
    for p, base in enumerate(places):
        for _ in range(KFS_PER_PLACE):
            idx = rng.choice(len(base), DESC_PER_KF, replace=False)
            desc = base[idx].copy()
            flips = rng.random(desc.shape) < 0.04       # ~4% de los BYTES
            noise = rng.integers(0, 256, desc.shape, dtype=np.uint8)
            desc[flips] ^= noise[flips]
            kfs.append((kf_id, p, desc))
            kf_id += 1
    return kfs


def test_binary_retrieval_and_timing():
    kfs = _binary_scene()
    bow = BagOfVisualWords(n_words=256)
    train = np.vstack([desc for _, _, desc in kfs[:6]])
    t0 = time.perf_counter()
    bow.fit(train)
    t_fit = 1000 * (time.perf_counter() - t0)
    for kf_id, _, desc in kfs:
        bow.add(kf_id, desc)

    hits = 0
    t_q = 0.0
    for kf_id, place, desc in kfs:
        t0 = time.perf_counter()
        results = bow.query(desc, top_k=3)
        t_q += time.perf_counter() - t0
        top = [k for k, _ in results if k != kf_id][:2]
        by_place = {k: p for k, p, _ in kfs}
        hits += sum(1 for k in top if by_place[k] == place)
    recall = hits / (2 * len(kfs))
    t_q = 1000 * t_q / len(kfs)
    print(f"    [uint8] fit {t_fit:.0f} ms | query {t_q:.1f} ms/consulta "
          f"| recall top-2 mismo lugar: {recall:.2f}")
    assert recall > 0.9, f"recall {recall:.2f}: el BoW no discrimina lugares"
    assert t_q < 20, f"query demasiado lento: {t_q:.1f} ms"


def test_float_retrieval():
    rng = np.random.default_rng(3)
    places = [rng.normal(0, 1, (600, 64)).astype(np.float32)
              for _ in range(N_PLACES)]
    kfs, kf_id = [], 0
    for p, base in enumerate(places):
        for _ in range(KFS_PER_PLACE):
            idx = rng.choice(len(base), DESC_PER_KF, replace=False)
            desc = (base[idx] + rng.normal(0, 0.15, (DESC_PER_KF, 64))
                    ).astype(np.float32)
            kfs.append((kf_id, p, desc))
            kf_id += 1
    bow = BagOfVisualWords(n_words=256)
    bow.fit(np.vstack([d for _, _, d in kfs[:6]]))
    for k, _, d in kfs:
        bow.add(k, d)
    by_place = {k: p for k, p, _ in kfs}
    hits = 0
    for k, p, d in kfs:
        top = [kk for kk, _ in bow.query(d, top_k=3) if kk != k][:2]
        hits += sum(1 for kk in top if by_place[kk] == p)
    recall = hits / (2 * len(kfs))
    print(f"    [float32] recall top-2 mismo lugar: {recall:.2f}")
    assert recall > 0.9, f"recall float {recall:.2f}"


def main() -> int:
    test_binary_retrieval_and_timing()
    test_float_retrieval()
    print("OK: los 2 tests de reconocimiento de lugar (BoW) pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
