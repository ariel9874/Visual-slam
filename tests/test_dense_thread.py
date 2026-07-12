#!/usr/bin/env python3
"""Tests del hilo de mapeo denso (v0.7 hito 5).

El contrato que el criterio de v0.7 exige al lazo en vivo:
(1) `submit` NO bloquea: encolar un keyframe cuesta microsegundos (una copia),
    aunque el worker esté ocupado optimizando — el tracking nunca espera.
(2) El worker CONSUME: integra las vistas, siembra gaussianas desde la
    profundidad y gasta el presupuesto sobrante en optimize() por chunks.
(3) `update_poses` re-ancla a través del proxy (bucle/GBA en caliente).

CPU y sin dataset (backend de referencia a 32×32); SKIP limpio sin torch.
"""

from __future__ import annotations

import sys
import time
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


CAM = PinholeCamera(fx=40.0, fy=40.0, cx=16.0, cy=16.0, width=32, height=32)


def _kf(i):
    rng = np.random.default_rng(i)
    image = rng.integers(0, 255, (32, 32), dtype=np.uint8)
    depth = (rng.random((32, 32), dtype=np.float32) * 1.5 + 2.0)
    T = np.eye(4)
    T[0, 3] = 0.05 * i
    return image, depth, T


def test_submit_is_cheap_and_worker_consumes():
    from vslam.mapping.gaussian import GaussianSplattingMapper
    from vslam.mapping.dense_thread import DenseMappingThread

    mapper = GaussianSplattingMapper(CAM, device="cpu")
    dense = DenseMappingThread(mapper, CAM, seed_step=4, chunk_iters=5)
    t0 = time.perf_counter()
    for i in range(6):
        image, depth, T = _kf(i)
        dense.submit(i, image, depth, T)
    dt_ms = (time.perf_counter() - t0) * 1000 / 6
    assert dt_ms < 5.0, f"submit tardó {dt_ms:.2f} ms de media (debe ser ~µs)"

    # El worker consume la cola y deja presupuesto para optimizar.
    deadline = time.time() + 30.0
    while (dense.integrated < 6 or dense.opt_iters == 0) and time.time() < deadline:
        time.sleep(0.05)
    dense.stop()
    assert dense.integrated == 6, f"integrados {dense.integrated}/6"
    assert len(mapper._means) > 0, "no sembró gaussianas desde la profundidad"
    assert dense.opt_iters > 0, "nunca optimizó con el presupuesto sobrante"
    assert dense.failures == 0, f"{dense.failures} excepciones en el worker"


def test_update_poses_through_proxy():
    from vslam.mapping.gaussian import GaussianSplattingMapper
    from vslam.mapping.dense_thread import DenseMappingThread

    mapper = GaussianSplattingMapper(CAM, device="cpu")
    dense = DenseMappingThread(mapper, CAM, seed_step=8, chunk_iters=5)
    image, depth, T0 = _kf(0)
    dense.submit(0, image, depth, T0)
    deadline = time.time() + 30.0
    while dense.integrated < 1 and time.time() < deadline:
        time.sleep(0.05)
    before = mapper.get_map()["means"].copy()

    T_new = np.eye(4)
    T_new[:3, 3] = [0.5, -0.2, 0.1]
    dense.update_poses({0: T_new})
    dense.stop()
    got = mapper.get_map()["means"]
    D = T_new @ np.linalg.inv(T0)
    expected = (D[:3, :3] @ before.T).T + D[:3, 3]
    assert np.allclose(got, expected, atol=1e-4), "el re-anclaje no pasó por el proxy"


def test_process_worker_smoke():
    """El PROCESO de mapeo (lección 42: el hilo comparte GIL con el tracking y
    duplica su latencia — medido 36→64 ms; el proceso no). Misma interfaz:
    submit/update_poses/stop; stop() drena y devuelve las estadísticas."""
    from vslam.mapping.dense_thread import DenseMappingProcess

    dense = DenseMappingProcess(CAM, backend="reference", seed_step=8,
                                chunk_iters=5)
    for i in range(2):
        image, depth, T = _kf(i)
        dense.submit(i, image, depth, T)
    dense.update_poses({0: np.eye(4)})
    stats = dense.stop(timeout=180.0)      # el hijo importa torch: darle margen
    assert stats["integrated"] == 2, stats
    assert stats["n_gaussians"] > 0, stats
    assert stats["failures"] == 0, stats


def main() -> int:
    if not _has_torch():
        print("SKIP: torch no instalado.")
        return 0
    test_submit_is_cheap_and_worker_consumes()
    test_update_poses_through_proxy()
    test_process_worker_smoke()
    print("OK: los 3 tests del mapeo denso en hilo/proceso (v0.7 hito 5) pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
