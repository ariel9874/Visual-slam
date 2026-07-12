"""Hilo de mapeo DENSO (v0.7 hito 5): el 3DGS corre JUNTO al tracker.

El tercer hilo de la arquitectura ORB-SLAM (tracking / mapeo local / aquí, el
mapa denso): el tracker no puede pagar ni un milisegundo por el foto-realismo.
El contrato de base.py lo exige — `integrate_keyframe` no bloquea — y esta
clase lo materializa:

  - `submit(...)` (llamado por el hilo de TRACKING): copia la imagen/profundidad
    del keyframe y la ENCOLA. Coste: una copia de memoria (~µs). Nada de torch,
    nada de GPU en el hilo del tracker.
  - el WORKER (hilo propio): drena la cola (integra la vista + SIEMBRA gaussianas
    retro-proyectando la profundidad, lección 39/41) y, cuando la cola está
    vacía, gasta el presupuesto sobrante en `optimize()` por TROZOS cortos
    (chunks): entre trozo y trozo vuelve a mirar la cola, así un keyframe nuevo
    nunca espera más de un chunk (~1 s con gsplat).
  - `update_poses(...)`: proxy al mapper (re-anclaje rígido bajo su lock) — el
    driver lo llama tras un cierre de bucle o el BA global.

El criterio de v0.7 se mide con esto: fps y latencia por frame del tracking con
el hilo denso ON vs OFF (examples/08) — mismos frames procesados, presupuesto
del mapper = lo que sobra. Los refinamientos de pose/exposición y el decay del
lr (lección 41) son del PULIDO OFFLINE, no del lazo en vivo: aquí se optimiza
con lr_decay=1.0 y sin deltas por keyframe (llegarían a medio hornear).
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np

from vslam.core.frame import Frame


def _seed_from_depth(mapper, cam, kf_id: int, image: np.ndarray,
                     depth: np.ndarray, T: np.ndarray, seed_step: int,
                     depth_min: float, depth_max: float) -> None:
    """Siembra por retro-proyección de una rejilla (lección 39: la nube dispersa
    es demasiado rala) con escala = huella de la celda step·z/fx (lección 41).
    Función de módulo: la comparten el hilo y el proceso."""
    h, w = depth.shape
    gv, gu = np.mgrid[0:h:seed_step, 0:w:seed_step]
    gu, gv = gu.ravel(), gv.ravel()
    z = depth[gv, gu].astype(np.float32)
    ok = (z > depth_min) & (z < depth_max)
    if not ok.any():
        return
    u, v, z = gu[ok], gv[ok], z[ok]
    Xc = np.stack([(u - cam.cx) / cam.fx * z,
                   (v - cam.cy) / cam.fy * z, z], axis=1)
    pos = (T[:3, :3] @ Xc.T).T + T[:3, 3]
    col = image[v, u].astype(np.float32) / 255.0
    mapper.add_points(pos, col[:, None], kf_id, scales=seed_step * z / cam.fx)


class DenseMappingThread:
    """Envuelve un mapper denso (GaussianSplattingMapper) en su propio hilo."""

    def __init__(self, mapper, camera, seed_step: int = 4,
                 chunk_iters: int = 50, max_gaussians: int = 500000,
                 depth_min: float = 0.1, depth_max: float = 10.0,
                 yield_s: float = 0.01) -> None:
        self.mapper = mapper
        self.camera = camera
        self.seed_step = seed_step
        self.chunk_iters = chunk_iters
        self.max_gaussians = max_gaussians
        self.depth_min, self.depth_max = depth_min, depth_max
        self.yield_s = yield_s           # pausa entre chunks: cede GIL/CPU
        self.integrated = 0              # keyframes consumidos por el worker
        self.opt_iters = 0               # iteraciones de optimize ya gastadas
        self.failures = 0                # excepciones capturadas del worker
        self._queue: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="vslam-dense-mapping")
        self._thread.start()

    # ── hilo de TRACKING (barato) ────────────────────────────────────────────

    def submit(self, kf_id: int, image: np.ndarray,
               depth: Optional[np.ndarray], T_w_c: np.ndarray) -> None:
        """Encola un keyframe. Solo copia buffers: el tracker sigue a lo suyo."""
        self._queue.put(("kf", (int(kf_id), np.copy(image),
                                None if depth is None else np.copy(depth),
                                np.copy(T_w_c))))

    def update_poses(self, optimized_poses) -> None:
        """Re-anclaje tras bucle/GBA. Se SERIALIZA por la cola del worker: el
        re-anclaje muta los tensores del mapa y ejecutarlo desde este hilo
        chocaría con un optimize() en vuelo (acceso CUDA ilegal — carrera real
        cazada por el ejemplo 08). En la cola, corre entre chunk y chunk."""
        self._queue.put(("poses", {int(k): np.copy(v)
                                   for k, v in dict(optimized_poses).items()}))

    def stop(self, timeout: float = 30.0) -> dict:
        """Drena lo pendiente, para el worker y devuelve las estadísticas
        finales (misma forma que DenseMappingProcess.stop)."""
        self._stop.set()
        self._thread.join(timeout)
        return {"integrated": self.integrated, "opt_iters": self.opt_iters,
                "n_gaussians": len(self.mapper._means),
                "psnr": self.mapper.mean_psnr() if self.integrated else 0.0,
                "failures": self.failures}

    # ── WORKER (hilo propio: integra, siembra, optimiza) ─────────────────────

    def _seed_from_depth(self, kf_id: int, image: np.ndarray,
                         depth: np.ndarray, T: np.ndarray) -> None:
        _seed_from_depth(self.mapper, self.camera, kf_id, image, depth, T,
                         self.seed_step, self.depth_min, self.depth_max)

    def _worker(self) -> None:
        # PRESUPUESTO (lección 42): torch usa por defecto TODOS los cores para
        # su trabajo de CPU — el worker estrangulaba al tracking (mediana 36→67
        # ms). El mapa denso vive en la GPU: 1 core de CPU le basta para lanzar
        # kernels, y el sleep entre chunks cede el GIL al hilo de tracking.
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass
        while True:
            try:
                tag, payload = self._queue.get(timeout=0.05)
            except queue.Empty:
                if self._stop.is_set():
                    return
                # Cola vacía: presupuesto sobrante → un trozo de optimización.
                try:
                    if self.integrated and len(self.mapper._means):
                        self.mapper.optimize(iters=self.chunk_iters,
                                             lr_decay=1.0,
                                             max_gaussians=self.max_gaussians)
                        self.opt_iters += self.chunk_iters
                        if self.yield_s:
                            import time
                            time.sleep(self.yield_s)
                except Exception:
                    self.failures += 1
                continue
            try:
                if tag == "poses":                    # re-anclaje serializado
                    self.mapper.update_poses(payload)
                    continue
                kf_id, image, depth, T = payload
                self.mapper.integrate_keyframe(
                    Frame(frame_id=kf_id, timestamp=0.0, image=image, T_w_c=T,
                          is_keyframe=True))
                if depth is not None:
                    self._seed_from_depth(kf_id, image, depth, T)
                self.integrated += 1
            except Exception:
                self.failures += 1


# ── PROCESO de mapeo denso (lección 42: el GIL) ──────────────────────────────
# El hilo NO basta en Python: cada iter de optimize hace cientos de llamadas
# Python→torch que retienen el GIL, y el tracking (Python+numpy) lo necesita
# constantemente → medido en fr1/desk: mediana 36→64 ms (2×) con el hilo, aunque
# el worker use 1 core y duerma entre chunks. La solución es la de MonoGS: un
# PROCESO (base.py ya lo decía: "hilo/proceso"). Los keyframes viajan por
# mp.Queue (~77 KB a 320×240 — trivial) y torch/CUDA viven SOLO en el hijo
# (contexto spawn: el padre nunca inicializa CUDA).


def _dense_process_worker(q_in, q_out, cam_params, backend, seed_step,
                          chunk_iters, max_gaussians, depth_min, depth_max):
    """Bucle del PROCESO hijo. Importa torch aquí (no en el padre)."""
    import queue as _queue
    import torch
    from vslam.core.camera import PinholeCamera
    from vslam.mapping.gaussian import GaussianSplattingMapper

    torch.set_num_threads(1)     # el mapa vive en la GPU: 1 core para lanzar
    #                              kernels; los demás quedan para el tracking.
    fx, fy, cx, cy, w, h = cam_params
    cam = PinholeCamera(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)
    mapper = GaussianSplattingMapper(cam, backend=backend)
    integrated = opt_iters = failures = 0
    while True:
        try:
            tag, payload = q_in.get(timeout=0.05)
        except _queue.Empty:
            try:
                if integrated and len(mapper._means):
                    mapper.optimize(iters=chunk_iters, lr_decay=1.0,
                                    max_gaussians=max_gaussians)
                    opt_iters += chunk_iters
            except Exception:
                failures += 1
            continue
        try:
            if tag == "stop":       # la cola ya se drenó (FIFO): reporte final
                q_out.put({"integrated": integrated, "opt_iters": opt_iters,
                           "n_gaussians": len(mapper._means),
                           "psnr": mapper.mean_psnr() if integrated else 0.0,
                           "failures": failures})
                return
            if tag == "poses":
                mapper.update_poses(payload)
                continue
            kf_id, image, depth, T = payload
            mapper.integrate_keyframe(Frame(frame_id=kf_id, timestamp=0.0,
                                            image=image, T_w_c=T,
                                            is_keyframe=True))
            if depth is not None:
                _seed_from_depth(mapper, cam, kf_id, image, depth, T,
                                 seed_step, depth_min, depth_max)
            integrated += 1
        except Exception:
            failures += 1


class DenseMappingProcess:
    """Mapeo denso en PROCESO propio: misma interfaz que DenseMappingThread
    (submit/update_poses/stop), pero sin compartir GIL con el tracking. `stop()`
    drena la cola y devuelve las estadísticas finales del hijo (dict)."""

    def __init__(self, camera, backend: str = "gsplat", seed_step: int = 4,
                 chunk_iters: int = 50, max_gaussians: int = 500000,
                 depth_min: float = 0.1, depth_max: float = 10.0) -> None:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")            # el padre no toca CUDA
        self._q_in = ctx.Queue()
        self._q_out = ctx.Queue()
        cam_params = (camera.fx, camera.fy, camera.cx, camera.cy,
                      camera.width, camera.height)
        self._proc = ctx.Process(
            target=_dense_process_worker,
            args=(self._q_in, self._q_out, cam_params, backend, seed_step,
                  chunk_iters, max_gaussians, depth_min, depth_max),
            daemon=True, name="vslam-dense-mapping")
        self._proc.start()

    def submit(self, kf_id: int, image: np.ndarray,
               depth: Optional[np.ndarray], T_w_c: np.ndarray) -> None:
        self._q_in.put(("kf", (int(kf_id), np.copy(image),
                               None if depth is None else np.copy(depth),
                               np.copy(T_w_c))))

    def update_poses(self, optimized_poses) -> None:
        self._q_in.put(("poses", {int(k): np.copy(v)
                                  for k, v in dict(optimized_poses).items()}))

    def stop(self, timeout: float = 120.0) -> dict:
        self._q_in.put(("stop", None))
        stats = self._q_out.get(timeout=timeout)
        self._proc.join(10.0)
        return stats
