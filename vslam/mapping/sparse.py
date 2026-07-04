"""Mapa disperso de puntos: la primera implementación real de MapperBase.

El mapa es la memoria del sistema: puntos 3D con su descriptor, contra los
que el tracker hace matching 3D-2D (PnP) en cada frame. Cada punto queda
ANCLADO al keyframe que lo creó — ese anclaje es lo que permite implementar
`update_poses()` honestamente: si el backend corrige la pose de un keyframe
(p. ej. tras un cierre de bucle en v0.3), sus puntos se re-anclan con la
misma corrección rígida:

    p'  =  T_nuevo · T_viejo⁻¹ · p        (delta de la pose del keyframe ancla)

Es la versión dispersa de la estrategia de "submapas rígidos" que usan los
sistemas de Gaussian Splatting para deformar el mapa tras un bucle (docs/01 §3.2).

v0.2 deliberadamente simple: sin culling de puntos, sin fusión de duplicados,
sin descriptor representativo multi-vista (TODOs para v0.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from vslam.core.frame import Frame
from vslam.core.geometry import invert_se3
from vslam.mapping.base import MapperBase


class SparsePointMapper(MapperBase):
    """Almacén de puntos 3D + descriptores, anclados a keyframes."""

    def __init__(self) -> None:
        self._positions: List[np.ndarray] = []    # (3,) por punto, frame mundo
        self._descriptors: List[np.ndarray] = []  # (D,) por punto
        self._anchor_kf: List[int] = []           # keyframe que creó cada punto
        self._kf_poses: Dict[int, np.ndarray] = {}

    # ── escritura (la llama el tracker) ──────────────────────────────────────

    def integrate_keyframe(self, keyframe: Frame) -> None:
        """Registra la pose del keyframe (necesaria para re-anclar en
        update_poses). Barato: cumple el contrato de no bloquear."""
        self._kf_poses[keyframe.frame_id] = keyframe.T_w_c.copy()

    def add_points(self, positions: np.ndarray, descriptors: np.ndarray,
                   anchor_kf_id: int) -> List[int]:
        """Añade puntos triangulados. Devuelve sus ids (índices estables)."""
        start = len(self._positions)
        for p, d in zip(positions, descriptors):
            self._positions.append(np.asarray(p, dtype=np.float64))
            self._descriptors.append(np.asarray(d))
            self._anchor_kf.append(anchor_kf_id)
        return list(range(start, len(self._positions)))

    # ── lectura (matching 3D-2D del tracker) ─────────────────────────────────

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        """(positions (M, 3), descriptors (M, D)) para el matching por frame.

        v0.2 devuelve TODO el mapa (a esta escala, miles de puntos, el BF
        matching lo absorbe). v0.3: "mapa local" por covisibilidad, como
        ORB-SLAM, para que el costo no crezca con el recorrido.
        """
        if not self._positions:
            dim = 32
            return np.zeros((0, 3)), np.zeros((0, dim), dtype=np.uint8)
        return np.stack(self._positions), np.stack(self._descriptors)

    def __len__(self) -> int:
        return len(self._positions)

    # ── contrato MapperBase ───────────────────────────────────────────────────

    def update_poses(self, optimized_poses: Dict[int, np.ndarray]) -> None:
        """Re-ancla los puntos cuyos keyframes fueron corregidos (fórmula arriba)."""
        deltas = {}
        for kf_id, T_new in optimized_poses.items():
            T_old = self._kf_poses.get(kf_id)
            if T_old is not None:
                deltas[kf_id] = T_new @ invert_se3(T_old)
                self._kf_poses[kf_id] = T_new.copy()
        for i, kf_id in enumerate(self._anchor_kf):
            delta = deltas.get(kf_id)
            if delta is not None:
                p = self._positions[i]
                self._positions[i] = delta[:3, :3] @ p + delta[:3, 3]

    def get_map(self) -> np.ndarray:
        """Nube de puntos (M, 3) en el frame del mundo."""
        return self.snapshot()[0]

    # ── exportación ───────────────────────────────────────────────────────────

    def save_ply(self, path: str | Path) -> None:
        """Guarda la nube en PLY ASCII (abrible en MeshLab/CloudCompare)."""
        pts = self.get_map()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = ("ply\nformat ascii 1.0\n"
                  f"element vertex {len(pts)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "end_header\n")
        body = "\n".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in pts)
        path.write_text(header + body + ("\n" if len(pts) else ""), encoding="utf-8")
