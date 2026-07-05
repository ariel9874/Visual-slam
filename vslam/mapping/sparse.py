"""Mapa disperso de puntos: la primera implementación real de MapperBase.

El mapa es la memoria del sistema: puntos 3D con su descriptor, contra los
que el tracker hace matching 3D-2D (PnP) en cada frame. Desde v0.35 también
almacena las OBSERVACIONES (keyframe, punto, píxel) — el combustible del
bundle adjustment: sin saber quién vio qué y dónde, no hay nada que refinar.

Cada punto queda ANCLADO al keyframe que lo creó — ese anclaje permite
implementar `update_poses()` honestamente: si el backend corrige la pose de
un keyframe (p. ej. tras un cierre de bucle), sus puntos se re-anclan con la
misma corrección rígida:

    p'  =  T_nuevo · T_viejo⁻¹ · p        (delta de la pose del keyframe ancla)

Es la versión dispersa de la estrategia de "submapas rígidos" que usan los
sistemas de Gaussian Splatting para deformar el mapa tras un bucle (docs/01 §3.2).

Simplificaciones deliberadas que siguen pendientes: culling de puntos, fusión
de duplicados y descriptor representativo multi-vista (TODOs para v0.4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from vslam.core.frame import Frame
from vslam.core.geometry import invert_se3
from vslam.mapping.base import MapperBase


class SparsePointMapper(MapperBase):
    """Almacén de puntos 3D + descriptores + observaciones, anclados a keyframes."""

    def __init__(self) -> None:
        self._positions: List[np.ndarray] = []    # (3,) por punto, frame mundo
        self._descriptors: List[np.ndarray] = []  # (D,) por punto
        self._anchor_kf: List[int] = []           # keyframe que creó cada punto
        self._kf_poses: Dict[int, np.ndarray] = {}
        # Observaciones por keyframe: [(point_id, píxel (2,)), ...]
        self._obs: Dict[int, List[Tuple[int, np.ndarray]]] = {}

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

    def add_observations(self, kf_id: int, point_ids: Iterable[int],
                         pixels: np.ndarray) -> None:
        """Registra que el keyframe observó estos puntos en estos píxeles."""
        entries = self._obs.setdefault(kf_id, [])
        for pid, uv in zip(point_ids, np.asarray(pixels, dtype=np.float64)):
            entries.append((int(pid), uv.copy()))

    # ── lectura (matching 3D-2D del tracker y BA del backend) ────────────────

    def snapshot(self, anchor_kfs: Optional[Iterable[int]] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(ids, positions (M, 3), descriptors (M, D)) para el matching.

        Con `anchor_kfs` devuelve solo los puntos anclados a esos keyframes:
        el "MAPA LOCAL". Es la palanca de escalabilidad clásica (el costo del
        matching deja de crecer con el recorrido) y también lo que hace que
        la deriva vuelva a existir — y con ella, la necesidad del cierre de
        bucle. Sin argumento devuelve el mapa global (comportamiento v0.2).
        """
        if not self._positions:
            return (np.zeros(0, dtype=int), np.zeros((0, 3)),
                    np.zeros((0, 32), dtype=np.uint8))
        if anchor_kfs is None:
            ids = np.arange(len(self._positions))
        else:
            wanted = set(anchor_kfs)
            ids = np.array([i for i, kf in enumerate(self._anchor_kf) if kf in wanted],
                           dtype=int)
            if len(ids) == 0:
                return (np.zeros(0, dtype=int), np.zeros((0, 3)),
                        np.zeros((0, self._descriptors[0].shape[0]),
                                 dtype=self._descriptors[0].dtype))
        return (ids,
                np.stack([self._positions[i] for i in ids]),
                np.stack([self._descriptors[i] for i in ids]))

    def observations(self, kf_ids: Iterable[int]
                     ) -> List[Tuple[int, int, np.ndarray]]:
        """[(kf_id, point_id, píxel), ...] de los keyframes pedidos (para BA)."""
        out = []
        for kf in kf_ids:
            for pid, uv in self._obs.get(kf, []):
                out.append((kf, pid, uv))
        return out

    def keyframe_pose(self, kf_id: int) -> np.ndarray:
        return self._kf_poses[kf_id].copy()

    def point_positions(self, ids: Iterable[int]) -> Dict[int, np.ndarray]:
        return {int(i): self._positions[i].copy() for i in ids}

    def __len__(self) -> int:
        return len(self._positions)

    # ── escritura de resultados del backend (BA / grafo de poses) ────────────

    def set_keyframe_pose(self, kf_id: int, T_w_c: np.ndarray) -> None:
        """Sobrescribe la pose (resultado del BA — que optimiza poses
        directamente, a diferencia de update_poses que re-ancla por delta)."""
        self._kf_poses[kf_id] = np.asarray(T_w_c, dtype=float).copy()

    def set_point_positions(self, positions: Dict[int, np.ndarray]) -> None:
        """Sobrescribe posiciones de puntos refinadas por el BA."""
        for pid, p in positions.items():
            self._positions[pid] = np.asarray(p, dtype=np.float64).copy()

    def apply_similarity(self, kf_ids: Iterable[int], s: float,
                         R: np.ndarray, t: np.ndarray) -> None:
        """Aplica una SIMILITUD (escala + rotación + traslación) a un
        segmento del mapa: poses de esos keyframes y sus puntos anclados.

        Es la corrección de los cierres de bucle MONOCULARES: la deriva
        incluye escala, y una corrección rígida (SE(3)) no puede absorberla
        (lo medimos: 14% de deriva de escala sin BA; el grafo SE(3) solo
        EMPEORABA las cosas). Puntos: p' = s·R·p + t. Poses: la orientación
        rota (R·R_kf) y la posición se transforma como un punto — la escala
        NO entra al bloque de rotación (las poses siguen en SE(3); es el
        MAPA el que cambia de escala, y con él, todo PnP posterior).
        """
        wanted = set(kf_ids)
        for kf_id in wanted:
            T = self._kf_poses.get(kf_id)
            if T is None:
                continue
            T2 = np.eye(4)
            T2[:3, :3] = R @ T[:3, :3]
            T2[:3, 3] = s * (R @ T[:3, 3]) + t
            self._kf_poses[kf_id] = T2
        for i, anchor in enumerate(self._anchor_kf):
            if anchor in wanted:
                self._positions[i] = s * (R @ self._positions[i]) + t

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
        if not self._positions:
            return np.zeros((0, 3))
        return np.stack(self._positions)

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
