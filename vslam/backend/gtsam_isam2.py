"""BA local INCREMENTAL con iSAM2 — v0.5, el ataque al pico del keyframe.

El BA batch (aunque sea GTSAM) re-resuelve la ventana entera en cada keyframe:
~250-350 ms por KF. iSAM2 mantiene UN grafo de toda la sesión y, al llegar un
KF, solo re-linealiza/re-elimina las cliques afectadas del árbol de Bayes:
medido en el probe, **~14 ms por update y NO crece con el grafo** (60 KFs,
2360 puntos). Es la razón de ser de iSAM2 (Kaess et al., IJRR 2012).

─── La matemática (idea central) ──────────────────────────────────────────────
El batch factoriza J = QR (o el Hessiano por Cholesky) DESDE CERO cada vez.
iSAM2 guarda esa factorización como ÁRBOL DE BAYES (cliques de variables
eliminadas); un factor nuevo solo invalida las cliques entre sus variables y
la raíz — el resto del árbol se reutiliza. La re-linealización es selectiva:
solo variables cuyo delta supera un umbral (relinearizeThreshold) se
re-linealizan (fluid relinearization). Coste por update ≈ tamaño de la zona
afectada, no del grafo.

─── Decisiones de integración (medidas en el probe / heredadas del repo) ─────
1. **Los puntos ENTRAN con ≥ 2 observaciones**: con una sola, iSAM2 lanza
   IndeterminantLinearSystemException (medido). Es la lección 19 del repo en
   versión backend: todo punto nace con sus dos extremos de triangulación. Las
   observaciones huérfanas esperan en un buffer de PENDIENTES hasta juntar 2.
2. **Consumo INCREMENTAL de observaciones**: cursores por keyframe sobre las
   listas append-only del mapper (`mapper._obs` — acceso interno deliberado:
   la lista filtrada de `observations()` encoge con el culling y rompería los
   índices). Nada se re-alimenta dos veces.
3. **Gauge**: prior fuerte (σ=1e-4) sobre las 2 primeras poses de cada época
   (el gauge monocular tiene 7 gdl; 2 poses lo fijan — §bundle_adjustment).
4. **RESET tras cierre de bucle**: la corrección Sim(3) del bucle reescribe
   poses y puntos FUERA de iSAM2 → su linealización queda obsoleta. Se vacía
   el grafo (los cursores se conservan: lo viejo no se re-alimenta) y las
   variables antiguas que reaparezcan se RE-SIEMBRAN con su valor actual del
   mapper + un prior (pose σ=1e-4, punto σ=0.02): historia congelada, igual
   que el segmento viejo del grafo del bucle (lección 11).
5. Si update() aun así falla (quiralidad extrema, sistema mal puesto), se
   captura, se resetea y el tracker sigue con las poses del PnP: degradación
   suave, nunca un crash a mitad de secuencia.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from vslam.core.camera import PinholeCamera

_INSTALL_MSG = ("El backend iSAM2 requiere `gtsam` (conda-forge):\n"
                "    conda install -c conda-forge gtsam   (exige numpy<2)")


class ISAM2LocalBA:
    """BA local incremental. El tracker lo llama una vez por keyframe
    (`process_keyframe`) y tras cada cierre de bucle (`reset`)."""

    POSE_PRIOR_SIGMA = 1e-4     # gauge / historia congelada
    POINT_PRIOR_SIGMA = 0.02    # puntos re-sembrados tras un reset

    def __init__(self, camera: PinholeCamera, huber_px: float = 2.5) -> None:
        try:
            import gtsam
        except ImportError as exc:                   # pragma: no cover
            raise ImportError(_INSTALL_MSG) from exc
        self._gtsam = gtsam
        self._sym = gtsam.symbol
        self.camera = camera
        self._K = gtsam.Cal3_S2(camera.fx, camera.fy, 0.0, camera.cx, camera.cy)
        px = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        self._robust = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(huber_px), px)
        self._pose_prior = gtsam.noiseModel.Isotropic.Sigma(6, self.POSE_PRIOR_SIGMA)
        self._point_prior = gtsam.noiseModel.Isotropic.Sigma(3, self.POINT_PRIOR_SIGMA)
        self.n_failures = 0          # updates fallidos capturados (diagnóstico)
        # pids vistos en TODA la sesión (sobrevive a los resets): distingue un
        # punto NUEVO (entra libre) de uno ANTIGUO re-entrando tras un reset
        # (entra con prior de historia congelada) en O(1).
        self._session_pids: Set[int] = set()
        self._hard_reset()

    # ── ciclo de vida ─────────────────────────────────────────────────────────

    def _hard_reset(self) -> None:
        gtsam = self._gtsam
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.1)
        self._isam = gtsam.ISAM2(params)
        self._poses_in: Set[int] = set()
        self._points_in: Set[int] = set()
        self._pose_priors_added = 0
        self._pending: Dict[int, List[Tuple[int, np.ndarray]]] = {}

    def reset(self) -> None:
        """Tras un cierre de bucle: el mapa cambió bajo nuestros pies (Sim(3)
        externa) → linealización obsoleta. Vaciar; los cursores NO se tocan
        (los guarda el tracker): lo antiguo re-entra solo si se re-observa,
        re-sembrado desde el mapper con prior (decisión 4)."""
        self._hard_reset()

    # ── inserción incremental ─────────────────────────────────────────────────

    def process_keyframe(self, mapper, window: List[int],
                         new_obs: List[Tuple[int, int, np.ndarray]],
                         ) -> Optional[Tuple[Dict[int, np.ndarray],
                                             Dict[int, np.ndarray]]]:
        """Alimenta las observaciones nuevas, corre un update incremental y
        devuelve ({kf: T_w_c}, {pid: pos}) refinados para la VENTANA (lo mismo
        que escribía el BA batch), o None si el update falló (degradación suave).

        `new_obs`: [(kf_id, point_id, píxel)] aún no alimentadas (el tracker
        lleva los cursores). Los puntos nuevos deben venir con sus ≥2 obs.
        """
        gtsam, sym = self._gtsam, self._sym
        X = lambda i: sym("x", i)                    # noqa: E731
        L = lambda j: sym("l", j)                    # noqa: E731
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        # 1) Agrupar por punto; juntar con pendientes de rondas anteriores.
        by_pid: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        for kf, pid, uv in new_obs:
            by_pid.setdefault(pid, []).append((kf, uv))
        for pid, entries in list(self._pending.items()):
            if pid in by_pid:
                by_pid[pid] = entries + by_pid[pid]
                del self._pending[pid]

        # 2) Decidir qué factores entran; recolectar poses necesarias. Un pid
        # visto en una época anterior (reset de por medio) es RE-SIEMBRA.
        reseed_pids = {pid for pid in by_pid
                       if pid in self._session_pids and pid not in self._points_in}
        self._session_pids.update(by_pid)
        factors: List[Tuple[int, int, np.ndarray]] = []
        needed_poses: Set[int] = set()
        for pid, entries in by_pid.items():
            if pid not in self._points_in and len(entries) < 2 \
                    and pid not in reseed_pids:
                self._pending[pid] = entries         # espera su segunda obs
                continue
            if pid not in self._points_in:
                p = mapper.point_positions([pid])[pid]
                values.insert(L(pid), gtsam.Point3(float(p[0]), float(p[1]),
                                                   float(p[2])))
                self._points_in.add(pid)
                if pid in reseed_pids:               # historia congelada
                    graph.add(gtsam.PriorFactorPoint3(
                        L(pid), gtsam.Point3(float(p[0]), float(p[1]),
                                             float(p[2])), self._point_prior))
            for kf, uv in entries:
                factors.append((kf, pid, uv))
                needed_poses.add(kf)

        # 3) Poses nuevas (o re-sembradas tras reset) con su valor del mapper.
        current_kf = window[-1] if window else max(needed_poses, default=-1)
        for kf in sorted(needed_poses):
            if kf in self._poses_in:
                continue
            T = mapper.keyframe_pose(kf)
            values.insert(X(kf), gtsam.Pose3(T))
            self._poses_in.add(kf)
            # Gauge: las 2 primeras poses de cada época llevan prior; una pose
            # que NO es el KF actual solo puede ser historia re-sembrada tras
            # un reset → prior también (historia congelada, lección 11).
            if self._pose_priors_added < 2 or kf != current_kf:
                graph.add(gtsam.PriorFactorPose3(X(kf), gtsam.Pose3(T),
                                                 self._pose_prior))
                self._pose_priors_added += 1

        for kf, pid, uv in factors:
            # [:2]: las observaciones RGB-D (v0.6) traen [u, v, u_R]; el factor
            # estéreo de iSAM2 es deuda — se proyecta con [u, v] (paridad).
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                np.asarray(uv, float)[:2], self._robust, X(kf), L(pid), self._K))

        # 4) Update incremental (con red de seguridad: decisión 5).
        try:
            self._isam.update(graph, values)
        except RuntimeError:
            self.n_failures += 1
            self._hard_reset()
            return None

        # 5) Write-back de la ventana (el mismo contrato que el BA batch).
        est = self._isam.calculateEstimate()
        opt_poses = {k: est.atPose3(X(k)).matrix()
                     for k in window if k in self._poses_in}
        window_pids = {pid for k in window
                       for pid, _ in mapper._obs.get(k, [])
                       if pid in self._points_in}
        opt_points = {pid: np.asarray(est.atPoint3(L(pid)), float)
                      for pid in window_pids}
        return opt_poses, opt_points
