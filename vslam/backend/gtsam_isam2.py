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
6. **Modo VISUAL-INERCIAL** (v1.1 hito 3, opt-in vía `configure_imu`): estados
   de velocidad V(kf) y sesgo B(kf) por keyframe + `CombinedImuFactor` entre
   KFs consecutivos, alimentado con el segmento IMU crudo del intervalo (la
   preintegración y su teoría viven en imu_preintegration.py, lección 47; el
   solver es SOLO esta gemela — decisión de docs/05 §7). Las variables de pose
   SIGUEN siendo poses de CÁMARA: el extrínseco cámara←IMU va en
   `body_P_sensor` y GTSAM transforma las medidas — nada más cambia de frame.
   Con IMU el gauge cae (la gravedad fija roll/pitch, el sensor fija la
   escala): prior SOLO en la PRIMERA pose de cada época — con las 2 del modo
   visual, el segundo prior (σ=1e-4) CONGELA la escala del valor inicial y
   pelea contra el IMU (medido en el par nulo/observable del test). La cadena
   se RE-ANCLA tras cada reset: V/B del primer KF nuevo llevan prior con el
   último estimado (el sesgo es físico, sobrevive al reset) y el segmento que
   cruza el reset se descarta — época nueva, mismo patrón de la lección 46.
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
        self._huber_px = huber_px
        # Ruta estéreo (v0.6): calibración con baseline construida perezosamente
        # cuando llega el primer bf (el tracker lo pasa por keyframe; es constante
        # por secuencia). Ruido 3D con la misma Huber.
        self._stereo_bf = 0.0
        self._K_stereo = None
        self._robust_stereo = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(huber_px),
            gtsam.noiseModel.Isotropic.Sigma(3, 1.0))
        self._pose_prior = gtsam.noiseModel.Isotropic.Sigma(6, self.POSE_PRIOR_SIGMA)
        self._point_prior = gtsam.noiseModel.Isotropic.Sigma(3, self.POINT_PRIOR_SIGMA)
        self.n_failures = 0          # updates fallidos capturados (diagnóstico)
        # ── modo visual-inercial (v1.1, configure_imu) ──
        # Estado PERSISTENTE entre resets: el sesgo y la velocidad son físicos
        # (no gauge) — re-anclan la cadena de la época nueva.
        self._imu_params = None      # PreintegrationCombinedParams si VI activo
        self._imu_priors = None      # (noise_V, noise_B) de los anclajes
        self._vel_last = np.zeros(3)
        self._bias_last = None       # gtsam ConstantBias vigente
        # pids vistos en TODA la sesión (sobrevive a los resets): distingue un
        # punto NUEVO (entra libre) de uno ANTIGUO re-entrando tras un reset
        # (entra con prior de historia congelada) en O(1).
        self._session_pids: Set[int] = set()
        self._hard_reset()

    # ── ciclo de vida ─────────────────────────────────────────────────────────

    def _hard_reset(self) -> None:
        gtsam = self._gtsam
        params = gtsam.ISAM2Params()
        if self._imu_params is not None:
            # Modo VI: las correcciones son GRANDES y no lineales (escala,
            # gravedad, sesgos). Con el default relinearizeSkip=10 el grafo se
            # queda linealizado en el valor inicial y el ImuFactor no puede
            # tirar de la escala (medido en el test: 1.25 de 1.3). Relinealizar
            # SIEMPRE y con umbral más fino — solo en VI: la sintonía del modo
            # visual (14 ms/update, lección 32) no se toca.
            params.setRelinearizeThreshold(0.05)
            params.relinearizeSkip = 1       # (propiedad, no setter, en 4.2)
            # Gauss-Newton puro no navega correcciones grandes (escala 1.3 se
            # quedaba en 1.3 con residuos de ~70σ — medido): DOGLEG (región de
            # confianza) sí, es el LM incremental de iSAM2.
            params.setOptimizationParams(gtsam.ISAM2DoglegParams())
        else:
            params.setRelinearizeThreshold(0.1)
        self._isam = gtsam.ISAM2(params)
        self._poses_in: Set[int] = set()
        self._points_in: Set[int] = set()
        self._pose_priors_added = 0
        self._pending: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        self._vels_in: Set[int] = set()      # KFs con V/B en ESTA época
        self._imu_prev: Optional[int] = None  # último eslabón de la cadena IMU

    def reset(self) -> None:
        """Tras un cierre de bucle: el mapa cambió bajo nuestros pies (Sim(3)
        externa) → linealización obsoleta. Vaciar; los cursores NO se tocan
        (los guarda el tracker): lo antiguo re-entra solo si se re-observa,
        re-sembrado desde el mapper con prior (decisión 4)."""
        self._hard_reset()

    # ── modo visual-inercial (v1.1 hito 3) ────────────────────────────────────

    def configure_imu(self, noise, gravity_map: np.ndarray,
                      T_cam_imu: Optional[np.ndarray] = None,
                      init_gyro_bias: Optional[np.ndarray] = None,
                      init_accel_bias: Optional[np.ndarray] = None,
                      init_velocity: Optional[np.ndarray] = None,
                      vel_prior_sigma: float = 0.1,
                      bias_prior_sigma_accel: float = 0.5,
                      bias_prior_sigma_gyro: float = 0.01) -> None:
        """Activa el modo VI (decisión 6 del módulo). Llamar UNA vez, antes
        del primer keyframe.

        `noise`: densidades continuas (duck-type de `ImuNoiseParams`,
        imu_preintegration.py — GTSAM multiplica por dt internamente).
        `gravity_map`: g expresada en el frame del MAPA (con el mapa anclado
        en la primera cámara: g_map = R_cam_body · g_body de la init estática,
        imu_init.py). `T_cam_imu`: pose 4×4 del IMU en la CÁMARA (body_P_sensor;
        en EuRoC = inv(T_BS de cam0), porque imu0 ES el cuerpo).
        Sesgos/velocidad iniciales: de la init estática (b_a no es observable
        en reposo → default 0 con prior laxo de 0.5 m/s², lección 48).
        """
        gtsam = self._gtsam
        eye3 = np.eye(3)
        params = gtsam.PreintegrationCombinedParams(
            np.asarray(gravity_map, dtype=np.float64))
        params.setGyroscopeCovariance(noise.gyro_noise_density ** 2 * eye3)
        params.setAccelerometerCovariance(noise.accel_noise_density ** 2 * eye3)
        params.setIntegrationCovariance(1e-8 * eye3)
        params.setBiasAccCovariance(noise.accel_random_walk ** 2 * eye3)
        params.setBiasOmegaCovariance(noise.gyro_random_walk ** 2 * eye3)
        params.setBiasAccOmegaInit(1e-5 * np.eye(6))
        if T_cam_imu is not None:
            params.setBodyPSensor(gtsam.Pose3(np.asarray(T_cam_imu,
                                                         dtype=np.float64)))
        self._imu_params = params
        self._imu_priors = (
            gtsam.noiseModel.Isotropic.Sigma(3, vel_prior_sigma),
            gtsam.noiseModel.Diagonal.Sigmas(np.array(
                [bias_prior_sigma_accel] * 3 + [bias_prior_sigma_gyro] * 3)))
        bg = np.zeros(3) if init_gyro_bias is None else np.asarray(init_gyro_bias)
        ba = np.zeros(3) if init_accel_bias is None else np.asarray(init_accel_bias)
        self._bias_last = gtsam.imuBias.ConstantBias(ba.astype(np.float64),
                                                     bg.astype(np.float64))
        self._vel_last = (np.zeros(3) if init_velocity is None
                          else np.asarray(init_velocity, dtype=np.float64))
        # Re-crear iSAM2 con los parámetros del modo VI (ver _hard_reset).
        # Por eso configure_imu debe llamarse ANTES del primer keyframe.
        self._hard_reset()

    def _add_imu_chain(self, graph, values, cur: int, imu_data) -> None:
        """Encadena el KF actual: V/B nuevos y, si hay eslabón previo en ESTA
        época, el CombinedImuFactor del segmento. Sin eslabón (arranque de
        sesión o tras un reset): anclar V/B con priors del último estimado."""
        gtsam, sym = self._gtsam, self._sym
        if cur in self._vels_in:
            return
        values.insert(sym("v", cur), self._vel_last)
        values.insert(sym("b", cur), self._bias_last)
        self._vels_in.add(cur)
        prev = self._imu_prev
        segment_ok = (imu_data is not None and len(imu_data[0]) >= 2)
        if prev is None or prev not in self._vels_in or not segment_ok:
            noise_v, noise_b = self._imu_priors
            graph.add(gtsam.PriorFactorVector(sym("v", cur), self._vel_last,
                                              noise_v))
            graph.add(gtsam.PriorFactorConstantBias(sym("b", cur),
                                                    self._bias_last, noise_b))
            self._imu_prev = cur
            return
        ts, gyr, acc = imu_data
        pim = gtsam.PreintegratedCombinedMeasurements(self._imu_params,
                                                      self._bias_last)
        for k in range(len(ts) - 1):
            dt = float(ts[k + 1] - ts[k])
            if dt > 0.0:
                pim.integrateMeasurement(np.asarray(acc[k], dtype=np.float64),
                                         np.asarray(gyr[k], dtype=np.float64),
                                         dt)
        graph.add(gtsam.CombinedImuFactor(
            sym("x", prev), sym("v", prev), sym("x", cur), sym("v", cur),
            sym("b", prev), sym("b", cur), pim))
        self._imu_prev = cur

    @property
    def imu_chain_tail(self) -> Optional[int]:
        """Último KF encadenado de ESTA época (None si la cadena no arrancó).
        El tracker pide al driver el segmento desde aquí hasta el KF nuevo —
        si dos KFs coalescen en el worker, el segmento cubre ambos y la
        cadena sigue siendo una partición del tiempo."""
        return self._imu_prev

    @property
    def last_velocity(self) -> np.ndarray:
        """Última velocidad estimada (frame del mapa) — prior de movimiento
        para el frontend (hito 4)."""
        return self._vel_last.copy()

    @property
    def last_bias(self) -> Tuple[np.ndarray, np.ndarray]:
        """(b_gyro, b_accel) vigentes."""
        if self._bias_last is None:
            return np.zeros(3), np.zeros(3)
        return (np.asarray(self._bias_last.gyroscope(), dtype=np.float64),
                np.asarray(self._bias_last.accelerometer(), dtype=np.float64))

    # ── inserción incremental ─────────────────────────────────────────────────

    def _stereo_cal(self, bf: float):
        """Cal3_S2Stereo(baseline = bf/fx), cacheada por bf (constante/secuencia)."""
        if self._K_stereo is None or self._stereo_bf != bf:
            self._K_stereo = self._gtsam.Cal3_S2Stereo(
                self.camera.fx, self.camera.fy, 0.0,
                self.camera.cx, self.camera.cy, bf / self.camera.fx)
            self._stereo_bf = bf
        return self._K_stereo

    def process_keyframe(self, mapper, window: List[int],
                         new_obs: List[Tuple[int, int, np.ndarray]],
                         stereo_bf: float = 0.0,
                         imu_data: Optional[Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray]] = None,
                         ) -> Optional[Tuple[Dict[int, np.ndarray],
                                             Dict[int, np.ndarray]]]:
        """Alimenta las observaciones nuevas, corre un update incremental y
        devuelve ({kf: T_w_c}, {pid: pos}) refinados para la VENTANA (lo mismo
        que escribía el BA batch), o None si el update falló (degradación suave).

        `new_obs`: [(kf_id, point_id, píxel)] aún no alimentadas (el tracker
        lleva los cursores). Los puntos nuevos deben venir con sus ≥2 obs.
        `stereo_bf > 0` (RGB-D/estéreo): las obs (3,) con u_R finita entran como
        factor estéreo (ancla la escala métrica en el BA incremental, v0.6).
        `imu_data` (modo VI, requiere configure_imu): segmento CRUDO
        (ts, gyro, accel) entre el KF anterior y ESTE — el backend preintegra
        con el sesgo vigente y añade el CombinedImuFactor (decisión 6).
        """
        gtsam, sym = self._gtsam, self._sym
        K_s = self._stereo_cal(stereo_bf) if stereo_bf > 0.0 else None
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
        # OJO (bug cazado por el modo VI, v1.1): _session_pids debe registrar
        # solo pids INSERTADOS de verdad — registrarlos al verlos (incluidos
        # los que van a pendientes) convertía la 2ª obs de CADA punto en
        # "re-siembra" y le plantaba un PriorFactorPoint3 fantasma a la
        # posición del mapper. La visión pura no lo notaba (el prior es
        # consistente con su propia solución — gauge); el IMU, el primer
        # sensor que DISCREPA del mapa, lo desenmascaró: la escala corrupta
        # del test nulo/observable quedaba clavada en 1.25 por 28 priors.
        reseed_pids = {pid for pid in by_pid
                       if pid in self._session_pids and pid not in self._points_in}
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
                self._session_pids.add(pid)
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
            # Gauge: las 2 primeras poses de cada época llevan prior — salvo
            # en modo VI, donde solo la PRIMERA (el 2º prior congelaría la
            # escala del valor inicial y pelearía con el IMU — decisión 6).
            # Una pose que NO es el KF actual solo puede ser historia
            # re-sembrada tras un reset → prior también (lección 11).
            gauge_priors = 1 if self._imu_params is not None else 2
            if self._pose_priors_added < gauge_priors or kf != current_kf:
                graph.add(gtsam.PriorFactorPose3(X(kf), gtsam.Pose3(T),
                                                 self._pose_prior))
                self._pose_priors_added += 1

        for kf, pid, uv in factors:
            uv = np.asarray(uv, float)
            if K_s is not None and len(uv) == 3 and np.isfinite(uv[2]):
                # StereoPoint2(u_L, u_R, v): ancla la escala en el BA incremental.
                graph.add(gtsam.GenericStereoFactor3D(
                    gtsam.StereoPoint2(float(uv[0]), float(uv[2]), float(uv[1])),
                    self._robust_stereo, X(kf), L(pid), K_s))
            else:
                graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    uv[:2], self._robust, X(kf), L(pid), self._K))

        # 3b) Cadena IMU (modo VI): V/B del KF actual + factor del segmento.
        if self._imu_params is not None and current_kf >= 0:
            if current_kf not in self._poses_in:
                # KF sin obs consumibles todavía (p. ej. el PRIMERO: todas sus
                # obs esperan la 2ª vista en pendientes) — su pose entra aquí,
                # el factor IMU la necesita. MISMA política de gauge que el
                # paso 3: sin esto, X(0) quedaba SIN prior y el ancla de la
                # época acababa en X(1) con el valor inicial (bug medido).
                T0 = mapper.keyframe_pose(current_kf)
                values.insert(X(current_kf), self._gtsam.Pose3(T0))
                self._poses_in.add(current_kf)
                if self._pose_priors_added < 1:
                    graph.add(self._gtsam.PriorFactorPose3(
                        X(current_kf), self._gtsam.Pose3(T0), self._pose_prior))
                    self._pose_priors_added += 1
            self._add_imu_chain(graph, values, current_kf, imu_data)

        # 4) Update incremental (con red de seguridad: decisión 5).
        try:
            self._isam.update(graph, values)
            if self._imu_params is not None:
                # Iteraciones no lineales EXTRA (update vacío = otro paso GN
                # sobre las cliques afectadas): el acople pose↔velocidad↔sesgo
                # converge más lento que el puramente visual.
                self._isam.update()
                self._isam.update()
        except RuntimeError as exc:
            self.n_failures += 1
            self.last_error = str(exc)       # diagnóstico (qué variable/por qué)
            self._hard_reset()
            return None

        # 5) Write-back de la ventana (el mismo contrato que el BA batch).
        est = self._isam.calculateEstimate()
        if self._imu_params is not None and self._imu_prev is not None \
                and self._imu_prev in self._vels_in:
            self._vel_last = np.asarray(est.atVector(sym("v", self._imu_prev)),
                                        dtype=np.float64)
            self._bias_last = est.atConstantBias(sym("b", self._imu_prev))
        opt_poses = {k: est.atPose3(X(k)).matrix()
                     for k in window if k in self._poses_in}
        window_pids = {pid for k in window
                       for pid, _ in mapper._obs.get(k, [])
                       if pid in self._points_in}
        opt_points = {pid: np.asarray(est.atPoint3(L(pid)), float)
                      for pid in window_pids}
        return opt_poses, opt_points
