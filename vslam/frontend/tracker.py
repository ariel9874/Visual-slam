"""Trackers: el contrato de la capa de tracking y su implementación 3D-2D (v0.2).

La implementación 2D-2D didáctica (matriz esencial pura, un solo archivo
legible de arriba a abajo) sigue en examples/01_monocular_vo.py. Aquí vive el
tracker "de producción" de v0.2: PnPTracker, que trackea contra un mapa
disperso persistente.

─── La matemática de cada estrategia ───
2D-2D (ejemplo 01): sin mapa. La pose relativa sale de la restricción epipolar
x̂'ᵀ·E·x̂ = 0 (derivada en examples/01). Coste estructural: la escala de t
es inobservable y TODO se re-estima en cada par de frames — el error de
DIRECCIÓN y de ESCALA se integra frame a frame (el zigzag del benchmark).

3D-2D / PnP (esta clase): con mapa {X_i ↔ u_i}. Se minimiza el ERROR DE
REPROYECCIÓN — la función objetivo central de todo el SLAM geométrico:

    T* = argmin_T  Σ_i  ρ( ‖ π(K, T⁻¹·X_i) − u_i ‖² )

donde T = T_w_c (la pose buscada), π es la proyección pinhole del punto
llevado al frame de cámara, y ρ un kernel robusto (aquí: RANSAC + refinado LM
solo con inliers). Como los X_i ya tienen escala, la pose la HEREDA del mapa:
la deriva de escala frame a frame desaparece, y localizar contra el mapa es
una medición ABSOLUTA (en el marco del mapa), no un incremento que se apila.

KLT (v0.4): en lugar de re-detectar y describir, sigue cada parche
minimizando el error fotométrico local (Lucas-Kanade):
    Δu* = argmin_Δu Σ_parche ( I_t(u + Δu) − I_{t−1}(u) )²
linealizado con el gradiente de imagen (Gauss-Newton sobre 2 parámetros por
punto, en pirámide para movimientos grandes). Es la puerta de entrada a los
métodos DIRECTOS (DSO generaliza esta idea a pose + profundidades).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np

from vslam.backend.bundle_adjustment import local_bundle_adjustment
from vslam.core.camera import PinholeCamera
from vslam.core.frame import Frame
from vslam.core.geometry import invert_se3, solve_pnp, triangulate_two_views
from vslam.frontend.features import create_extractor
from vslam.frontend.matching import create_matcher
from vslam.mapping.sparse import SparsePointMapper


class TrackerBase(ABC):
    """Contrato: recibe frames, devuelve poses; decide keyframes."""

    @abstractmethod
    def process(self, frame: Frame) -> Optional[np.ndarray]:
        """Estima y escribe frame.T_w_c. Devuelve la pose, o None si el
        tracking se perdió (el llamador decide: relocalizar o reiniciar)."""
        raise NotImplementedError


class PnPTracker(TrackerBase):
    """Tracking monocular 3D-2D contra un mapa disperso (estilo ORB-SLAM-mini).

    Ciclo de vida:
      1. INICIALIZACIÓN (2D-2D, una sola vez): se espera paralaje suficiente
         contra un frame de referencia, se estima E, se recupera (R, t) y se
         TRIANGULA el primer mapa. La escala monocular es libre (gauge): se
         fija normalizando la profundidad mediana a 1.0 — a partir de ahí
         TODO el sistema es consistente con esa unidad.
      2. TRACKING (3D-2D, cada frame): matching descriptores del mapa ↔ frame,
         PnP robusto → pose absoluta en el marco del mapa.
      3. KEYFRAMES: cuando los inliers caen (los puntos salen del campo de
         visión), se promueve el frame a keyframe y se triangulan puntos
         NUEVOS contra el keyframe anterior — el mapa crece con el recorrido
         y hereda la escala fijada en (1) a través de las poses PnP.

    ─── La matemática: fijar el gauge de escala ───
    Si (T, {X_i}) explica las imágenes, (s·t, {s·X_i}) también (examples/01).
    Esa familia de soluciones es un "gauge" (grado de libertad no observable).
    Elegir mediana(profundidad) = 1.0 selecciona UN representante; no es una
    medición, es una convención — pero al triangular los puntos nuevos desde
    poses YA expresadas en ese gauge, la escala se propaga coherentemente y
    solo puede degradarse por acumulación de error (deriva de escala lenta),
    no re-sortearse en cada frame como en 2D-2D.
    """

    # Umbrales (didácticos; en producción serían configuración externa).
    # Paralaje mínimo (mediana, px) para inicializar. Medido en el barrido de
    # parámetros del repo: con 25 px el mapa inicial hereda demasiado error de
    # dirección de E (ATE 10.8 cm en la secuencia sintética); con 40 px el ATE
    # baja a 6.9 cm. Esperar cuesta frames anclados al origen; vale la pena.
    MIN_INIT_FLOW_PX = 40.0
    MIN_INIT_POINTS = 50        # puntos triangulados mínimos del mapa inicial
    INIT_VALIDATION_RATIO = 0.7  # inliers/matches mínimos en la 3ª vista
    INIT_SURVIVAL_RATIO = 0.5    # triangulados válidos / inliers epipolares
    MIN_MAP_MATCHES = 30        # matches mapa↔frame para intentar PnP
    MIN_PNP_INLIERS = 15        # inliers mínimos para aceptar la pose
    KF_MIN_GAP = 3              # frames mínimos entre keyframes
    KF_MAX_GAP = 15             # frames MÁXIMOS: insertar aunque no haya hambre
    KF_INLIER_RATIO = 0.6       # nuevo KF si inliers < 60% de los del último KF
    KF_MIN_INLIERS = 100        # ... o si caen de este absoluto
    CHEIRALITY_DIST_THRESH = 2000.0  # ver la "TRAMPA CLÁSICA" en examples/01
    BA_WINDOW = 5               # keyframes en la ventana del BA local
    BA_ITERATIONS = 6
    # ── cierre de bucle (v0.35) ──
    LOOP_TEMPORAL_GAP = 60      # frames mínimos de antigüedad del candidato
    LOOP_MIN_MATCHES = 200      # matches brutos para considerar un candidato
    LOOP_MIN_INLIERS = 40       # inliers PnP de la verificación geométrica
    LOOP_COOLDOWN = 40          # frames sin reintentar tras cerrar un bucle

    def __init__(self, camera: PinholeCamera, extractor=None, matcher=None,
                 mapper: Optional[SparsePointMapper] = None,
                 local_window: Optional[int] = None,
                 local_ba: bool = True,
                 loop_closure: bool = False) -> None:
        """Args (además del frontend intercambiable):
            local_window: si se da, el matching 3D-2D usa solo los puntos de
                los últimos N keyframes (mapa LOCAL: costo acotado, pero la
                deriva reaparece — el escenario que exige cierre de bucle).
                None = mapa global (comportamiento v0.2).
            local_ba: refina la ventana de keyframes con bundle adjustment
                tras cada inserción (vslam/backend/bundle_adjustment.py).
            loop_closure: reconocimiento de lugar + verificación geométrica +
                corrección por grafo de poses en cada keyframe (ver
                _try_close_loop). Pensado para usarse con local_window.
        """
        import cv2  # local para mantener el módulo importable en docs/tests ligeros
        self._cv2 = cv2
        self.camera = camera
        self.extractor = extractor or create_extractor("orb")
        self.matcher = matcher or create_matcher("ratio")
        # Ojo: `mapper or ...` sería un bug — un mapper VACÍO define __len__=0
        # y Python lo evalúa como falsy, creando silenciosamente otro objeto.
        self.mapper = mapper if mapper is not None else SparsePointMapper()
        self.local_window = local_window
        self.local_ba = local_ba
        self.loop_closure = loop_closure
        self._kf_ids: list = []          # keyframes en orden de inserción
        self._kf_db: list = []           # historial para reconocimiento de lugar
        self._last_loop_frame = -10 ** 9
        self.loop_events: list = []      # [(frame, kf_antiguo)] para informes

        self.T_w_c = np.eye(4)
        self._T_prev = np.eye(4)         # pose anterior (para velocidad constante)
        self._T_rel = np.eye(4)          # último movimiento (coasting)
        self._initialized = False
        self._frame_idx = -1

        # Referencia de inicialización y último keyframe.
        self._ref: Optional[Tuple[list, np.ndarray]] = None   # (kps, desc)
        self._init_buffer: list = []     # frames entre la referencia y el actual
        self._kf: Optional[Dict] = None  # {"id", "kps", "desc", "mp": idx→map_id}
        self._kf_inliers = 0
        self._frames_since_kf = 0

    # ── API ────────────────────────────────────────────────────────────────────

    def process(self, frame: Frame) -> Optional[np.ndarray]:
        T, info = self.process_frame(frame.image)
        frame.T_w_c = T
        return T if info["tracked"] else None

    def process_frame(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Procesa un frame; devuelve (T_w_c, info) — misma forma que el
        pipeline del ejemplo 01, para poder compararlos en el benchmark."""
        self._frame_idx += 1
        kps, desc = self.extractor.detect_and_compute(gray)
        info = {"n_kps": len(kps), "n_matches": 0, "n_inliers": 0,
                "tracked": False, "state": "", "n_map": len(self.mapper),
                "pts_prev": None, "pts_curr": None}

        if not self._initialized:
            self._initialize_step(gray, kps, desc, info)
        else:
            self._track_step(gray, kps, desc, info)

        info["n_map"] = len(self.mapper)
        return self.T_w_c, info

    # ── fase 1: inicialización 2D-2D + triangulación ──────────────────────────

    def _initialize_step(self, gray, kps, desc, info) -> None:
        cv2 = self._cv2
        info["state"] = "INIT"
        if self._ref is None:
            # El primer frame define el origen del mundo y la referencia.
            self._ref = (kps, desc)
            self._init_buffer = []
            info["tracked"] = True
            return

        # Guardamos los frames intermedios: serán la "tercera vista" que
        # valide (o desmienta) la estructura inicial.
        self._init_buffer.append((kps, desc))

        ref_kps, ref_desc = self._ref
        matches = self.matcher.match(ref_desc, desc, ref_kps, kps, gray.shape)
        info["n_matches"] = len(matches)
        if len(matches) < 2 * self.MIN_INIT_POINTS:
            self._ref = (kps, desc)     # se perdió la referencia: reintentar
            self._init_buffer = []
            return

        pts0 = np.float64([ref_kps[m.queryIdx].pt for m in matches])
        pts1 = np.float64([kps[m.trainIdx].pt for m in matches])

        # Sin paralaje no hay triangulación: esperar a que la cámara se mueva.
        flow = np.median(np.linalg.norm(pts1 - pts0, axis=1))
        if flow < self.MIN_INIT_FLOW_PX:
            info["tracked"] = True      # no es un fallo: aún anclados al origen
            return

        # Geometría epipolar (la matemática completa está en examples/01).
        # MAGSAC++ marginaliza el umbral de inliers y es notablemente más
        # estable que el RANSAC clásico en escenas cuasi-planas como esta.
        E, mask = cv2.findEssentialMat(pts0, pts1, self.camera.K,
                                       method=cv2.USAC_MAGSAC, prob=0.999,
                                       threshold=1.0)
        if E is None or E.shape != (3, 3):
            return
        n_inl, R, t, mask, _ = cv2.recoverPose(
            E, pts0, pts1, self.camera.K,
            distanceThresh=self.CHEIRALITY_DIST_THRESH, mask=mask)
        if n_inl < self.MIN_INIT_POINTS:
            return

        # Pose provisional del frame actual con ||t|| = 1 (gauge aún libre).
        T_c1_c0 = np.eye(4)
        T_c1_c0[:3, :3] = R
        T_c1_c0[:3, 3] = t.ravel()
        T_w_c1 = invert_se3(T_c1_c0)    # el mundo ES la cámara de referencia

        inl = mask.ravel().astype(bool)
        points_w, valid = triangulate_two_views(
            self.camera, np.eye(4), T_w_c1, pts0[inl], pts1[inl],
            min_parallax_deg=0.5)
        # TASA DE SUPERVIVENCIA: con la pose CORRECTA, la gran mayoría de los
        # inliers epipolares triangula limpio; con la solución falsa del
        # twisted pair, los filtros de quiralidad/paralaje masacran los puntos
        # (lo medimos: ~90% vs ~13%). Una supervivencia baja delata pose mala.
        if valid.sum() < self.MIN_INIT_POINTS or valid.mean() < self.INIT_SURVIVAL_RATIO:
            return

        # FIJAR EL GAUGE: profundidad mediana (vista desde la referencia) = 1.
        pts = points_w[valid]
        scale = 1.0 / np.median(pts[:, 2])
        pts = pts * scale
        T_w_c1[:3, 3] *= scale
        idx_curr = np.array([m.trainIdx for m in matches])[inl][valid]

        # ── VALIDACIÓN CON UNA TERCERA VISTA ─────────────────────────────────
        # ─── La matemática: por qué dos vistas pueden mentir ───
        # La triangulación de dos vistas ajusta SUS dos vistas por
        # construcción: para (casi) cualquier pose relativa —incluida la
        # solución FALSA del "twisted pair", que en escenas cuasi-planas como
        # esta también pasa el test de quiralidad— existe una estructura 3D
        # con error de reproyección ≈ 0 en ambas. El error de reproyección de
        # dos vistas NO valida la pose. Una vista INTERMEDIA sí: no participó
        # en la construcción, y una estructura falsa no se re-proyecta
        # coherentemente en ella (es la idea del model-selection de ORB-SLAM,
        # en versión mínima). Si la validación falla, seguimos esperando
        # mejores condiciones en lugar de hornear un mapa envenenado.
        if not self._third_view_confirms(pts, desc[idx_curr], T_w_c1):
            if flow > 4 * self.MIN_INIT_FLOW_PX:
                self._ref = (kps, desc)          # demasiado tiempo: reiniciar
                self._init_buffer = []
            info["tracked"] = True
            return

        # Nace el mapa: keyframe 0 (referencia) y keyframe actual.
        kf_id = self._frame_idx
        self.mapper.integrate_keyframe(Frame(frame_id=0, timestamp=0.0, T_w_c=np.eye(4),
                                             is_keyframe=True))
        self.mapper.integrate_keyframe(Frame(frame_id=kf_id, timestamp=0.0,
                                             T_w_c=T_w_c1, is_keyframe=True))
        ids = self.mapper.add_points(pts, desc[idx_curr], anchor_kf_id=kf_id)
        # Observaciones de ambos keyframes fundadores: el combustible del BA.
        self.mapper.add_observations(0, ids, pts0[inl][valid])
        self.mapper.add_observations(kf_id, ids, pts1[inl][valid])
        self._kf_ids = [0, kf_id]
        # Base de datos de keyframes para el reconocimiento de lugar.
        idx_ref = np.array([m.queryIdx for m in matches])[inl][valid]
        self._kf_db = [
            {"id": 0, "kps": ref_kps, "desc": ref_desc,
             "mp": dict(zip(idx_ref.tolist(), ids))},
            {"id": kf_id, "kps": kps, "desc": desc,
             "mp": dict(zip(idx_curr.tolist(), ids))},
        ]

        self.T_w_c = T_w_c1
        self._T_prev = T_w_c1.copy()
        self._kf = {"id": kf_id, "kps": kps, "desc": desc,
                    "mp": dict(zip(idx_curr.tolist(), ids)), "T": T_w_c1.copy()}
        self._kf_inliers = len(ids)
        self._frames_since_kf = 0
        self._initialized = True
        self._init_buffer = []
        info.update(tracked=True, n_inliers=int(valid.sum()), state="INIT-OK")

    def _third_view_confirms(self, points_w, descriptors, T_w_c1) -> bool:
        """Valida la estructura candidata localizando una vista intermedia
        contra ella con PnP (racional: comentario en _initialize_step).

        Criterios: PnP debe converger con una fracción alta de inliers, y la
        posición recuperada debe caer ENTRE el origen y el frame actual (la
        vista intermedia se capturó físicamente en ese trayecto: con la
        solución falsa del twisted pair esta coherencia se rompe).
        """
        if len(self._init_buffer) < 3:
            return False                # aún no hay vista intermedia real
        mid_kps, mid_desc = self._init_buffer[len(self._init_buffer) // 2]

        matches = self.matcher.match(descriptors, mid_desc, None, mid_kps, None)
        if len(matches) < self.MIN_MAP_MATCHES:
            return False
        obj = points_w[[m.queryIdx for m in matches]]
        img = np.float64([mid_kps[m.trainIdx].pt for m in matches])

        T_mid, inliers = solve_pnp(self.camera, obj, img)
        if T_mid is None or inliers.sum() < max(
                self.MIN_MAP_MATCHES, self.INIT_VALIDATION_RATIO * len(matches)):
            return False

        # Proyección escalar de la posición intermedia sobre el segmento
        # origen→frame actual: debe estar dentro, t = (p·e)/(e·e) ∈ (0, 1).
        p_end = T_w_c1[:3, 3]
        ratio = float(T_mid[:3, 3] @ p_end) / float(p_end @ p_end + 1e-12)
        return 0.0 < ratio < 1.0

    # ── fase 2: tracking 3D-2D ────────────────────────────────────────────────

    def _track_step(self, gray, kps, desc, info) -> None:
        info["state"] = "TRACK"
        # Mapa global (v0.2) o LOCAL (últimos N keyframes): ver __init__.
        window = None if self.local_window is None else self._kf_ids[-self.local_window:]
        ids, map_pts, map_desc = self.mapper.snapshot(window)

        # Matching descriptores del MAPA (query) contra el frame (train).
        matches = self.matcher.match(map_desc, desc, None, kps, gray.shape)
        info["n_matches"] = len(matches)
        if len(matches) < self.MIN_MAP_MATCHES:
            self._coast(info)
            return

        obj = map_pts[[m.queryIdx for m in matches]]
        img = np.float64([kps[m.trainIdx].pt for m in matches])

        T_w_c, inlier_mask = solve_pnp(self.camera, obj, img)
        if T_w_c is None or inlier_mask.sum() < self.MIN_PNP_INLIERS:
            self._coast(info)
            return

        # Pose absoluta contra el mapa: no se apila sobre la anterior.
        self._T_rel = invert_se3(self._T_prev) @ T_w_c
        self._T_prev = T_w_c.copy()
        self.T_w_c = T_w_c
        n_inliers = int(inlier_mask.sum())
        info.update(tracked=True, n_inliers=n_inliers)

        # Para visualización: residuo mapa proyectado → observación (inliers).
        proj = self._project(obj[inlier_mask])
        info["pts_prev"], info["pts_curr"] = proj, img[inlier_mask]

        # ¿Keyframe? Dos disparadores: HAMBRE (el mapa visible se agota) o
        # INTERVALO MÁXIMO. El segundo es una lección de los sistemas reales
        # (lo aprendimos midiendo: en una escena siempre co-visible el hambre
        # nunca llega y el sistema pasó 90 frames sin un solo keyframe — sin
        # KFs no hay BA, ni mapa local honesto, ni base para detectar bucles).
        # ORB-SLAM lo resuelve insertando con generosidad y podando después.
        # ... pero NUNCA desde una pose incierta: un keyframe con pocos
        # inliers triangula cientos de puntos basura que envenenan el mapa
        # (medido: un KF con 26 inliers creó 584 puntos y el tracker saltó
        # 6 unidades en un frame). Si la pose es dudosa, mejor esperar.
        self._frames_since_kf += 1
        starving = (n_inliers < self.KF_INLIER_RATIO * self._kf_inliers
                    or n_inliers < self.KF_MIN_INLIERS)
        overdue = self._frames_since_kf >= self.KF_MAX_GAP
        healthy = n_inliers >= 3 * self.MIN_PNP_INLIERS
        if self._frames_since_kf >= self.KF_MIN_GAP and healthy \
                and (starving or overdue):
            # kp del frame → id GLOBAL del punto (ids traduce índices del
            # snapshot — que puede ser un subconjunto local — a ids del mapa).
            matched_frame_idx = {m.trainIdx: int(ids[m.queryIdx])
                                 for m, ok in zip(matches, inlier_mask) if ok}
            self._insert_keyframe(gray, kps, desc, matched_frame_idx, info)

    def _insert_keyframe(self, gray, kps, desc, matched_frame_idx, info) -> None:
        """Promueve el frame a keyframe y triangula puntos NUEVOS contra el
        keyframe anterior (matches que no corresponden a puntos ya mapeados)."""
        kf = self._kf
        matches = self.matcher.match(kf["desc"], desc, kf["kps"], kps, gray.shape)
        fresh = [m for m in matches
                 if m.queryIdx not in kf["mp"] and m.trainIdx not in matched_frame_idx]

        kf_id = self._frame_idx
        # Asociaciones del nuevo keyframe: kp del frame → id global del punto.
        mp = dict(matched_frame_idx)

        if len(fresh) >= 10:
            pts0 = np.float64([kf["kps"][m.queryIdx].pt for m in fresh])
            pts1 = np.float64([kps[m.trainIdx].pt for m in fresh])
            # Las poses ya viven en el gauge del mapa → los puntos nuevos
            # heredan la escala automáticamente (no se re-normaliza nada).
            points_w, valid = triangulate_two_views(
                self.camera, kf["T"], self.T_w_c, pts0, pts1)
            if valid.any():
                idx_curr = np.array([m.trainIdx for m in fresh])[valid]
                ids = self.mapper.add_points(points_w[valid], desc[idx_curr],
                                             anchor_kf_id=kf_id)
                mp.update(dict(zip(idx_curr.tolist(), ids)))
                # Registrar TAMBIÉN la observación en el keyframe previo (el
                # otro extremo de la triangulación). Sin ella el punto queda
                # con una sola observación y el BA puede deslizarlo libremente
                # a lo largo de su rayo visual (C_p de rango 2: 3 incógnitas,
                # 2 ecuaciones) — lo medimos: el BA EMPEORABA el ATE por esto.
                self.mapper.add_observations(kf["id"], ids, pts0[valid])

        self.mapper.integrate_keyframe(Frame(frame_id=kf_id, timestamp=0.0,
                                             T_w_c=self.T_w_c.copy(), is_keyframe=True))
        if mp:
            kp_idx = list(mp.keys())
            self.mapper.add_observations(
                kf_id, list(mp.values()),
                np.float64([kps[i].pt for i in kp_idx]))
        self._kf_ids.append(kf_id)
        self._kf = {"id": kf_id, "kps": kps, "desc": desc, "mp": mp,
                    "T": self.T_w_c.copy()}
        self._kf_db.append({"id": kf_id, "kps": kps, "desc": desc, "mp": mp})
        self._kf_inliers = max(info["n_inliers"], 1)
        self._frames_since_kf = 0
        info["state"] = "TRACK+KF"

        if self.local_ba:
            self._run_local_ba()
        if self.loop_closure:
            self._try_close_loop(gray, kps, desc, info)

    def _run_local_ba(self) -> None:
        """Bundle adjustment sobre la ventana de keyframes recientes.

        Se anclan los DOS keyframes más viejos de la ventana: uno fija
        rotación/traslación y el segundo fija la ESCALA — el gauge monocular
        tiene 7 grados de libertad (la lección medida en bundle_adjustment.py).
        Simplificación v0.35: las observaciones desde keyframes fuera de la
        ventana no participan (ORB-SLAM las incluye como cámaras fijas).
        """
        window = self._kf_ids[-self.BA_WINDOW:]
        if len(window) < 3:
            return
        obs = self.mapper.observations(window)
        if len(obs) < 60:
            return
        kf_poses = {k: self.mapper.keyframe_pose(k) for k in window}
        # Solo se optimizan puntos con ≥ 2 observaciones DENTRO de la ventana:
        # con una, el punto es libre a lo largo de su rayo (sub-determinado).
        counts: dict = {}
        for _, pid, _ in obs:
            counts[pid] = counts.get(pid, 0) + 1
        points = self.mapper.point_positions(
            {pid for pid, c in counts.items() if c >= 2})
        opt_poses, opt_points = local_bundle_adjustment(
            self.camera, kf_poses, points, obs, fixed_kfs=set(window[:2]),
            iterations=self.BA_ITERATIONS)

        for k, T in opt_poses.items():
            self.mapper.set_keyframe_pose(k, T)
        self.mapper.set_point_positions(opt_points)
        # El keyframe recién insertado ES el frame actual: heredar su refinado.
        cur = self._kf["id"]
        self.T_w_c = opt_poses[cur].copy()
        self._T_prev = self.T_w_c.copy()
        self._kf["T"] = self.T_w_c.copy()

    def _try_close_loop(self, gray, kps, desc, info) -> None:
        """Cierre de bucle en tres actos (el pipeline canónico del SLAM):

        1. RECONOCIMIENTO DE LUGAR: ¿este keyframe se parece a uno antiguo?
           Aquí, matching de descriptores por fuerza bruta contra la base de
           keyframes (a decenas de KFs es trivial; a miles se usa bolsa de
           palabras/BoW — mismo rol, costo sub-lineal). El filtro temporal
           excluye keyframes recientes: parecerse al pasado inmediato no es
           un bucle, es continuidad.
        2. VERIFICACIÓN GEOMÉTRICA: la apariencia miente (pasillos gemelos);
           la geometría no. PnP del frame actual contra los puntos 3D del
           candidato: si converge con muchos inliers, da además la pose
           actual EXPRESADA EN EL MARCO DEL SEGMENTO ANTIGUO — la medición
           del bucle.
        3. CORRECCIÓN tipo relocalización, con ESCALA (la lección monocular
           dura, medida en este repo): la deriva monocular incluye escala
           (14% sin BA local) y una corrección rígida SE(3) no puede
           absorberla — reparte la inconsistencia como error de traslación y
           EMPEORA el resultado (medido: ATE 35.9 → 94.1 cm). Es el motivo
           por el que ORB-SLAM cierra bucles monoculares en Sim(3) (Strasdat
           et al., RSS 2010). La versión mínima implementada aquí: los
           puntos vistos por ambos segmentos existen DOS veces en el mapa
           (gauge viejo y gauge derivado); la similitud (s, R, t) que casa
           esas dos nubes (Umeyama, la matemática de vslam/evaluation.py)
           re-alinea y RE-ESCALA la ventana local sobre el mapa antiguo. A
           partir de ahí el tracking continúa en el gauge original.

           Lo que NO se hace (todavía): redistribuir la corrección hacia
           atrás por la cadena. Eso exige un grafo de poses Sim(3) (7 gdl
           por nodo) — nuestro grafo SE(3) de backend/pose_graph.py lo
           probamos aquí y, con deriva de escala, empeoraba las cosas.
           Queda en la hoja de ruta (v0.4) con su propia álgebra en lie.py.
        """
        if self._frame_idx - self._last_loop_frame < self.LOOP_COOLDOWN:
            return
        cur_id = self._kf["id"]

        # 1) Reconocimiento de lugar.
        best = None
        for old in self._kf_db[:-1]:
            if cur_id - old["id"] < self.LOOP_TEMPORAL_GAP:
                continue
            matches = self.matcher.match(old["desc"], desc, old["kps"], kps,
                                         gray.shape)
            if len(matches) >= self.LOOP_MIN_MATCHES and \
                    (best is None or len(matches) > len(best[1])):
                best = (old, matches)
        if best is None:
            return
        old, matches = best

        # 2) Verificación geométrica contra los puntos 3D del candidato.
        pairs = [(old["mp"][m.queryIdx], m.trainIdx) for m in matches
                 if m.queryIdx in old["mp"]]
        if len(pairs) < self.LOOP_MIN_INLIERS:
            return
        positions = self.mapper.point_positions(pid for pid, _ in pairs)
        obj = np.array([positions[pid] for pid, _ in pairs])
        img = np.float64([kps[t].pt for _, t in pairs])
        T_loop, inliers = solve_pnp(self.camera, obj, img)
        if T_loop is None or inliers.sum() < self.LOOP_MIN_INLIERS:
            return

        # 3) Corrección de la VENTANA LOCAL (el segmento cuyo gauge conocemos:
        # de ahí salen los puntos X_new). Un mismo keypoint del frame actual
        # puede estar asociado a un punto del mapa VIEJO (por el matching del
        # bucle) y a otro del NUEVO (por el tracking local): esos pares de
        # nubes 3D definen la similitud entre gauges.
        loop_by_kp = {kp: pid for (pid, kp), ok in zip(pairs, inliers) if ok}
        shared = [(self._kf["mp"][kp], pid_old)
                  for kp, pid_old in loop_by_kp.items()
                  if kp in self._kf["mp"] and self._kf["mp"][kp] != pid_old]

        segment = (self._kf_ids[-self.local_window:] if self.local_window
                   else [k for k in self._kf_ids if k > old["id"]])
        if len(shared) >= 10:
            from vslam.evaluation import umeyama_alignment
            pos_new = self.mapper.point_positions(pid for pid, _ in shared)
            pos_old = self.mapper.point_positions(pid for _, pid in shared)
            X_new = np.array([pos_new[pid] for pid, _ in shared])
            X_old = np.array([pos_old[pid] for _, pid in shared])
            s, R_u, t_u = umeyama_alignment(X_new, X_old)   # nuevo → viejo
        else:
            # Sin nube compartida suficiente: corrección rígida que lleva la
            # pose actual exactamente a la medición del bucle (sin escala).
            delta = T_loop @ invert_se3(self.T_w_c)
            s, R_u, t_u = 1.0, delta[:3, :3], delta[:3, 3]
        self.mapper.apply_similarity(segment, s, R_u, t_u)

        self.T_w_c = self.mapper.keyframe_pose(cur_id)
        self._T_prev = self.T_w_c.copy()
        self._kf["T"] = self.T_w_c.copy()

        self._last_loop_frame = self._frame_idx
        self.loop_events.append((self._frame_idx, old["id"]))
        info["state"] = "TRACK+KF+LOOP"

    # ── auxiliares ─────────────────────────────────────────────────────────────

    def _coast(self, info) -> None:
        """Fallo de tracking: modelo de velocidad constante (como examples/01).
        v0.3 intentará RELOCALIZAR contra el mapa antes de rendirse."""
        self.T_w_c = self.T_w_c @ self._T_rel
        self._T_prev = self.T_w_c.copy()
        info["state"] = "COAST"

    def _project(self, points_w: np.ndarray) -> np.ndarray:
        T_c_w = invert_se3(self.T_w_c)
        pts_c = (T_c_w[:3, :3] @ points_w.T).T + T_c_w[:3, 3]
        return self.camera.project(pts_c)
