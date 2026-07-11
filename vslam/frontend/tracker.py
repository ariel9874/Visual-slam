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

import queue
import threading
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np

from vslam.backend.bundle_adjustment import local_bundle_adjustment
from vslam.backend.pose_graph import GaussNewtonPoseGraph
from vslam.core.camera import PinholeCamera
from vslam.core.frame import Frame
from vslam.core.geometry import invert_se3, solve_pnp, triangulate_two_views
from vslam.frontend.features import create_extractor
from vslam.frontend.matching import create_matcher
from vslam.frontend.place_recognition import BagOfVisualWords
from vslam.mapping.sparse import SparsePointMapper

# Popcount por byte: Hamming(a, b) = Σ popcount(a XOR b) sobre los 32 bytes de
# un descriptor ORB. Con esta tabla la distancia de un descriptor a un conjunto
# es vectorizada (sin bucle Python) — la base del matching guiado (v0.45).
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)

# Núcleo C++ (v0.5): si el módulo compilado está disponible (cpp/ + pybind11),
# el matching guiado usa la ruta rápida — un gemelo EXACTO de la referencia
# Python (tests/test_guided_match_cpp.py). Sin él, todo funciona igual en Python.
try:
    import vslam_cpp as _fast_cpp
except ImportError:                                  # pragma: no cover
    _fast_cpp = None


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
    # Piso de salud para INSERTAR un keyframe (lección 8): un KF desde pose
    # incierta triangula puntos basura. En sintético = 3×MIN_PNP_INLIERS (45),
    # con tracking sano de 100-350 inliers. En datos reales el tracking sano
    # ronda 20-52 inliers: 45 AHOGA el mapa (inanición de KFs → colapso, medido
    # en fr2_desk frame 1565). Perilla de re-calibración por dataset (v0.45).
    KF_HEALTH_INLIERS = 45
    CHEIRALITY_DIST_THRESH = 2000.0  # ver la "TRAMPA CLÁSICA" en examples/01
    BA_WINDOW = 5               # keyframes en la ventana del BA local
    BA_ITERATIONS = 6
    GBA_ITERATIONS = 50         # BA global offline: converge lento en mapas grandes
    #                             (fr2_xyz 246 KFs: 10 iters→12 cm, 50→0.4 cm; medido)
    # ── cierre de bucle (v0.35) ──
    LOOP_TEMPORAL_GAP = 60      # frames mínimos de antigüedad del candidato
    LOOP_MIN_MATCHES = 200      # matches brutos para considerar un candidato
    LOOP_MIN_INLIERS = 40       # inliers PnP de la verificación geométrica
    LOOP_COOLDOWN = 40          # frames sin reintentar tras cerrar un bucle
    # ── relocalización + compuerta de movimiento (v0.4b) ──
    RELOC_AFTER = 3             # frames en coast antes de intentar relocalizar
    RELOC_MIN_MATCHES = 150     # más laxo que LOOP: aquí no hay bucle sin sentido
    RELOC_MIN_INLIERS = 40      # inliers PnP para aceptar la relocalización
    GATE_MIN_SAMPLES = 20       # historial mínimo antes de activar la compuerta
    GATE_STEP_FACTOR = 6.0      # rechazar pasos > FACTOR × percentil 95 del historial
    GATE_HISTORY = 200          # ventana del historial de pasos aceptados
    # ── matching guiado por reproyección (v0.45) ──
    GUIDED_RADIUS_PX = 15.0     # ventana de búsqueda alrededor del punto proyectado
    GUIDED_MAX_HAMMING = 64     # distancia ORB máxima aceptable (ORB-SLAM: TH_LOW 50)
    GUIDED_MAX_L2 = 0.7         # ídem para descriptores float (SuperPoint, provisional)
    # ── reconocimiento de lugar por BoW (v0.5) ──
    BOW_WORDS = 512             # tamaño del vocabulario visual (en sesión)
    BOW_TRAIN_KFS = 5           # entrenar el vocabulario al llegar a este nº de KFs
    BOW_TOP_K = 5               # candidatos que pagan verificación geométrica
    # ── RGB-D (v0.6) ──
    DEPTH_MIN = 0.3             # rango útil del sensor (Kinect: 0.3-8 m; fuera
    DEPTH_MAX = 8.0             # de él la profundidad es ruido o 0 = sin dato)
    DEPTH_MAX_NEW_POINTS = 400  # tope de puntos nuevos por KF desde profundidad
    #                             (sin tope, ~1500 kps/KF inflan el mapa; el
    #                             culling limpia después, pero mejor no crear basura)
    STEREO_BF = 40.0            # fx·b de la cámara derecha VIRTUAL (px·m): la
    #                             profundidad entra al BA como u_R = u − bf/z
    #                             (teoría en bundle_adjustment.py). 40 ≈ fx≈520
    #                             × b≈7.5 cm del par IR-RGB del Kinect — el
    #                             mismo valor que usa ORB-SLAM2 para TUM.

    def __init__(self, camera: PinholeCamera, extractor=None, matcher=None,
                 mapper: Optional[SparsePointMapper] = None,
                 local_window: Optional[int] = None,
                 local_ba: bool = True,
                 loop_closure: bool = False,
                 ba_backend: str = "numpy",
                 async_mapping: bool = False) -> None:
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
        # LightGlue solo empareja 2D-2D (necesita keypoints de AMBOS lados y el
        # tamaño de imagen: su atención es espacial). Pero el matching 3D-2D
        # contra el mapa y la validación de 3ª vista no tienen kps del lado del
        # mapa → hace falta un matcher por descriptor para esos dos casos. Con
        # LightGlue se usa uno de ratio; con un matcher clásico, el mismo (sin
        # cambio de comportamiento). Ver los usos marcados con _desc_matcher.
        self._desc_matcher = (create_matcher("ratio")
                              if getattr(self.matcher, "name", "") == "lightglue"
                              else self.matcher)
        # Ruta C++ del matching guiado (v0.5): auto si el módulo compilado
        # existe; poner False para forzar la referencia Python (equivalencia).
        self.use_cpp = _fast_cpp is not None
        # Reconocimiento de lugar por BoW (v0.5): baja el coste del candidato
        # de bucle/reloc de O(KFs)·knnMatch a un query de ~3 ms (lección 34).
        # use_bow=False fuerza la fuerza bruta original (referencia).
        self.use_bow = True
        self._bow = BagOfVisualWords(self.BOW_WORDS)

        # HILO DE MAPEO (v0.5, arquitectura ORB-SLAM): con async_mapping=True,
        # el bloque pesado del keyframe (BA + cierre de bucle + culling — el
        # perfil lo midió en ~320 ms, dominado por el matching del bucle) corre
        # en un worker; el hilo de tracking solo triangula e inserta (~40 ms).
        # El lock protege las SECCIONES DE ESCRITURA/LECTURA del mapa (los
        # cómputos pesados corren fuera de él); en modo síncrono queda sin
        # contención (coste despreciable). Las correcciones grandes del worker
        # (bucle) llegan al tracking como un DELTA pendiente que se aplica al
        # inicio del siguiente frame (mismo patrón que reloc/GBA).
        self.async_mapping = async_mapping
        self._map_lock = threading.RLock()
        self._pending_pose_delta: Optional[np.ndarray] = None
        self.map_failures = 0            # excepciones capturadas del worker
        self._map_queue: Optional[queue.Queue] = None
        if async_mapping:
            self._map_queue = queue.Queue()
            self._map_thread = threading.Thread(
                target=self._mapping_worker, daemon=True, name="vslam-mapping")
            self._map_thread.start()
        # Ojo: `mapper or ...` sería un bug — un mapper VACÍO define __len__=0
        # y Python lo evalúa como falsy, creando silenciosamente otro objeto.
        self.mapper = mapper if mapper is not None else SparsePointMapper()
        self.local_window = local_window
        self.local_ba = local_ba
        self.loop_closure = loop_closure
        # Backend del BA (v0.5): "numpy" (referencia didáctica), "gtsam" (batch
        # C++ — el BA era el 57% del tiempo) o "isam2" (INCREMENTAL: ~14 ms/KF
        # constantes vs ~300 del batch; ver gtsam_isam2.py). El BA global
        # offline usa siempre la variante batch correspondiente.
        self._isam2 = None
        self._isam2_cursor: Dict[int, int] = {}
        if ba_backend == "gtsam":
            from vslam.backend.gtsam_ba import gtsam_bundle_adjustment
            self._ba = gtsam_bundle_adjustment
        elif ba_backend == "isam2":
            from vslam.backend.gtsam_ba import gtsam_bundle_adjustment
            from vslam.backend.gtsam_isam2 import ISAM2LocalBA
            self._isam2 = ISAM2LocalBA(camera)
            self._ba = gtsam_bundle_adjustment   # GBA offline: batch
        else:
            self._ba = local_bundle_adjustment
        self._kf_ids: list = []          # keyframes en orden de inserción
        self._kf_db: list = []           # historial para reconocimiento de lugar
        self._metric = False             # mapa en METROS (init RGB-D, v0.6)
        self._last_loop_frame = -10 ** 9
        self.loop_events: list = []      # [(frame, kf_antiguo)] para informes
        self.reloc_events: list = []     # [(frame, kf_reconocido)] (v0.4b)

        self.T_w_c = np.eye(4)
        self._T_prev = np.eye(4)         # pose anterior (para velocidad constante)
        self._T_rel = np.eye(4)          # último movimiento (coasting)
        self._initialized = False
        self._frame_idx = -1
        self._coast_count = 0            # frames consecutivos en coast (v0.4b)
        self._step_history: list = []    # ||paso|| de poses aceptadas (compuerta)
        self._local_ref_kf: Optional[int] = None  # ancla del mapa local tras reloc
        self._depth: Optional[np.ndarray] = None  # profundidad del frame (v0.6)

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

    def keyframe_trajectory(self) -> list:
        """Trayectoria FINAL de keyframes: (frame_id, T_w_c) con las poses
        OPTIMIZADAS del mapper (tras cierres de bucle + BA global).

        Es la métrica estándar de un SLAM con cierre de bucle (la que reporta
        ORB-SLAM): las poses ONLINE se emiten frame a frame ANTES de las
        correcciones y no las reflejan — medir sobre ellas oculta todo el
        beneficio del backend (medido en fr2_desk: online 22 cm vs final 2 cm).
        """
        return [(k, self.mapper.keyframe_pose(k)) for k in self._kf_ids]

    def process_frame(self, gray: np.ndarray,
                      depth: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, Dict]:
        """Procesa un frame; devuelve (T_w_c, info) — misma forma que el
        pipeline del ejemplo 01, para poder compararlos en el benchmark.

        `depth` (v0.6, RGB-D): mapa de profundidad en METROS alineado al gris
        (0 = sin dato). Con él, la inicialización es instantánea y métrica y
        los puntos nuevos de keyframe nacen por retro-proyección — la escala
        deja de ser un gauge y pasa a ser una MEDICIÓN.
        """
        self._frame_idx += 1
        self._depth = depth
        # Corrección pendiente del worker de mapeo (cierre de bucle async): el
        # mundo se movió bajo nuestros pies → transformar el estado del tracking
        # al marco corregido y descartar la velocidad (ya no vale).
        delta = self._pending_pose_delta
        if delta is not None:
            self._pending_pose_delta = None
            self.T_w_c = delta @ self.T_w_c
            self._T_prev = delta @ self._T_prev
            self._T_rel = np.eye(4)
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

    def _initialize_rgbd(self, kps, desc, info) -> bool:
        """INIT RGB-D (v0.6): mapa MÉTRICO instantáneo desde un solo frame.

        ─── La matemática: la escala deja de ser gauge ───
        En monocular, (T, {X}) y (s·T, {s·X}) explican las mismas imágenes: la
        escala es un grado de libertad no observable y se FIJA por convención
        (mediana = 1). La profundidad lo cambia todo: z es una MEDICIÓN en
        metros, y la retro-proyección  X_c = z·K⁻¹·[u, v, 1]ᵀ  da puntos en la
        unidad del sensor. Nada de esperar paralaje, nada de matriz esencial,
        nada de twisted pair (lecciones 1-2 son problemas ESTRICTAMENTE
        monoculares): el primer frame con profundidad válida YA es un mapa.
        Los puntos nacen con UNA observación (no hay segunda vista): el BA los
        excluye hasta que el tracking los re-observe (lección 7) y el buffer de
        pendientes de iSAM2 los retiene igual — el diseño existente ya cubría
        este caso.
        """
        depth = self._depth
        h, w = depth.shape
        px, zs, idx = [], [], []
        for i, kp in enumerate(kps):
            u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
            if 0 <= u < w and 0 <= v < h:
                z = float(depth[v, u])
                if self.DEPTH_MIN < z < self.DEPTH_MAX:
                    px.append(kp.pt)
                    zs.append(z)
                    idx.append(i)
        if len(idx) < self.MIN_INIT_POINTS:
            return False                 # profundidad insuficiente: reintentar

        pts_w = self.camera.backproject(np.float64(px), np.float64(zs))
        kf_id = self._frame_idx          # el mundo ES esta cámara (T = I, metros)
        with self._map_lock:
            self.mapper.integrate_keyframe(Frame(frame_id=kf_id, timestamp=0.0,
                                                 T_w_c=np.eye(4), is_keyframe=True))
            ids = self.mapper.add_points(pts_w, desc[idx], anchor_kf_id=kf_id)
            self.mapper.add_observations(kf_id, ids,
                                         self._with_virtual_right(np.float64(px)))
        self._kf_ids = [kf_id]
        mp = dict(zip(idx, ids))
        self._kf_db = [{"id": kf_id, "kps": kps, "desc": desc, "mp": mp}]
        self.T_w_c = np.eye(4)
        self._T_prev = np.eye(4)
        self._kf = {"id": kf_id, "kps": kps, "desc": desc, "mp": mp,
                    "T": np.eye(4)}
        self._kf_inliers = len(ids)
        self._frames_since_kf = 0
        self._coast_count = 0
        self._initialized = True
        self._metric = True              # la escala ya no es gauge: es medición
        info.update(tracked=True, n_inliers=len(ids), state="INIT-OK")
        return True

    def _initialize_step(self, gray, kps, desc, info) -> None:
        cv2 = self._cv2
        info["state"] = "INIT"
        # RGB-D (v0.6): con profundidad no hay danza de dos vistas.
        if self._depth is not None and self._initialize_rgbd(kps, desc, info):
            return
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
        self._coast_count = 0
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

        matches = self._desc_matcher.match(descriptors, mid_desc, None, mid_kps, None)
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

    def _local_kfs(self) -> Optional[list]:
        """El mapa local (v0.4): keyframes recientes ∪ COVISIBLES del último.

        La recencia da continuidad; la covisibilidad re-incorpora los
        keyframes antiguos cuando se re-visita su zona (y con ellos, sus
        puntos: el matching los asocia en lugar de re-triangular duplicados).
        El puente entre segmentos lo tiende el cierre de bucle, que registra
        sus pares verificados como observaciones del keyframe actual.
        """
        if self.local_window is None:
            return None
        recent = self._kf_ids[-self.local_window:]
        anchors = set(recent)
        if recent:
            anchors |= set(self.mapper.covisible_kfs(recent[-1]))
        # Tras una RELOCALIZACIÓN, re-anclar el mapa local en el KF reconocido y
        # su covisibilidad, no solo en la recencia temporal — que tras un salto
        # apunta a la zona de ANTES del secuestro. Sin esto el tracking no puede
        # continuar tras reloc (el mapa local no cubre dónde estamos): medido, el
        # matching guiado lo destapó (el secuestro reincidía en coast/reloc). Se
        # limpia al insertar el próximo keyframe (la recencia vuelve a valer).
        if self._local_ref_kf is not None:
            anchors.add(self._local_ref_kf)
            anchors |= set(self.mapper.covisible_kfs(self._local_ref_kf))
        return sorted(anchors)

    @staticmethod
    def _desc_distances(one: np.ndarray, many: np.ndarray) -> np.ndarray:
        """Distancia de un descriptor a un conjunto (K,): Hamming si es binario
        (ORB, vía la tabla de popcount), L2 si es float (SuperPoint)."""
        if one.dtype == np.uint8:
            xor = np.bitwise_xor(one[None, :], many)          # (K, 32)
            return _POPCOUNT8[xor].sum(axis=1)
        return np.linalg.norm(many.astype(np.float64) - one, axis=1)

    def _guided_match(self, kps, desc, T_pred, map_pts, map_desc):
        """Matching GUIADO por reproyección (v0.45): en vez de comparar todos
        los descriptores contra todos, se PREDICE la pose (velocidad constante),
        se proyecta el mapa local a la imagen y cada punto se busca solo entre
        los keypoints dentro de un radio pequeño.

        ─── La matemática / por qué gana ───
        El matching global por descriptor ignora la geometría: en datos reales,
        contra un mapa grande, produce muchas asociaciones ambiguas (ORB tiene
        vecinos casuales a esa escala → 0 inliers, lección 22). Un prior de pose
        —aunque sea burdo (constante)— restringe cada punto a una ventana de ~15
        px: dentro de ella el vecino correcto casi no tiene rival, así que suben
        los inliers verdaderos Y baja el ruido. Es el "track local map" de
        ORB-SLAM. Asignación GREEDY por distancia ascendente: cada keypoint y
        cada punto del mapa se usan una sola vez (evita correspondencias en
        conflicto que envenenarían el PnP). Si el prior es malo (tras reloc o un
        salto), el guiado rinde poco y el llamador cae al matching global.
        """
        cv2 = self._cv2
        h, w = (self.camera.height or 10 ** 9), (self.camera.width or 10 ** 9)

        # Ruta C++ (v0.5): gemela exacta de lo de abajo, ~2 órdenes más rápida
        # (era el 37% del frame). tests/test_guided_match_cpp.py verifica la
        # equivalencia par a par contra esta referencia Python.
        if self.use_cpp and _fast_cpp is not None and len(map_pts) and len(kps):
            fn = None
            if desc.dtype == np.uint8:
                fn, lim = _fast_cpp.guided_match_hamming, float(self.GUIDED_MAX_HAMMING)
            elif desc.dtype == np.float32:
                fn, lim = _fast_cpp.guided_match_l2, float(self.GUIDED_MAX_L2)
            if fn is not None:
                kp_xy = np.array([kp.pt for kp in kps], dtype=np.float64)
                mi, kj, dd = fn(kp_xy, desc, np.asarray(T_pred, np.float64),
                                np.asarray(map_pts, np.float64), map_desc,
                                self.camera.fx, self.camera.fy,
                                self.camera.cx, self.camera.cy,
                                float(w), float(h),
                                float(self.GUIDED_RADIUS_PX), lim)
                return [cv2.DMatch(int(i), int(j), float(d))
                        for i, j, d in zip(mi, kj, dd)]

        T_c_w = invert_se3(T_pred)
        pc = (T_c_w[:3, :3] @ map_pts.T).T + T_c_w[:3, 3]      # mapa en cámara pred.
        uv = self.camera.project(pc)                          # (M, 2)
        kp_xy = np.array([kp.pt for kp in kps], dtype=np.float64)  # (N, 2)
        r2 = self.GUIDED_RADIUS_PX ** 2
        max_dist = self.GUIDED_MAX_HAMMING if desc.dtype == np.uint8 else self.GUIDED_MAX_L2

        cand_pairs = []      # (dist, idx_mapa, idx_kp)
        visible = (pc[:, 2] > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < w) \
            & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        for i in np.flatnonzero(visible):
            near = np.flatnonzero(np.sum((kp_xy - uv[i]) ** 2, axis=1) <= r2)
            if not len(near):
                continue
            dists = self._desc_distances(map_desc[i], desc[near])
            k = int(np.argmin(dists))
            if dists[k] <= max_dist:
                cand_pairs.append((float(dists[k]), int(i), int(near[k])))

        cand_pairs.sort()
        used_kp, used_mp, out = set(), set(), []
        for dist, i, j in cand_pairs:
            if i in used_mp or j in used_kp:
                continue
            used_mp.add(i); used_kp.add(j)
            out.append(cv2.DMatch(i, j, dist))
        return out

    def _track_step(self, gray, kps, desc, info) -> None:
        info["state"] = "TRACK"
        # Mapa global (v0.2) o LOCAL por recencia+covisibilidad (v0.4). El lock
        # evita leer el mapa a medio corregir por el worker de mapeo (async).
        with self._map_lock:
            ids, map_pts, map_desc = self.mapper.snapshot(self._local_kfs())

        # Matching GUIADO por reproyección (v0.45) si hay prior de movimiento;
        # si rinde poco (tras reloc, salto, o al inicio), CAE al global por
        # descriptor. El guiado sube los inliers reales en datos reales (lección
        # 22) — la palanca contra la inanición de KFs y la deriva.
        matches = []
        if len(map_pts):
            T_pred = self._T_prev @ self._T_rel
            matches = self._guided_match(kps, desc, T_pred, map_pts, map_desc)
        info["guided"] = len(matches)
        if len(matches) < self.MIN_MAP_MATCHES:
            # Fallback 3D-2D por descriptor (el mapa no tiene kps → no LightGlue).
            matches = self._desc_matcher.match(map_desc, desc, None, kps, gray.shape)
        info["n_matches"] = len(matches)
        if len(matches) < self.MIN_MAP_MATCHES:
            self._coast(gray, kps, desc, info)
            return

        obj = map_pts[[m.queryIdx for m in matches]]
        img = np.float64([kps[m.trainIdx].pt for m in matches])

        T_w_c, inlier_mask = solve_pnp(self.camera, obj, img)
        if T_w_c is None or inlier_mask.sum() < self.MIN_PNP_INLIERS:
            self._coast(gray, kps, desc, info)
            return

        # COMPUERTA DE MOVIMIENTO (v0.4b, ahora SÍ — emparejada con reloc).
        # ─── La matemática / la lección medida ───
        # En v0.4 esta compuerta se probó SOLA y cortaba por ambos lados:
        # bloqueaba los teletransportes falsos Y las recuperaciones legítimas
        # tras un tramo de deriva (ATE 8.4 → 37.7 cm), y con referencia de
        # ventana reciente se auto-congelaba tras una pausa (→202 cm). Se
        # retiró. La cura no era ajustar el umbral: era darle una SALIDA. Al
        # rechazar un paso anómalo ya no nos quedamos ciegos — caemos a _coast,
        # cuyo contador dispara la RELOCALIZACIÓN, que decide con verificación
        # geométrica GLOBAL a qué pose volver (el paso legítimo grande se
        # re-acepta vía reloc; el salto espurio no encuentra soporte y se
        # descarta). Umbral robusto por percentil, no absoluto: 6 × el p95 de
        # los pasos aceptados (con ≥ 20 muestras) — generoso con la dinámica
        # real, cerrado a los saltos de un modo falso del PnP.
        step = float(np.linalg.norm(T_w_c[:3, 3] - self._T_prev[:3, 3]))
        if len(self._step_history) >= self.GATE_MIN_SAMPLES:
            p95 = float(np.percentile(self._step_history, 95))
            if step > self.GATE_STEP_FACTOR * p95:
                info["state"] = "GATE-REJECT"
                self._coast(gray, kps, desc, info)
                return

        # Pose absoluta contra el mapa: no se apila sobre la anterior.
        self._T_rel = invert_se3(self._T_prev) @ T_w_c
        self._T_prev = T_w_c.copy()
        self.T_w_c = T_w_c
        self._coast_count = 0
        self._step_history.append(step)
        self._step_history = self._step_history[-self.GATE_HISTORY:]
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
        healthy = n_inliers >= self.KF_HEALTH_INLIERS
        if self._frames_since_kf >= self.KF_MIN_GAP and healthy \
                and (starving or overdue):
            # kp del frame → id GLOBAL del punto (ids traduce índices del
            # snapshot — que puede ser un subconjunto local — a ids del mapa).
            matched_frame_idx = {m.trainIdx: int(ids[m.queryIdx])
                                 for m, ok in zip(matches, inlier_mask) if ok}
            self._insert_keyframe(gray, kps, desc, matched_frame_idx, info)

    def _with_virtual_right(self, px: np.ndarray) -> np.ndarray:
        """Píxeles (N, 2) → (N, 3) añadiendo la coordenada derecha virtual
        u_R = u − bf/z desde el mapa de profundidad del frame ACTUAL (teoría en
        bundle_adjustment.py, § estéreo virtual). u_R = NaN donde el píxel no
        tiene z válida → el BA cae al residuo 2D para esa observación. Solo se
        llama con el frame en mano (init/inserción de KF): el puente del bucle
        NO la usa — en modo asíncrono `self._depth` puede ser ya de un frame
        más nuevo que el KF del bucle y fabricaría mediciones falsas."""
        px = np.asarray(px, np.float64).reshape(-1, 2)
        u_r = np.full(len(px), np.nan)
        depth = self._depth
        h, w = depth.shape
        for n, (u, v) in enumerate(px):
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < w and 0 <= vi < h:
                z = float(depth[vi, ui])
                if self.DEPTH_MIN < z < self.DEPTH_MAX:
                    u_r[n] = u - self.STEREO_BF / z
        return np.column_stack([px, u_r])

    def _insert_keyframe(self, gray, kps, desc, matched_frame_idx, info) -> None:
        """Promueve el frame a keyframe y crea puntos NUEVOS: por
        retro-proyección de la profundidad (RGB-D, v0.6) o triangulando contra
        el keyframe anterior (monocular)."""
        kf_id = self._frame_idx
        # Asociaciones del nuevo keyframe: kp del frame → id global del punto.
        mp = dict(matched_frame_idx)

        if self._depth is not None and self._metric:
            # PUNTOS DESDE PROFUNDIDAD (v0.6): cada keypoint NO asociado al mapa
            # (SOLO en mapa MÉTRICO: estos puntos nacen en METROS — inyectarlos
            # en un mapa a escala gauge (init monocular) crea un mapa de DOS
            # escalas en tensión permanente; lo medimos en fr1_desk cuando la
            # profundidad arrancaba tarde y la init caía a monocular)
            # y con z válida se retro-proyecta al mundo — métrico, sin baseline
            # (por eso RGB-D no sufre el fallo de fr1 handheld: crear mapa no
            # requiere paralaje). Nacen con la observación de ESTE keyframe
            # (abajo, con todo mp); la 2ª llega al re-observarlos (lección 7 /
            # buffer de iSAM2). El filtro anti-duplicados aplica igual.
            depth = self._depth
            h, w = depth.shape
            cand = []
            for i, kp in enumerate(kps):
                if i in matched_frame_idx:
                    continue
                u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
                if 0 <= u < w and 0 <= v < h:
                    z = float(depth[v, u])
                    if self.DEPTH_MIN < z < self.DEPTH_MAX:
                        cand.append((i, kp.pt[0], kp.pt[1], z))
            if len(cand) > self.DEPTH_MAX_NEW_POINTS:
                stride = int(np.ceil(len(cand) / self.DEPTH_MAX_NEW_POINTS))
                cand = cand[::stride]    # submuestreo uniforme (tope de mapa)
            if cand:
                idx = np.array([c[0] for c in cand])
                px = np.float64([(c[1], c[2]) for c in cand])
                zs = np.float64([c[3] for c in cand])
                pts_c = self.camera.backproject(px, zs)
                pts_w = (self.T_w_c[:3, :3] @ pts_c.T).T + self.T_w_c[:3, 3]
                valid = np.ones(len(pts_w), dtype=bool)
                with self._map_lock:
                    _, map_pts, _ = self.mapper.snapshot(self._local_kfs())
                if len(map_pts):
                    cam = self.T_w_c[:3, 3]
                    for n in range(len(pts_w)):
                        d_min = np.min(np.linalg.norm(map_pts - pts_w[n], axis=1))
                        if d_min < 0.015 * np.linalg.norm(pts_w[n] - cam):
                            valid[n] = False
                if valid.any():
                    with self._map_lock:
                        ids = self.mapper.add_points(pts_w[valid],
                                                     desc[idx[valid]],
                                                     anchor_kf_id=kf_id)
                    mp.update(dict(zip(idx[valid].tolist(), ids)))
            fresh = []                   # sin triangulación 2-vistas en RGB-D
        else:
            kf = self._kf
            matches = self.matcher.match(kf["desc"], desc, kf["kps"], kps,
                                         gray.shape)
            fresh = [m for m in matches
                     if m.queryIdx not in kf["mp"]
                     and m.trainIdx not in matched_frame_idx]

        if len(fresh) >= 10:
            pts0 = np.float64([kf["kps"][m.queryIdx].pt for m in fresh])
            pts1 = np.float64([kps[m.trainIdx].pt for m in fresh])
            # Las poses ya viven en el gauge del mapa → los puntos nuevos
            # heredan la escala automáticamente (no se re-normaliza nada).
            points_w, valid = triangulate_two_views(
                self.camera, kf["T"], self.T_w_c, pts0, pts1)
            # FILTRO ANTI-DUPLICADOS (v0.4): si el candidato cae a < 1.5% de
            # su profundidad de un punto YA existente, es (casi seguro) la
            # misma característica física re-triangulada — típico al re-visitar
            # una zona: el matching al mapa falla para algunos keypoints y sin
            # este filtro nacen nubes duplicadas desplazadas por la deriva,
            # que vuelven BIESTABLE al PnP (medido: teleports de 0.25 u con
            # cientos de "inliers" coherentes del modo falso). Se descarta la
            # CREACIÓN, no se asocia: descartar no puede inventar observaciones
            # falsas (la fusión por proyección que probamos sí podía — en
            # textura densa los descriptores vecinos están correlacionados y
            # envenenaba el BA: ATE 8 → 202 cm).
            if valid.any():
                with self._map_lock:
                    _, map_pts, _ = self.mapper.snapshot(self._local_kfs())
                if len(map_pts):
                    cam = self.T_w_c[:3, 3]
                    for n in np.flatnonzero(valid):
                        d_min = np.min(np.linalg.norm(map_pts - points_w[n], axis=1))
                        depth = np.linalg.norm(points_w[n] - cam)
                        if d_min < 0.015 * depth:
                            valid[n] = False
            if valid.any():
                idx_curr = np.array([m.trainIdx for m in fresh])[valid]
                with self._map_lock:
                    ids = self.mapper.add_points(points_w[valid], desc[idx_curr],
                                                 anchor_kf_id=kf_id)
                    # Registrar TAMBIÉN la observación en el keyframe previo (el
                    # otro extremo de la triangulación). Sin ella el punto queda
                    # con una sola observación y el BA puede deslizarlo libremente
                    # a lo largo de su rayo visual (C_p de rango 2: 3 incógnitas,
                    # 2 ecuaciones) — lo medimos: el BA EMPEORABA el ATE por esto.
                    self.mapper.add_observations(kf["id"], ids, pts0[valid])
                mp.update(dict(zip(idx_curr.tolist(), ids)))

        with self._map_lock:
            self.mapper.integrate_keyframe(Frame(frame_id=kf_id, timestamp=0.0,
                                                 T_w_c=self.T_w_c.copy(),
                                                 is_keyframe=True))
            if mp:
                kp_idx = list(mp.keys())
                obs_px = np.float64([kps[i].pt for i in kp_idx])
                if self._depth is not None and self._metric:
                    # RGB-D: cada observación lleva su u_R — TAMBIÉN las de
                    # puntos viejos re-observados: esa es la medición métrica
                    # fresca que ancla la estructura en el BA (v0.6 hito 2).
                    obs_px = self._with_virtual_right(obs_px)
                self.mapper.add_observations(kf_id, list(mp.values()), obs_px)
        self._kf_ids.append(kf_id)
        self._local_ref_kf = None        # un KF nuevo re-centra el mapa local por recencia
        self._kf = {"id": kf_id, "kps": kps, "desc": desc, "mp": mp,
                    "T": self.T_w_c.copy()}
        self._kf_db.append({"id": kf_id, "kps": kps, "desc": desc, "mp": mp})
        if self.use_bow:
            with self._map_lock:
                if not self._bow.trained and len(self._kf_db) >= self.BOW_TRAIN_KFS:
                    # Entrenamiento ÚNICO del vocabulario (~50 ms medidos) con
                    # los descriptores de los primeros keyframes de la sesión;
                    # acto seguido se indexan los KFs ya acumulados.
                    self._bow.fit(np.vstack([e["desc"] for e in self._kf_db]))
                    for e in self._kf_db:
                        self._bow.add(e["id"], e["desc"])
                elif self._bow.trained:
                    self._bow.add(kf_id, desc)
        self._kf_inliers = max(info["n_inliers"], 1)
        self._frames_since_kf = 0
        info["state"] = "TRACK+KF"

        if self.async_mapping:
            # HILO DE MAPEO (v0.5): BA + bucle + culling van al worker. Se pasa
            # el mp del KF (el worker lo necesita para la escala del bucle) —
            # self._kf habrá avanzado cuando el job se procese.
            self._map_queue.put((gray, kps, desc, kf_id, mp))
            return

        if self.local_ba:
            self._run_local_ba()
        if self.loop_closure:
            self._try_close_loop(gray, kps, desc, info)

        # CULLING (v0.4b): tras refinar (BA) y cerrar bucles, retirar los puntos
        # que nacieron hace varios keyframes y nadie volvió a observar — casi
        # siempre triangulaciones espurias. Adelgaza el mapa sin tocar la zona
        # activa (la ventana de gracia protege lo recién creado). Ver
        # SparsePointMapper.cull_points para el criterio y la medición.
        with self._map_lock:
            self.mapper.cull_points(self._kf_ids)

    # ── hilo de mapeo (v0.5) ──────────────────────────────────────────────────

    def _mapping_worker(self) -> None:
        """Consume jobs de keyframe: BA local + cierre de bucle + culling —
        exactamente lo que el modo síncrono hace inline (~320 ms medidos, con
        el matching del bucle como pieza dominante), pero sin bloquear el hilo
        de tracking. Una excepción aquí NO tira el sistema: se cuenta en
        map_failures y ese keyframe queda sin refinar (el tracking sigue)."""
        while True:
            job = self._map_queue.get()
            try:
                if job is None:
                    return                           # sentinela de apagado
                gray, kps, desc, kf_id, mp = job
                if self.local_ba:
                    self._run_local_ba(sync=False)
                if self.loop_closure:
                    self._try_close_loop(gray, kps, desc, {"state": ""},
                                         cur_id=kf_id, cur_mp=mp)
                with self._map_lock:
                    self.mapper.cull_points(self._kf_ids)
            except Exception:                        # noqa: BLE001
                self.map_failures += 1
            finally:
                self._map_queue.task_done()

    def wait_mapping(self) -> None:
        """Drena la cola del hilo de mapeo. Llamar antes de leer los resultados
        finales (keyframe_trajectory / evaluación); global_bundle_adjustment lo
        hace solo. En modo síncrono es un no-op."""
        if self.async_mapping and self._map_queue is not None:
            self._map_queue.join()

    def stop_mapping(self) -> None:
        """Apaga el worker (opcional: es daemon; útil en tests/benchmarks)."""
        if self.async_mapping and self._map_queue is not None:
            self._map_queue.put(None)
            self._map_thread.join(timeout=30.0)

    def _run_local_ba(self, sync: bool = True) -> None:
        """Bundle adjustment sobre la ventana de keyframes recientes.

        Se anclan los DOS keyframes más viejos de la ventana: uno fija
        rotación/traslación y el segundo fija la ESCALA — el gauge monocular
        tiene 7 grados de libertad (la lección medida en bundle_adjustment.py).
        Simplificación v0.35: las observaciones desde keyframes fuera de la
        ventana no participan (ORB-SLAM las incluye como cámaras fijas).

        `sync=False` (worker de mapeo): NO tocar el estado del tracking
        (T_w_c/_T_prev/_kf) — el tracking ya avanzó; heredar aquí una pose
        vieja sería un teletransporte hacia atrás. El refinado llega al
        tracking por la vía correcta: el PnP es una medición ABSOLUTA contra
        el mapa, y el mapa sí queda refinado.
        """
        window = self._kf_ids[-self.BA_WINDOW:]
        if len(window) < 3:
            return
        if self._isam2 is not None:
            # Ruta INCREMENTAL (iSAM2, v0.5): alimentar solo las observaciones
            # NUEVAS desde la última llamada. Cursores sobre las listas
            # append-only del mapper (_obs crudo: la vista filtrada de
            # observations() encoge con el culling y rompería los índices).
            # El lock cubre lectura de cursores + update + write-back: el
            # update (~34 ms) bloquea a lo sumo un snapshot del tracking.
            with self._map_lock:
                new_obs = []
                for kf, entries in self.mapper._obs.items():
                    start = self._isam2_cursor.get(kf, 0)
                    if start < len(entries):
                        new_obs.extend((kf, pid, uv) for pid, uv in entries[start:])
                        self._isam2_cursor[kf] = len(entries)
                result = self._isam2.process_keyframe(
                    self.mapper, window, new_obs,
                    stereo_bf=self.STEREO_BF if self._metric else 0.0)
            if result is None:
                return          # update fallido: seguir con la pose del PnP
            opt_poses, opt_points = result
        else:
            with self._map_lock:
                obs = self.mapper.observations(window)
                if len(obs) < 60:
                    return
                kf_poses = {k: self.mapper.keyframe_pose(k) for k in window}
                # Solo se optimizan puntos con ≥ 2 observaciones DENTRO de la
                # ventana: con una, el punto se desliza por su rayo.
                counts: dict = {}
                for _, pid, _ in obs:
                    counts[pid] = counts.get(pid, 0) + 1
                points = self.mapper.point_positions(
                    {pid for pid, c in counts.items() if c >= 2})
            # El solve (lo pesado) corre FUERA del lock: el tracking sigue.
            opt_poses, opt_points = self._ba(
                self.camera, kf_poses, points, obs, fixed_kfs=set(window[:2]),
                iterations=self.BA_ITERATIONS,
                stereo_bf=self.STEREO_BF if self._metric else 0.0)

        with self._map_lock:
            for k, T in opt_poses.items():
                self.mapper.set_keyframe_pose(k, T)
            self.mapper.set_point_positions(opt_points)
        # El keyframe recién insertado ES el frame actual: heredar su refinado
        # (solo en modo síncrono — ver docstring).
        cur = self._kf["id"]
        if sync and cur in opt_poses:
            self.T_w_c = opt_poses[cur].copy()
            self._T_prev = self.T_w_c.copy()
            self._kf["T"] = self.T_w_c.copy()

    def _match_against_kf(self, old, gray, kps, desc):
        """Empareja el frame actual contra un keyframe de la base y lo verifica
        con PnP contra los puntos 3D de ese keyframe. Es el mecanismo común del
        CIERRE DE BUCLE y de la RELOCALIZACIÓN (v0.4b): ambos reconocen un lugar
        y confirman la geometría; solo cambia el filtro de candidatos (el bucle
        exige antigüedad temporal, la reloc no) y los umbrales.

        Devuelve (n_matches, pairs, T_pnp, inliers), donde `pairs` son los
        (point_id_del_kf, idx_kp_actual) con correspondencia al mapa y `T_pnp`
        es la pose del frame actual EN EL MARCO DEL MAPA (o None si PnP falla).
        """
        matches = self.matcher.match(old["desc"], desc, old["kps"], kps, gray.shape)
        pairs = [(old["mp"][m.queryIdx], m.trainIdx) for m in matches
                 if m.queryIdx in old["mp"]]
        if len(pairs) < self.LOOP_MIN_INLIERS:
            return len(matches), pairs, None, None
        with self._map_lock:
            positions = self.mapper.point_positions(pid for pid, _ in pairs)
        obj = np.array([positions[pid] for pid, _ in pairs])
        img = np.float64([kps[t].pt for _, t in pairs])
        T_pnp, inliers = solve_pnp(self.camera, obj, img)
        return len(matches), pairs, T_pnp, inliers

    def global_bundle_adjustment(self, iterations: Optional[int] = None) -> None:
        """BUNDLE ADJUSTMENT GLOBAL: re-optimiza poses Y puntos de TODO el mapa
        desde los residuos de reproyección (v0.45). Es un refinamiento OFFLINE:
        el llamador lo invoca UNA vez tras procesar la secuencia, y luego lee
        `keyframe_trajectory()` — no se ejecuta en caliente.

        ─── Por qué OFFLINE y no tras cada bucle ───
        Probado online (tras cada bucle grande): el BA global sobre un mapa
        grande (~240 KFs, BA didáctico) sacude el mapa y descarrila el tracking
        online — fr2_xyz pasó de 5 a 346 frames perdidos, ni el cooldown ni
        re-anclar el mapa local lo salvaban (el problema es el BA a esa escala
        corriendo repetido, no solo el salto de pose). Como solo evaluamos la
        trayectoria FINAL de keyframes (lección 25), UN BA al final da el
        beneficio sin tocar el tracking. Es el "full BA" offline de ORB-SLAM.

        ─── Por qué hace falta ADEMÁS del grafo de poses del bucle ───
        El grafo Sim(3) del cierre de bucle corrige POSES dadas las restricciones
        relativas, pero deja los PUNTOS con su posición derivada y no re-estima
        la escala intermedia desde las observaciones — la escala monocular seguía
        divagando (fr2_desk: escala por cuartos 1.48/1.53/1.86/1.20). El BA global
        optimiza puntos y poses juntos, y como el cierre de bucle registró las
        OBSERVACIONES PUENTE (el KF actual ve puntos del segmento viejo), esas
        observaciones ATAN los extremos del bucle → la corrección de escala se
        reparte por la cadena. Medido en fr2_desk (trayectoria final): 4.8→2.0 cm.
        Gauge: se fijan los 2 KFs más viejos (baseline → ancla escala, §BA).
        """
        self.wait_mapping()          # drenar el hilo de mapeo antes del full BA
        kfs = list(self._kf_ids)
        if len(kfs) < 3:
            return
        obs = self.mapper.observations(kfs)
        if len(obs) < 60:
            return
        kf_poses = {k: self.mapper.keyframe_pose(k) for k in kfs}
        counts: dict = {}
        for _, pid, _ in obs:
            counts[pid] = counts.get(pid, 0) + 1
        points = self.mapper.point_positions(
            {pid for pid, c in counts.items() if c >= 2})
        opt_poses, opt_points = self._ba(
            self.camera, kf_poses, points, obs, fixed_kfs=set(kfs[:2]),
            iterations=iterations or self.GBA_ITERATIONS,
            stereo_bf=self.STEREO_BF if self._metric else 0.0)
        for k, T in opt_poses.items():
            self.mapper.set_keyframe_pose(k, T)
        self.mapper.set_point_positions(opt_points)

    def _try_close_loop(self, gray, kps, desc, info,
                        cur_id: Optional[int] = None,
                        cur_mp: Optional[dict] = None) -> None:
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
        3. CORRECCIÓN por grafo de poses Sim(3) (v0.4): la deriva monocular
           incluye ESCALA (14% medido sin BA local) y una corrección SE(3)
           no puede absorberla — reparte la inconsistencia como error de
           traslación y EMPEORA el resultado (medido: ATE 35.9 → 94.1 cm en
           v0.35). Es el motivo por el que ORB-SLAM cierra bucles monoculares
           en Sim(3) (Strasdat et al., RSS 2010) — reproducido en
           tests/test_pose_graph.py. El factor de bucle lleva DOS mediciones:
           la pose por PnP (rotación/traslación en el gauge antiguo) y la
           escala relativa por Umeyama sobre los puntos que existen dos veces
           en el mapa (uno por segmento). El grafo distribuye pose Y escala
           por la cadena, y update_poses_sim3 re-ancla/re-escala el mapa.
        """
        if self._frame_idx - self._last_loop_frame < self.LOOP_COOLDOWN:
            return
        # En modo async el job trae SU keyframe (self._kf ya avanzó); en modo
        # síncrono cur_id/cur_mp son los del keyframe actual.
        async_job = cur_id is not None
        if cur_id is None:
            cur_id = self._kf["id"]
        if cur_mp is None:
            cur_mp = self._kf["mp"]

        # 1+2) Reconocimiento de lugar + verificación geométrica en un solo
        # paso, vía el helper compartido con la relocalización (§_match_against_kf).
        # Con BoW (v0.5, lección 34): el índice invertido propone TOP-K
        # candidatos en ~3 ms y SOLO esos pagan el matching completo — antes se
        # matcheaba contra TODA la base (el cuello medido del keyframe, lección
        # 32). Mientras no haya vocabulario (primeros KFs), fuerza bruta. De los
        # candidatos con suficiente ANTIGÜEDAD (el filtro temporal: parecerse al
        # pasado inmediato es continuidad, no un bucle) nos quedamos con el de
        # más matches brutos que ADEMÁS pase la verificación PnP.
        if self.use_bow and self._bow.trained:
            with self._map_lock:
                ranked = self._bow.query(desc, top_k=self.BOW_TOP_K + 5)
            db_by_id = {e["id"]: e for e in self._kf_db}
            candidates = [db_by_id[k] for k, _ in ranked
                          if k in db_by_id
                          and cur_id - k >= self.LOOP_TEMPORAL_GAP
                          ][:self.BOW_TOP_K]
        else:
            candidates = [old for old in self._kf_db[:-1]
                          if cur_id - old["id"] >= self.LOOP_TEMPORAL_GAP]
        best = None
        for old in candidates:
            n_matches, pairs, T_loop, inliers = self._match_against_kf(
                old, gray, kps, desc)
            if n_matches < self.LOOP_MIN_MATCHES:
                continue
            if T_loop is None or len(pairs) < self.LOOP_MIN_INLIERS \
                    or int(inliers.sum()) < self.LOOP_MIN_INLIERS:
                continue
            if best is None or n_matches > best[0]:
                best = (n_matches, old, pairs, T_loop, inliers)
        if best is None:
            return
        _, old, pairs, T_loop, inliers = best

        # 3) Corrección de la VENTANA LOCAL (el segmento cuyo gauge conocemos:
        # de ahí salen los puntos X_new). Un mismo keypoint del frame actual
        # puede estar asociado a un punto del mapa VIEJO (por el matching del
        # bucle) y a otro del NUEVO (por el tracking local): esos pares de
        # nubes 3D definen la similitud entre gauges.
        loop_by_kp = {kp: pid for (pid, kp), ok in zip(pairs, inliers) if ok}
        shared = [(cur_mp[kp], pid_old)
                  for kp, pid_old in loop_by_kp.items()
                  if kp in cur_mp and cur_mp[kp] != pid_old]

        # ─── La matemática: el grupo del bucle depende de QUIÉN fija la escala ───
        # Monocular: la escala es GAUGE (no observable) y deriva — el bucle debe
        # medirla (Umeyama sobre nubes duplicadas) y el grafo redistribuirla:
        # Sim(3), Strasdat et al. RGB-D: la escala es una MEDICIÓN del sensor;
        # el mapa nace métrico y NO deriva en escala (medido en fr2_xyz: 3669
        # frames sin bucles, escala por ventanas 0.90-1.09, ATE 1.1 cm). Un
        # bucle Sim(3) aquí es veneno que ADEMÁS compone: el s_rel ruidoso del
        # Umeyama re-escala el mapa viejo, los puntos nuevos siguen naciendo
        # métricos, y el siguiente bucle mide esa discrepancia y re-escala otra
        # vez (medido: 22 bucles → escala 2.09, ATE 22 cm). Por eso ORB-SLAM2
        # cierra bucles RGB-D/estéreo en SE(3) y reserva Sim(3) para monocular.
        if not self._metric and len(shared) >= 10:
            from vslam.evaluation import umeyama_alignment
            with self._map_lock:
                pos_new = self.mapper.point_positions(pid for pid, _ in shared)
                pos_old = self.mapper.point_positions(pid for _, pid in shared)
            X_new = np.array([pos_new[pid] for pid, _ in shared])
            X_old = np.array([pos_old[pid] for _, pid in shared])
            s_rel, _, _ = umeyama_alignment(X_new, X_old)   # escala nuevo→viejo
        else:
            s_rel = 1.0     # métrico o sin nube compartida: bucle rígido

        # Grafo Sim(3) sobre TODOS los keyframes (v0.4): la corrección — con
        # su componente de escala — se redistribuye por la cadena en lugar de
        # aplicarse como salto (validado en tests/test_pose_graph.py con el
        # experimento de Strasdat: SE(3) no puede, Sim(3) sí). Los nodos
        # entran como SE(3) embebida (s = 1); la odometría asegura suavidad
        # de escala entre vecinos; el factor de bucle asegura la pose medida
        # por PnP Y la escala medida por Umeyama.
        with self._map_lock:
            kf_ids_snap = list(self._kf_ids)
            poses = {k: self.mapper.keyframe_pose(k) for k in kf_ids_snap}
        dim = 6 if self._metric else 7
        graph = GaussNewtonPoseGraph("se3" if self._metric else "sim3")
        # TODO el segmento antiguo queda FIJO (≤ keyframe del bucle): el
        # cierre corrige al recién llegado, no reescribe el mundo — mover la
        # referencia dejaría la historia ya emitida en otro marco (medido:
        # con solo el nodo 0 fijo, el ATE con BA empeoraba de 6.7 a 87 cm).
        for k in kf_ids_snap:
            graph.add_pose(k, poses[k], fixed=(k <= old["id"]))
        for a, b in zip(kf_ids_snap[:-1], kf_ids_snap[1:]):
            graph.add_odometry_factor(a, b, invert_se3(poses[a]) @ poses[b],
                                      np.eye(dim) * 1e2)
        S_cur_meas = np.eye(4)
        S_cur_meas[:3, :3] = s_rel * T_loop[:3, :3]
        S_cur_meas[:3, 3] = T_loop[:3, 3]
        graph.add_loop_factor(old["id"], cur_id,
                              invert_se3(poses[old["id"]]) @ S_cur_meas,
                              np.eye(dim) * 1e4)

        optimized = graph.optimize(iterations=20)     # lo pesado, fuera del lock
        with self._map_lock:
            T_old = poses[cur_id]
            self.mapper.update_poses_sim3(optimized)  # re-ancla y RE-ESCALA
            T_new = self.mapper.keyframe_pose(cur_id)
            # EL PUENTE DE COVISIBILIDAD: los pares verificados del bucle se
            # registran como observaciones del keyframe del bucle → los
            # keyframes ANTIGUOS de esta zona vuelven a ser covisibles, sus
            # puntos entran al mapa local, y el tracking re-usa la geometría
            # original en lugar de duplicarla (la causa del PnP biestable).
            bridge = [(pid, kp) for (pid, kp), ok in zip(pairs, inliers) if ok]
            self.mapper.add_observations(
                cur_id, [pid for pid, _ in bridge],
                np.float64([kps[kp].pt for _, kp in bridge]))
        if async_job:
            # El tracking ya avanzó: entregarle la corrección como DELTA en el
            # marco del mundo (T_nuevo·T_viejo⁻¹ del KF del bucle); la aplica
            # al inicio de su siguiente frame. El delta es rígido (las poses
            # se re-normalizan a SE(3)); la escala vive en el MAPA y el PnP la
            # absorbe — la convención de siempre (v0.4a).
            self._pending_pose_delta = T_new @ invert_se3(T_old)
        else:
            self.T_w_c = T_new
            self._T_prev = self.T_w_c.copy()
            self._kf["T"] = self.T_w_c.copy()
        if self._isam2 is not None:
            # La corrección Sim(3) reescribió poses y puntos FUERA de iSAM2:
            # su linealización quedó obsoleta → época nueva (ver gtsam_isam2).
            self._isam2.reset()
        for kp, pid in loop_by_kp.items():
            cur_mp.setdefault(kp, pid)

        self._last_loop_frame = self._frame_idx
        self.loop_events.append((self._frame_idx, old["id"]))
        info["state"] = "TRACK+KF+LOOP"

    # ── auxiliares ─────────────────────────────────────────────────────────────

    def _coast(self, gray, kps, desc, info) -> None:
        """Fallo (o rechazo) de tracking. Antes de rendirse a la velocidad
        constante, tras RELOC_AFTER frames perdidos intenta RELOCALIZAR contra
        toda la base de keyframes (v0.4b): el coast es el fallback si la reloc
        falla. Sin este contador, un solo frame malo dispararía una búsqueda
        global cara; con él, la reloc entra solo cuando de verdad estamos
        perdidos (oclusión persistente, secuestro)."""
        self._coast_count += 1
        if self._coast_count >= self.RELOC_AFTER \
                and self._relocalize(gray, kps, desc, info):
            return
        # Velocidad constante (como examples/01): extrapola el último movimiento.
        self.T_w_c = self.T_w_c @ self._T_rel
        self._T_prev = self.T_w_c.copy()
        if not info["state"].startswith("GATE"):
            info["state"] = "COAST"

    def _relocalize(self, gray, kps, desc, info) -> bool:
        """Recupera el tracking tras perderlo: reconoce el lugar contra TODA la
        base de keyframes y lo verifica con PnP — el mismo mecanismo del cierre
        de bucle SIN el filtro temporal (aquí parecerse a un keyframe reciente
        NO es un problema: no corregimos el mapa, solo re-medimos nuestra pose).

        ─── La matemática ───
        Relocalizar es un PnP GLOBAL sin prior de pose: se descarta TODO el
        estado de movimiento acumulado (la velocidad del coast ya no vale) y se
        re-mide la pose absoluta desde apariencia + geometría. Es lo que permite
        sobrevivir a un "secuestro" (la cámara teletransportada): un sistema que
        solo integra movimiento no puede; uno que localiza contra un mapa, sí.
        Candidatos: BoW top-K si hay vocabulario (v0.5 — el "a escala real es
        BoW" que esta nota prometía desde v0.4b, lección 34); si no, toda la
        base (fuerza bruta original). Se elige el candidato con MÁS inliers PnP
        (máximo soporte geométrico).
        """
        if self.use_bow and self._bow.trained:
            with self._map_lock:
                ranked = self._bow.query(desc, top_k=self.BOW_TOP_K)
            db_by_id = {e["id"]: e for e in self._kf_db}
            candidates = [db_by_id[k] for k, _ in ranked if k in db_by_id]
        else:
            candidates = list(self._kf_db)
        best = None
        for old in candidates:
            n_matches, pairs, T_pnp, inliers = self._match_against_kf(
                old, gray, kps, desc)
            if n_matches < self.RELOC_MIN_MATCHES or T_pnp is None:
                continue
            n_inl = int(inliers.sum())
            if n_inl < self.RELOC_MIN_INLIERS:
                continue
            if best is None or n_inl > best[1]:
                best = (T_pnp, n_inl, old["id"])
        if best is None:
            return False
        T_pnp, n_inl, old_id = best
        # Aceptar la pose global y BORRAR la velocidad acumulada del coast: tras
        # un secuestro, extrapolar el movimiento previo es justo lo incorrecto.
        self.T_w_c = T_pnp
        self._T_prev = T_pnp.copy()
        self._T_rel = np.eye(4)
        self._coast_count = 0
        self._local_ref_kf = old_id      # re-ancla el mapa local aquí (ver _local_kfs)
        self.reloc_events.append((self._frame_idx, old_id))
        info.update(tracked=True, n_inliers=n_inl, state="RELOC")
        return True

    def _project(self, points_w: np.ndarray) -> np.ndarray:
        T_c_w = invert_se3(self.T_w_c)
        pts_c = (T_c_w[:3, :3] @ points_w.T).T + T_c_w[:3, 3]
        return self.camera.project(pts_c)
