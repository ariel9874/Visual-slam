#!/usr/bin/env python3
"""
Ejemplo 01 — Odometría Visual Monocular (punto de entrada educativo)
====================================================================

Pipeline mínimo de odometría visual (VO) monocular, en escala de grises:

    imágenes ─▶ características ORB ─▶ matching (ratio test)
             ─▶ matriz esencial (RANSAC) ─▶ pose relativa (R, t)
             ─▶ composición de trayectoria (¡hasta escala!)

Todo el ciclo está implementado AQUÍ, en un solo archivo, para poder leerse de
arriba a abajo. Cada paso indica a qué módulo de `vslam/` corresponde en la
arquitectura real (docs/02_arquitectura.md):

    Paso 1  carga de imágenes      -> vslam/io/dataset.py        (se importa)
    Paso 2  extracción ORB         -> vslam/frontend/features.py (se importa)
    Paso 3  matching + ratio test  -> vslam/frontend/matching.py (se importa)
    Paso 4  geometría epipolar     -> vslam/frontend/tracker.py  (inline, didáctico)
    Paso 5  trayectoria            -> vslam/core/trajectory.py   (se importa)
    futuro  optimización           -> vslam/backend/factor_graph.py
    futuro  mapeo denso            -> vslam/mapping/

Limitaciones DELIBERADAS de esta versión (son la lección, no un descuido):

  * ESCALA: con una sola cámara, la traslación entre dos vistas solo se
    recupera en dirección, no en magnitud (ambigüedad de escala monocular).
    Aquí asumimos ||t|| = 1 en cada paso: la trayectoria tiene la FORMA
    correcta solo si la velocidad real es ~constante. Los sistemas reales
    resuelven la escala triangulando puntos 3D, o con estéreo/IMU.
  * DERIVA: cada pose se compone sobre la anterior; los errores se acumulan
    sin límite porque no hay backend (bundle adjustment) ni cierre de bucle.
  * 2D-2D SIEMPRE: re-estimamos la geometría desde cero en cada par de frames.
    Un sistema real triangula puntos y trackea 3D-2D (PnP), que es más estable.

Uso:
    python examples/01_monocular_vo.py --images data/synthetic/images \
        --calib data/synthetic/calib.txt --output output/synthetic [--show]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Permite ejecutar el ejemplo sin instalar el paquete. La forma "correcta" es
# `pip install -e .` desde la raíz del repo; esto es cortesía didáctica.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera          # Paso 1/4: intrínsecos K
from vslam.core.trajectory import Trajectory         # Paso 5: acumulación/exportación
from vslam.frontend.features import available_extractors, create_extractor  # Paso 2
from vslam.frontend.matching import available_matchers, create_matcher      # Paso 3
from vslam.io.dataset import ImageSequenceLoader      # Paso 1: secuencia


# ─────────────────────────── utilidades SE(3) ────────────────────────────────

def invert_se3(T: np.ndarray) -> np.ndarray:
    """Inversa cerrada de una transformación rígida 4x4.

    ─── La matemática: el grupo SE(3) ───
    Una pose es T = [[R, t], [0, 1]] con R ∈ SO(3) (RᵀR = I, det R = +1) y
    t ∈ ℝ³. Actúa sobre puntos como X' = R·X + t, y componer dos poses es
    multiplicar sus matrices (¡el orden importa: SE(3) no es conmutativo!).

    Para invertir, despeja X de X' = R·X + t:
        X = Rᵀ·X' − Rᵀ·t    ⇒    T⁻¹ = [[Rᵀ, −Rᵀ·t], [0, 1]]
    La forma cerrada es más barata que np.linalg.inv y garantiza que el
    resultado siga siendo exactamente rígido (Rᵀ es rotación perfecta;
    la inversa numérica genérica solo lo sería aproximadamente).

    Notación del repo: T_a_b lleva puntos del frame b al frame a. Los
    subíndices se encadenan "cancelándose", como unidades:
        T_w_c2 = T_w_c1 · T_c1_c2      (w←c1 por c1←c2 da w←c2)
    """
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# ──────────────────────── el estimador de odometría ──────────────────────────

class MonocularVO:
    """Odometría visual monocular 2D-2D mínima.

    Mantiene la pose acumulada T_w_c (convención del repo: transforma puntos
    de cámara a mundo; el primer frame define el origen del mundo).
    """

    # Umbrales didácticos: en un sistema real serían configuración.
    MIN_MATCHES = 30    # menos que esto -> el matching no es fiable
    MIN_INLIERS = 15    # menos que esto -> la geometría no es fiable
    RANSAC_PROB = 0.999
    RANSAC_THRESHOLD_PX = 1.0
    # Profundidad máxima aceptada en el test de quiralidad, en múltiplos del
    # baseline entre frames (ver la nota "TRAMPA CLÁSICA" en process_frame).
    CHEIRALITY_DIST_THRESH = 2000.0

    def __init__(self, camera: PinholeCamera, extractor=None, matcher=None) -> None:
        self.camera = camera
        # Detector y matcher son INTERCAMBIABLES (registros de vslam/frontend;
        # análisis de cada opción en docs/03_detectores_y_matchers.md). El
        # resto del pipeline no sabe cuál corre: esa es la tesis del repo.
        self.extractor = extractor or create_extractor("orb")
        self.matcher = matcher or create_matcher("ratio")

        self.T_w_c = np.eye(4)          # pose actual (mundo <- cámara)
        self.T_prev_rel = np.eye(4)     # último movimiento relativo (para el
                                        # modelo de velocidad constante)
        self._prev = None               # (keypoints, descriptores) del frame anterior

    def process_frame(self, gray: np.ndarray) -> tuple[np.ndarray, dict]:
        """Procesa un frame y devuelve (T_w_c actualizada, info de diagnóstico)."""
        info = {"n_kps": 0, "n_matches": 0, "n_inliers": 0, "tracked": False,
                "pts_prev": None, "pts_curr": None}

        # ── PASO 2 · Extracción de características (vslam/frontend/features) ──
        # ORB = esquinas FAST en pirámide de escalas + descriptor binario de
        # 256 bits. Reducimos la imagen (~300k píxeles) a ~2000 puntos
        # describibles y re-identificables desde otro punto de vista.
        keypoints, descriptors = self.extractor.detect_and_compute(gray)
        info["n_kps"] = len(keypoints)

        if self._prev is None:
            # Primer frame: fija el origen del mundo. No hay geometría aún.
            self._prev = (keypoints, descriptors)
            info["tracked"] = True
            return self.T_w_c, info

        prev_kps, prev_desc = self._prev

        # ── PASO 3 · Matching (vslam/frontend/matching) ──────────────────────
        # Buscamos el mismo punto físico visto en ambos frames. El matcher es
        # configurable (--matcher): ratio test de Lowe por defecto, que
        # descarta correspondencias ambiguas (texturas repetitivas): matches
        # limpios ahora = RANSAC más barato y fiable después. Se pasan también
        # los keypoints y el tamaño de imagen porque los matchers APRENDIDOS
        # (LightGlue) razonan sobre posiciones, no solo sobre descriptores.
        matches = self.matcher.match(prev_desc, descriptors,
                                     kps_a=prev_kps, kps_b=keypoints,
                                     image_shape=gray.shape)
        info["n_matches"] = len(matches)

        if len(matches) < self.MIN_MATCHES:
            return self._coast(keypoints, descriptors, info)

        # Coordenadas de píxel de cada correspondencia (prev -> curr).
        pts_prev = np.float64([prev_kps[m.queryIdx].pt for m in matches])
        pts_curr = np.float64([keypoints[m.trainIdx].pt for m in matches])

        # ── PASO 4 · Geometría epipolar (futuro vslam/frontend/tracker) ──────
        #
        # ─── La matemática: la restricción epipolar ───
        # Sea X un punto 3D visto por ambas cámaras, relacionadas por la pose
        # relativa (R, t):  X_curr = R·X_prev + t. Trabajamos con RAYOS en
        # coordenadas normalizadas x̂ = K⁻¹·[u, v, 1]ᵀ (los píxeles fuera).
        #
        # Los vectores x̂_curr, R·x̂_prev y t son COPLANARES: los dos rayos y el
        # baseline forman el "plano epipolar" que contiene a X y a ambos
        # centros ópticos. Coplanaridad = triple producto escalar nulo:
        #
        #     x̂_curr · (t × R·x̂_prev) = 0
        #  ⇔  x̂_currᵀ · [t]_× · R · x̂_prev = 0 ,       E ≜ [t]_× · R
        #
        # donde [t]_× es la matriz antisimétrica que implementa "t × ·":
        #     [t]_× = [[  0, -t3,  t2],
        #              [ t3,   0, -t1],
        #              [-t2,  t1,   0]]
        #
        # E (matriz ESENCIAL) empaqueta la rotación y la DIRECCIÓN de t en una
        # sola 3x3. Tiene 5 grados de libertad (3 de R + 2 de dirección de t:
        # la ecuación es homogénea, la escala no cuenta) → bastan 5
        # correspondencias: el solver de 5 puntos de Nistér, que además tolera
        # escenas planas donde el clásico de 8 puntos degenera.
        #
        # ─── La matemática: RANSAC ───
        # El ratio test limpió mucho, pero UN solo outlier arruina un ajuste
        # por mínimos cuadrados. RANSAC itera: muestrear 5 matches al azar →
        # resolver E → contar cuántos matches la satisfacen (distancia
        # epipolar < threshold px) → quedarse con el E de mayor consenso y
        # marcar sus inliers. Si w es la fracción de inliers, una muestra de
        # 5 sale toda-inlier con probabilidad w⁵, así que para acertar con
        # probabilidad p bastan
        #     N = log(1 − p) / log(1 − w⁵)   iteraciones
        # (con w = 0.5 y p = 0.999, N ≈ 218: por eso corre en tiempo real).
        E, inlier_mask = cv2.findEssentialMat(
            pts_prev, pts_curr, self.camera.K,
            method=cv2.RANSAC, prob=self.RANSAC_PROB,
            threshold=self.RANSAC_THRESHOLD_PX,
        )
        if E is None or E.shape != (3, 3):
            return self._coast(keypoints, descriptors, info)

        # ─── La matemática: de E a (R, t) ───
        # Con la SVD  E = U·diag(1, 1, 0)·Vᵀ  existen CUATRO factorizaciones
        # E = [t]_×·R:
        #     R ∈ { U·W·Vᵀ ,  U·Wᵀ·Vᵀ }     con  W = [[0,-1,0],[1,0,0],[0,0,1]]
        #     t ∈ { +u3 , -u3 }             (u3 = 3.ª columna de U, ||t|| = 1)
        # — el "twisted pair" (dos rotaciones) por el signo del baseline. Solo
        # UNA combinación deja los puntos triangulados con profundidad positiva
        # en ambas cámaras: ese es el test de quiralidad que recoverPose hace
        # por nosotros. Devuelve T_curr<-prev (puntos del frame anterior al
        # actual).
        #
        # ¿Por qué ||t|| = 1? Porque E = [t]_×·R es homogénea en t: si
        # (R, t, {X_i}) explica las imágenes, (R, s·t, {s·X_i}) las explica
        # EXACTAMENTE igual para todo s > 0. Una cámara monocular no puede
        # medir la escala del mundo; se fija ||t|| = 1 por convención. (En
        # v0.2 la escala se heredará del mapa triangulado; estéreo o IMU la
        # harían métrica de verdad.)
        #
        # TRAMPA CLÁSICA de OpenCV (descubierta verificando este repo): la
        # sobrecarga básica de recoverPose solo acepta como inliers los puntos
        # triangulados a menos de 50 unidades — y como ||t||=1, esas unidades
        # son MÚLTIPLOS DEL BASELINE. Si la cámara se mueve poco entre frames
        # respecto a la profundidad de la escena (depth/baseline > 50: lo
        # normal en KITTI, drones, o cualquier avance suave), TODOS los puntos
        # parecen estar "en el infinito" y los inliers caen a ~0 aunque la
        # geometría sea perfecta. La sobrecarga con distanceThresh permite
        # subir ese umbral (y además devuelve los puntos triangulados, que en
        # v0.2 usaremos para inicializar el mapa disperso).
        n_inliers, R, t, pose_mask, _triangulated = cv2.recoverPose(
            E, pts_prev, pts_curr, self.camera.K,
            distanceThresh=self.CHEIRALITY_DIST_THRESH, mask=inlier_mask,
        )
        info["n_inliers"] = int(n_inliers)

        if n_inliers < self.MIN_INLIERS:
            return self._coast(keypoints, descriptors, info)

        # CASO DEGENERADO que debes conocer: si la cámara solo ROTA, el
        # baseline t → 0 y con él E = [t]_×·R → 0: la restricción epipolar se
        # satisface trivialmente y la dirección de t que devuelve el solver es
        # puro ruido. (Matemáticamente: con t = 0 los rayos se relacionan por
        # la homografía de rotación x̂_curr ~ R·x̂_prev, sin paralaje no hay
        # traslación observable.) Los sistemas serios lo detectan eligiendo
        # por consenso entre modelo de homografía y esencial — así inicializa
        # ORB-SLAM. Aquí lo documentamos y seguimos: es un ejemplo educativo.

        # ── PASO 5 · Composición de la trayectoria (vslam/core/trajectory) ───
        # recoverPose nos dio T_curr<-prev; para acumular necesitamos el
        # movimiento de la CÁMARA en el mundo: T_w<-curr = T_w<-prev · T_prev<-curr
        # donde T_prev<-curr = (T_curr<-prev)^-1.
        T_curr_prev = np.eye(4)
        T_curr_prev[:3, :3] = R
        T_curr_prev[:3, 3] = t.ravel()  # ||t|| = 1: escala arbitraria (ver docstring)

        T_rel = invert_se3(T_curr_prev)          # prev <- curr
        self.T_w_c = self.T_w_c @ T_rel          # componer sobre la pose anterior
        self.T_prev_rel = T_rel                  # recordar para el modelo de coasting

        # Diagnóstico para visualización (solo inliers).
        ok = pose_mask.ravel().astype(bool)
        info.update(tracked=True, pts_prev=pts_prev[ok], pts_curr=pts_curr[ok])

        self._prev = (keypoints, descriptors)
        return self.T_w_c, info

    def _coast(self, keypoints, descriptors, info) -> tuple[np.ndarray, dict]:
        """Fallo de tracking: aplica el último movimiento conocido.

        Es el "modelo de velocidad constante" clásico, la mitigación más
        simple posible. Un sistema real intentaría relocalizar contra el mapa
        (bolsa de palabras) antes de rendirse.
        """
        self.T_w_c = self.T_w_c @ self.T_prev_rel
        self._prev = (keypoints, descriptors)
        return self.T_w_c, info


# ───────────────────────────── visualización ─────────────────────────────────

def draw_tracks(gray: np.ndarray, info: dict) -> np.ndarray:
    """Dibuja los inliers como vectores de movimiento (verde: prev -> curr)."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if info["pts_prev"] is not None:
        for (x0, y0), (x1, y1) in zip(info["pts_prev"], info["pts_curr"]):
            cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 0), 1)
            cv2.circle(vis, (int(x1), int(y1)), 2, (0, 255, 0), -1)
    cv2.putText(vis, f"kps {info['n_kps']}  matches {info['n_matches']}  inliers {info['n_inliers']}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return vis


def save_trajectory_plot(trajectory: Trajectory, path: Path) -> None:
    """Vista cenital (X-Z) de la trayectoria. matplotlib es opcional."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # sin ventana: solo guardar PNG
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aviso] matplotlib no instalado: se omite trajectory.png "
              "(pip install -e '.[viz]')")
        return
    p = trajectory.positions
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(p[:, 0], p[:, 2], "-", lw=1.5, label="estimada (escala arbitraria)")
    ax.plot(p[0, 0], p[0, 2], "go", label="inicio")
    ax.plot(p[-1, 0], p[-1, 2], "rs", label="fin")
    ax.set_xlabel("x [u.a.]"), ax.set_ylabel("z [u.a.]")
    ax.set_title("Odometría visual monocular — vista cenital")
    ax.axis("equal"), ax.grid(True, alpha=0.3), ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────── programa ────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--images", required=True, help="carpeta con la secuencia de imágenes")
    parser.add_argument("--calib", required=True, help=".txt de calibración: 'fx fy cx cy'")
    parser.add_argument("--output", default="output/run", help="carpeta de resultados")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = todos")
    parser.add_argument("--show", action="store_true", help="ventana con los tracks en vivo")
    parser.add_argument("--detector", default="orb", choices=available_extractors(),
                        help="detector/descriptor de características (ver docs/03)")
    parser.add_argument("--matcher", default="ratio", choices=available_matchers(),
                        help="estrategia de emparejamiento (ver docs/03)")
    args = parser.parse_args()

    # ── PASO 1 · Carga (vslam/io/dataset) ─────────────────────────────────────
    camera = PinholeCamera.from_file(args.calib)
    loader = ImageSequenceLoader(args.images)
    print(f"Secuencia: {len(loader)} imágenes | K: fx={camera.fx} fy={camera.fy} "
          f"cx={camera.cx} cy={camera.cy}")
    print(f"Frontend: detector={args.detector} matcher={args.matcher}")

    vo = MonocularVO(camera,
                     extractor=create_extractor(args.detector),
                     matcher=create_matcher(args.matcher))
    trajectory = Trajectory()

    for i, (timestamp, gray) in enumerate(loader):
        if args.max_frames and i >= args.max_frames:
            break

        T_w_c, info = vo.process_frame(gray)
        trajectory.append(timestamp, T_w_c)

        if i % 20 == 0 or not info["tracked"]:
            x, y, z = T_w_c[:3, 3]
            estado = "ok" if info["tracked"] else "COASTING (tracking débil)"
            print(f"frame {i:5d} | inliers {info['n_inliers']:4d} | "
                  f"pos [{x:+7.2f} {y:+7.2f} {z:+7.2f}] | {estado}")

        if args.show:
            cv2.imshow("vslam - ejemplo 01 (ESC para salir)", draw_tracks(gray, info))
            if cv2.waitKey(1) == 27:
                break

    if args.show:
        cv2.destroyAllWindows()

    # ── Resultados ────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    trajectory.save_tum(out / "trajectory.txt")
    save_trajectory_plot(trajectory, out / "trajectory.png")
    print(f"\nTrayectoria: {len(trajectory)} poses -> {out / 'trajectory.txt'} (formato TUM)")
    print("Evalúa contra ground truth con `evo`:  "
          f"evo_ape tum <groundtruth.txt> {out / 'trajectory.txt'} -as")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
