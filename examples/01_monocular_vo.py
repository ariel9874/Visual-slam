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
from vslam.frontend.features import FeatureExtractor  # Paso 2: ORB
from vslam.frontend.matching import match_descriptors  # Paso 3: ratio test
from vslam.io.dataset import ImageSequenceLoader      # Paso 1: secuencia


# ─────────────────────────── utilidades SE(3) ────────────────────────────────

def invert_se3(T: np.ndarray) -> np.ndarray:
    """Inversa cerrada de una transformación rígida 4x4.

    Si T = [R | t; 0 | 1], entonces T^-1 = [R^T | -R^T t; 0 | 1].
    (Mucho más barato y estable que np.linalg.inv para SE(3).)
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

    def __init__(self, camera: PinholeCamera, n_features: int = 2000) -> None:
        self.camera = camera
        self.extractor = FeatureExtractor(n_features=n_features)

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

        # ── PASO 3 · Matching con ratio test (vslam/frontend/matching) ───────
        # Buscamos el mismo punto físico visto en ambos frames. El ratio test
        # de Lowe descarta correspondencias ambiguas (texturas repetitivas):
        # matches limpios ahora = RANSAC más barato y fiable después.
        matches = match_descriptors(prev_desc, descriptors, ratio=0.75)
        info["n_matches"] = len(matches)

        if len(matches) < self.MIN_MATCHES:
            return self._coast(keypoints, descriptors, info)

        # Coordenadas de píxel de cada correspondencia (prev -> curr).
        pts_prev = np.float64([prev_kps[m.queryIdx].pt for m in matches])
        pts_curr = np.float64([keypoints[m.trainIdx].pt for m in matches])

        # ── PASO 4 · Geometría epipolar (futuro vslam/frontend/tracker) ──────
        # La matriz esencial E codifica la rotación y traslación entre dos
        # vistas calibradas: x_curr^T · E · x_prev = 0 para correspondencias
        # correctas (en coordenadas normalizadas por K).
        #
        # RANSAC es imprescindible: aunque el ratio test limpió mucho, siguen
        # quedando matches erróneos (outliers) y la estimación por mínimos
        # cuadrados se arruinaría con uno solo. RANSAC busca el modelo E con
        # más consenso y marca los inliers. (El solver de 5 puntos de Nistér
        # funciona incluso con escenas planas, donde el clásico de 8 puntos
        # para F degenera.)
        E, inlier_mask = cv2.findEssentialMat(
            pts_prev, pts_curr, self.camera.K,
            method=cv2.RANSAC, prob=self.RANSAC_PROB,
            threshold=self.RANSAC_THRESHOLD_PX,
        )
        if E is None or E.shape != (3, 3):
            return self._coast(keypoints, descriptors, info)

        # Descomponer E da 4 combinaciones (R, t) posibles; recoverPose elige
        # la única que deja los puntos triangulados DELANTE de ambas cámaras
        # (test de quiralidad). Devuelve la transformación T_curr<-prev:
        # lleva puntos del frame de la cámara anterior al de la actual,
        # con ||t|| = 1 (aquí está la ambigüedad de escala monocular).
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

        # CASO DEGENERADO que debes conocer: si la cámara solo ROTA (baseline
        # ~ 0), E no contiene información de traslación y t sale como ruido.
        # Los sistemas serios lo detectan (p. ej. eligiendo entre homografía y
        # esencial, como la inicialización de ORB-SLAM). Aquí lo documentamos
        # y seguimos: es un ejemplo educativo.

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
    args = parser.parse_args()

    # ── PASO 1 · Carga (vslam/io/dataset) ─────────────────────────────────────
    camera = PinholeCamera.from_file(args.calib)
    loader = ImageSequenceLoader(args.images)
    print(f"Secuencia: {len(loader)} imágenes | K: fx={camera.fx} fy={camera.fy} "
          f"cx={camera.cx} cy={camera.cy}")

    vo = MonocularVO(camera)
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
