"""Reconocimiento de lugar por BOLSA DE PALABRAS VISUALES (BoW) — v0.5.

El cierre de bucle y la relocalización necesitan responder "¿a qué keyframe
antiguo se parece este frame?". Hacerlo por fuerza bruta (knnMatch contra CADA
keyframe de la base) es O(KFs) con constante alta — el perfil lo midió como el
cuello del keyframe (~200 ms/KF, lección 32). BoW lo vuelve sub-lineal: cada
imagen se resume en un histograma disperso de "palabras visuales" y un índice
invertido devuelve los candidatos que comparten vocabulario en milisegundos.
Solo los top-K candidatos pasan a la verificación geométrica completa.

─── La matemática ─────────────────────────────────────────────────────────────
1. VOCABULARIO (k-medias en el espacio del descriptor). Para descriptores
   BINARIOS (ORB) la media aritmética no existe: el centroide que minimiza la
   suma de distancias de Hamming coordenada a coordenada es la MEDIANA por bit
   — el VOTO DE MAYORÍA (bit j del centroide = 1 si más de la mitad de sus
   miembros lo tienen). Para float (SuperPoint) es la media de siempre. La
   asignación (¿qué palabra le toca a cada descriptor?) es un vecino más
   cercano descriptor→centroide, que delegamos en cv2.BFMatcher (C++).

2. TF-IDF (la ponderación clásica de recuperación de texto, Sivic & Zisserman
   "Video Google", 2003). El histograma crudo sobre-pondera palabras que salen
   en TODAS partes (textura genérica). Se pondera cada palabra w por:

       tf(w, imagen) = n_w / n_total          (frecuencia en la imagen)
       idf(w)        = log(N / df_w)          (rareza en el corpus: df_w =
                                               nº de keyframes que la contienen)

   y la similitud entre dos imágenes es el COSENO entre sus vectores tf·idf
   normalizados — 1.0 = mismo reparto de palabras, 0.0 = disjuntas.

3. ÍNDICE INVERTIDO: palabra → keyframes que la contienen. La consulta solo
   toca los keyframes que comparten ALGUNA palabra con el query (en la
   práctica, el coseno se acumula palabra a palabra sobre ese subconjunto).
──────────────────────────────────────────────────────────────────────────────

Nota de alcance: DBoW2/3 (lo que usa ORB-SLAM) añade un árbol jerárquico de
vocabulario (k^L palabras con asignación O(k·L)) y vocabularios pre-entrenados
en millones de imágenes. Aquí el vocabulario se entrena EN SESIÓN (con los
descriptores de los primeros keyframes) — suficiente para bases de decenas o
cientos de KFs, y autocontenido (sin pesos externos que descargar).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class BagOfVisualWords:
    """Vocabulario + índice invertido + consulta TF-IDF. dtype-agnóstico:
    uint8 → Hamming/voto de mayoría; float32 → L2/media."""

    def __init__(self, n_words: int = 512, kmeans_iters: int = 6,
                 seed: int = 0) -> None:
        self.n_words = n_words
        self.kmeans_iters = kmeans_iters
        self._rng = np.random.default_rng(seed)
        self._vocab: Optional[np.ndarray] = None      # (n_words, D)
        self._matcher: Optional[cv2.BFMatcher] = None
        self._tf: Dict[int, Dict[int, float]] = {}    # kf_id -> {palabra: tf}
        self._df: Dict[int, int] = {}                 # palabra -> nº de KFs
        self._inverted: Dict[int, List[int]] = {}     # palabra -> [kf_id]

    # ── vocabulario ───────────────────────────────────────────────────────────

    @property
    def trained(self) -> bool:
        return self._vocab is not None

    def fit(self, descriptors: np.ndarray) -> None:
        """Entrena el vocabulario con k-medias sobre una muestra de descriptores
        (los de los primeros keyframes de la sesión). Coste único de ~decenas de
        ms: la asignación va por BFMatcher (C++) y la actualización es NumPy."""
        desc = np.asarray(descriptors)
        n = len(desc)
        k = min(self.n_words, max(2, n // 4))
        centroids = desc[self._rng.choice(n, size=k, replace=False)].copy()
        norm = cv2.NORM_HAMMING if desc.dtype == np.uint8 else cv2.NORM_L2
        bf = cv2.BFMatcher(norm)
        for _ in range(self.kmeans_iters):
            assign = np.array([m.trainIdx for m in bf.match(desc, centroids)])
            for j in range(k):
                members = desc[assign == j]
                if not len(members):
                    # cluster vacío: re-sembrar con un descriptor al azar
                    centroids[j] = desc[self._rng.integers(0, n)]
                elif desc.dtype == np.uint8:
                    # ─── centroide de Hamming: VOTO DE MAYORÍA por bit ───
                    bits = np.unpackbits(members, axis=1)
                    maj = (bits.mean(axis=0) >= 0.5).astype(np.uint8)
                    centroids[j] = np.packbits(maj)
                else:
                    centroids[j] = members.mean(axis=0)
        self._vocab = centroids
        self._matcher = bf

    def _quantize(self, desc: np.ndarray) -> Dict[int, float]:
        """Descriptores → histograma tf disperso {palabra: frecuencia}."""
        words = [m.trainIdx for m in self._matcher.match(desc, self._vocab)]
        tf: Dict[int, float] = {}
        for w in words:
            tf[w] = tf.get(w, 0.0) + 1.0
        inv = 1.0 / max(len(words), 1)
        return {w: c * inv for w, c in tf.items()}

    # ── índice ────────────────────────────────────────────────────────────────

    def add(self, kf_id: int, desc: np.ndarray) -> None:
        """Indexa un keyframe (llamar tras entrenar el vocabulario)."""
        tf = self._quantize(desc)
        self._tf[kf_id] = tf
        for w in tf:
            self._df[w] = self._df.get(w, 0) + 1
            self._inverted.setdefault(w, []).append(kf_id)

    def query(self, desc: np.ndarray, top_k: int = 5
              ) -> List[Tuple[int, float]]:
        """Los top_k keyframes más parecidos por coseno tf·idf. Solo visita a
        los KFs que comparten alguna palabra con el query (índice invertido)."""
        if not self.trained or not self._tf:
            return []
        import math
        q_tf = self._quantize(desc)
        n_kf = len(self._tf)
        # Memo de logaritmos: los df son enteros pequeños muy repetidos; evita
        # miles de llamadas a log por consulta (medido como relevante).
        logs: Dict[int, float] = {}

        def _idf(w: int) -> float:
            df = self._df.get(w, 0)
            if df <= 0:
                return 0.0
            if df not in logs:
                logs[df] = math.log(df)
            return math.log(n_kf) - logs[df]

        q = {w: tf * _idf(w) for w, tf in q_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q.values())) or 1.0

        # Producto punto acumulado SOLO sobre candidatos del índice invertido.
        dots: Dict[int, float] = {}
        for w, qv in q.items():
            if qv == 0.0:
                continue
            iw = _idf(w)
            for kf in self._inverted.get(w, ()):
                dots[kf] = dots.get(kf, 0.0) + qv * self._tf[kf].get(w, 0.0) * iw
        scores = []
        for kf, dot in dots.items():
            # Norma del documento con el idf COMPLETO de sus palabras (no solo
            # las del query) — si no, el coseno queda mal normalizado.
            d_norm = math.sqrt(sum((tf * _idf(w)) ** 2
                                   for w, tf in self._tf[kf].items())) or 1.0
            scores.append((kf, dot / (q_norm * d_norm)))
        scores.sort(key=lambda s: -s[1])
        return scores[:top_k]
