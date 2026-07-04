"""Adaptadores para extractores/matchers APRENDIDOS (SuperPoint, DISK, LightGlue).

Requieren el extra opcional:
    pip install -e ".[deep]"                                  # torch + kornia
    pip install git+https://github.com/cvg/LightGlue.git      # superpoint + lightglue

Estado: EXPERIMENTAL. Los adaptadores están escritos contra la API pública de
`lightglue` (cvg) y `kornia`, pero este repo aún no tiene CI con GPU: lo único
verificado en todas las máquinas es que, sin las dependencias, fallan con un
mensaje de instalación claro (nunca rompen la instalación base). Si los usas
y algo no cuadra, abre un issue con la versión de torch/kornia/lightglue.

─── La matemática (idea central de cada uno; análisis completo en docs/03) ───
SuperPoint: CNN con encoder compartido y dos cabezas — detector (clasifica
cada celda 8×8 en 65 clases: 64 posiciones + "sin punto") y descriptor 256D.
Entrenamiento auto-supervisado: esquinas sintéticas (MagicPoint) + Homographic
Adaptation (agregar detecciones bajo homografías aleatorias de la misma
imagen crea el ground truth de "esquinidad" sin etiquetar nada a mano).

DISK: entrenado con gradiente de política (REINFORCE): la recompensa son los
matches CORRECTOS tras emparejar → optimiza el objetivo final del pipeline y
no un proxy de repetibilidad. Denso y generoso en matches.

LightGlue: matcher de grafos con auto/cross-atención sobre (posición +
descriptor) y asignación por transporte óptimo. Adaptativo: capas con
early-exit según confianza y poda de puntos no emparejables. Es la razón de
que MatcherBase.match() reciba keypoints e image_shape: los matchers
aprendidos razonan sobre GEOMETRÍA, no solo sobre descriptores.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vslam.frontend.features import FeatureExtractorBase
from vslam.frontend.matching import MatcherBase

_INSTALL_MSG = (
    "El frontend aprendido '{name}' requiere dependencias opcionales.\n"
    "Instala:  pip install -e \".[deep]\"  (torch + kornia)\n"
    "y para superpoint/lightglue:  pip install git+https://github.com/cvg/LightGlue.git"
)


def _require(module: str, name: str):
    try:
        return __import__(module)
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG.format(name=name)) from exc


def _to_tensor(gray: np.ndarray, torch, device):
    """(H, W) uint8 → tensor (1, 1, H, W) float en [0, 1]."""
    return torch.from_numpy(gray).float().div(255.0)[None, None].to(device)


class SuperPointExtractor(FeatureExtractorBase):
    """SuperPoint vía el paquete `lightglue` (cvg). Pesos: solo investigación."""

    name = "superpoint"
    descriptor_type = "float"

    def __init__(self, n_features: int = 2048, device: Optional[str] = None) -> None:
        torch = _require("torch", self.name)
        lightglue = _require("lightglue", self.name)
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = lightglue.SuperPoint(max_num_keypoints=n_features).eval().to(self.device)

    def detect_and_compute(self, gray: np.ndarray) -> Tuple[Sequence, np.ndarray]:
        with self._torch.no_grad():
            feats = self._model.extract(_to_tensor(gray, self._torch, self.device))
        pts = feats["keypoints"][0].cpu().numpy()
        desc = feats["descriptors"][0].cpu().numpy().astype(np.float32)
        kps = [cv2.KeyPoint(float(x), float(y), 8.0) for x, y in pts]
        return kps, desc


class DISKExtractor(FeatureExtractorBase):
    """DISK vía kornia (pesos permisivos)."""

    name = "disk"
    descriptor_type = "float"

    def __init__(self, n_features: int = 2048, device: Optional[str] = None) -> None:
        torch = _require("torch", self.name)
        kornia = _require("kornia", self.name)
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_features = n_features
        self._model = kornia.feature.DISK.from_pretrained("depth").eval().to(self.device)

    def detect_and_compute(self, gray: np.ndarray) -> Tuple[Sequence, np.ndarray]:
        # DISK espera 3 canales; replicamos el gris.
        t = _to_tensor(gray, self._torch, self.device).repeat(1, 3, 1, 1)
        with self._torch.no_grad():
            out = self._model(t, n=self.n_features)[0]
        pts = out.keypoints.cpu().numpy()
        desc = out.descriptors.cpu().numpy().astype(np.float32)
        kps = [cv2.KeyPoint(float(x), float(y), 8.0) for x, y in pts]
        return kps, desc


class LightGlueMatcher(MatcherBase):
    """LightGlue (Apache-2.0) vía el paquete `lightglue` (cvg).

    A diferencia de los matchers clásicos, NECESITA las posiciones de los
    keypoints (kps_a/kps_b) y el tamaño de imagen: su atención es espacial.
    `features` debe coincidir con el extractor usado ("superpoint" o "disk").
    """

    name = "lightglue"

    def __init__(self, features: str = "superpoint", device: Optional[str] = None) -> None:
        torch = _require("torch", self.name)
        lightglue = _require("lightglue", self.name)
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = lightglue.LightGlue(features=features).eval().to(self.device)

    def match(self, desc_a, desc_b, kps_a=None, kps_b=None, image_shape=None) -> List[cv2.DMatch]:
        if kps_a is None or kps_b is None or image_shape is None:
            raise ValueError("LightGlueMatcher necesita kps_a, kps_b e image_shape "
                             "(los matchers aprendidos razonan sobre posiciones).")
        torch = self._torch
        h, w = image_shape

        def pack(kps, desc):
            pts = torch.tensor([[kp.pt[0], kp.pt[1]] for kp in kps],
                               dtype=torch.float32, device=self.device)[None]
            return {
                "keypoints": pts,
                "descriptors": torch.from_numpy(np.ascontiguousarray(desc)).float()[None].to(self.device),
                "image_size": torch.tensor([[w, h]], dtype=torch.float32, device=self.device),
            }

        with torch.no_grad():
            out = self._model({"image0": pack(kps_a, desc_a), "image1": pack(kps_b, desc_b)})
        pairs = out["matches"][0].cpu().numpy()          # (M, 2): índices en A y B
        scores = out["scores"][0].cpu().numpy() if "scores" in out else np.ones(len(pairs))
        return [cv2.DMatch(int(i), int(j), float(1.0 - s))
                for (i, j), s in zip(pairs, scores)]
