"""vslam — Visual SLAM educativo y modular.

Capas de la arquitectura (ver docs/02_arquitectura.md):
  - vslam.core      contratos de datos (Frame, PinholeCamera, Trajectory)
  - vslam.frontend  tracking rápido por frame
  - vslam.backend   optimización sobre grafos de factores
  - vslam.mapping   mapeo denso intercambiable
  - vslam.io        carga de datasets y calibración

─── API PÚBLICA (congelada en v0.9 para 1.0) ─────────────────────────────────
Lo que está en __all__ es el contrato con el usuario: nombres y firmas
estables bajo versionado semántico. Todo lo demás (métodos _privados, los
módulos de ejemplo, los umbrales internos) puede cambiar entre minors. Las
importaciones PESADAS (torch, gtsam) son perezosas: importar `vslam` no
arrastra GPU ni C++ — los mappers densos y los backends GTSAM se importan
explícitamente desde sus módulos.

    from vslam import PinholeCamera, PnPTracker, load_config
    from vslam.mapping.gaussian import GaussianSplattingMapper   # (torch)
──────────────────────────────────────────────────────────────────────────────
"""

__version__ = "0.9.0"

from vslam.config import apply_config, load_config
from vslam.core.camera import PinholeCamera
from vslam.core.frame import Frame
from vslam.core.trajectory import Trajectory
from vslam.frontend.features import available_extractors, create_extractor
from vslam.frontend.matching import create_matcher
from vslam.frontend.tracker import PnPTracker
from vslam.io.dataset import (EuRoCStereoLoader, EuRoCStereoRig,
                              TUMRGBDLoader, tum_camera)
from vslam.mapping.base import MapperBase
from vslam.mapping.sparse import SparsePointMapper

__all__ = [
    # contratos de datos
    "PinholeCamera", "Frame", "Trajectory",
    # frontend (el sistema completo vive detrás del tracker)
    "PnPTracker", "create_extractor", "create_matcher", "available_extractors",
    # mapping (los densos —torch— se importan de vslam.mapping.gaussian/…)
    "MapperBase", "SparsePointMapper",
    # datasets
    "TUMRGBDLoader", "EuRoCStereoLoader", "EuRoCStereoRig", "tum_camera",
    # configuración declarativa (v0.9)
    "load_config", "apply_config",
    "__version__",
]
