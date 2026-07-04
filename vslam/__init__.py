"""vslam — Visual SLAM educativo y modular.

Capas de la arquitectura (ver docs/02_arquitectura.md):
  - vslam.core      contratos de datos (Frame, PinholeCamera, Trajectory)
  - vslam.frontend  tracking rápido por frame
  - vslam.backend   optimización sobre grafos de factores
  - vslam.mapping   mapeo denso intercambiable
  - vslam.io        carga de datasets y calibración
"""

__version__ = "0.1.0"

from vslam.core.camera import PinholeCamera
from vslam.core.frame import Frame
from vslam.core.trajectory import Trajectory

__all__ = ["PinholeCamera", "Frame", "Trajectory", "__version__"]
