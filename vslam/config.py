"""Configuración declarativa (v0.9): las perillas del sistema, fuera del código.

─── La filosofía ──────────────────────────────────────────────────────────────
Las CONSTANTES DE CLASE son la documentación: cada umbral vive junto a su
porqué medido (KF_MIN_INLIERS=100 con su comentario, lección tal). Este módulo
NO las sustituye — las SOBREESCRIBE por instancia en el despliegue:

    tracker = PnPTracker(cam, ..., config=load_config("mi_robot.yaml"))

Reglas:
  · Config vacía / ausente ⇒ comportamiento BIT-IDÉNTICO a la referencia (el
    test de equivalencia de la lección 36: si el run cambia sin config, algo
    se rompió).
  · Solo se aceptan claves que EXISTEN como constante de la clase — un typo
    (KF_MIN_INLIER) falla en el arranque con la lista de claves válidas, no
    silenciosamente a mitad de secuencia.
  · La plantilla se GENERA desde las clases (`python -m vslam.config`): una
    sola fuente de verdad, sin YAML que envejece por su cuenta.

Formato: YAML si hay PyYAML (el contenedor ROS lo trae), JSON siempre (stdlib).
Secciones = módulos: {"tracker": {...}, "isam2": {...}, "pose_graph": {...}}.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Type


def _known_knobs(cls: Type) -> Dict[str, Any]:
    """Las perillas de una clase: atributos de clase EN MAYÚSCULAS con valor
    escalar (int/float/bool/str). Lo demás (métodos, privados) no es config."""
    out = {}
    for name in dir(cls):
        if name.isupper() and not name.startswith("_"):
            v = getattr(cls, name)
            if isinstance(v, (int, float, bool, str)):
                out[name] = v
    return out


def apply_config(obj: Any, overrides: Optional[Dict[str, Any]],
                 section: str = "") -> None:
    """Sobreescribe las perillas de `obj` (por INSTANCIA: la clase no se toca).
    Claves en minúsculas o mayúsculas; tipo forzado al del default (un `5`
    sobre un float queda 5.0 — sin sorpresas de comparación)."""
    if not overrides:
        return
    known = _known_knobs(type(obj))
    for key, value in overrides.items():
        name = key.upper()
        if name not in known:
            raise KeyError(
                f"config[{section or type(obj).__name__}]: '{key}' no es una "
                f"perilla de {type(obj).__name__}. Válidas: {sorted(known)}")
        default = known[name]
        if isinstance(default, bool):
            coerced = bool(value)
        elif isinstance(default, int) and not isinstance(value, bool):
            coerced = int(value)
        elif isinstance(default, float):
            coerced = float(value)
        else:
            coerced = value
        setattr(obj, name, coerced)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Lee YAML (si hay PyYAML) o JSON. Devuelve el dict de secciones."""
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith((".yaml", ".yml")):
        import yaml                          # el contenedor ROS lo trae
        return yaml.safe_load(text) or {}
    return json.loads(text)


def dump_template() -> str:
    """Genera la plantilla comentada DESDE las clases (fuente única de verdad).
    Solo importa lo que esté disponible (gtsam es opcional)."""
    # ASCII a proposito: esto se IMPRIME (regla Windows/cp1252 de docs/05 s2).
    lines = ["# Configuracion Visual-SLAM - generada por `python -m vslam.config`",
             "# Cada clave sobreescribe la constante homonima de su clase;",
             "# los porques (y las lecciones) viven en el codigo junto a ellas.",
             ""]
    from vslam.frontend.tracker import PnPTracker
    sections = [("tracker", PnPTracker)]
    try:
        from vslam.backend.gtsam_isam2 import ISAM2LocalBA
        sections.append(("isam2", ISAM2LocalBA))
    except ImportError:
        pass
    try:
        from vslam.backend.pose_graph import GaussNewtonPoseGraph
        sections.append(("pose_graph", GaussNewtonPoseGraph))
    except ImportError:
        pass
    for name, cls in sections:
        lines.append(f"{name}:")
        for key, val in sorted(_known_knobs(cls).items()):
            lines.append(f"  {key.lower()}: {val}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(dump_template())
