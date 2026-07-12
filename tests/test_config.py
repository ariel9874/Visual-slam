#!/usr/bin/env python3
"""Tests de la configuración declarativa (v0.9 hito 1).

El contrato: (1) SIN config, el tracker usa exactamente las constantes de
clase (bit-idéntico a la referencia — la garantía de no-regresión); (2) la
config sobreescribe POR INSTANCIA (la clase no se toca: otro tracker en el
mismo proceso no se contamina); (3) un typo falla EN EL ARRANQUE con la lista
de claves válidas; (4) la plantilla generada cubre todas las perillas y
recarga limpia (fuente única de verdad: las clases).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.config import _known_knobs, apply_config, dump_template, load_config
from vslam.core.camera import PinholeCamera
from vslam.frontend.tracker import PnPTracker

CAM = PinholeCamera(fx=100.0, fy=100.0, cx=64.0, cy=48.0, width=128, height=96)


def test_default_is_reference():
    t = PnPTracker(CAM)
    for name, val in _known_knobs(PnPTracker).items():
        assert getattr(t, name) == val, f"{name} difiere sin config"


def test_override_is_per_instance():
    t1 = PnPTracker(CAM, config={"tracker": {"kf_min_inliers": 55,
                                             "depth_max": 12}})
    t2 = PnPTracker(CAM)
    assert t1.KF_MIN_INLIERS == 55
    assert isinstance(t1.DEPTH_MAX, float) and t1.DEPTH_MAX == 12.0  # coerción
    assert t2.KF_MIN_INLIERS == PnPTracker.KF_MIN_INLIERS  # clase intacta
    assert PnPTracker.KF_MIN_INLIERS != 55


def test_typo_fails_at_startup():
    try:
        PnPTracker(CAM, config={"tracker": {"kf_min_inlier": 55}})   # typo
    except KeyError as e:
        assert "KF_MIN_INLIERS" in str(e), "el error debe listar las válidas"
    else:
        raise AssertionError("un typo debe fallar en el arranque")


def test_template_roundtrip():
    tpl = dump_template()
    assert "tracker:" in tpl
    # Toda perilla del tracker aparece en la plantilla.
    for name in _known_knobs(PnPTracker):
        assert name.lower() + ":" in tpl, f"{name} falta en la plantilla"
    # La sección tracker de la plantilla recarga y aplica sin error (JSON:
    # el parser YAML es opcional; aquí se reconstruye el dict a mano).
    section = {}
    in_tracker = False
    for line in tpl.splitlines():
        if line.strip() == "tracker:":
            in_tracker = True
            continue
        if in_tracker:
            if not line.startswith("  ") or not line.strip():
                break
            k, v = line.strip().split(": ")
            section[k] = json.loads(v.lower()) if v in ("True", "False") \
                else json.loads(v)
    t = PnPTracker(CAM, config={"tracker": section})
    for name, val in _known_knobs(PnPTracker).items():
        assert getattr(t, name) == val, f"{name} cambió en el roundtrip"


def test_load_json():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        p.write_text(json.dumps({"tracker": {"ba_window": 9}}), encoding="utf-8")
        cfg = load_config(p)
    t = PnPTracker(CAM, config=cfg)
    assert t.BA_WINDOW == 9


def main() -> int:
    test_default_is_reference()
    test_override_is_per_instance()
    test_typo_fails_at_startup()
    test_template_roundtrip()
    test_load_json()
    print("OK: los 5 tests de la config declarativa (v0.9) pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
