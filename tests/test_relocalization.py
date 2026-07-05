#!/usr/bin/env python3
"""Test de SECUESTRO / relocalizacion (v0.4b).

Criterio de aceptacion de la hoja de ruta (docs/04): recuperarse de un
"secuestro" (un salto brusco de la camara, como una oclusion total o un
teletransporte) en menos de ~2 s de video, sin degradar los numeros de
referencia. Este test lo reproduce de forma medible.

─── El montaje ───
Se recorre la ida del corredor (frames 0..79, se construye el mapa) y luego se
TELETRANSPORTA la camara de vuelta a la zona de salida (se re-alimentan los
frames 3..79). El salto 79->3 es el secuestro: la zona de salida esta MAPEADA
pero, por diseno, sus carteles son DISJUNTOS de los del extremo lejano donde
vive el mapa local (leccion 15) — asi que el PnP contra el mapa local falla y
el sistema DEBE relocalizarse globalmente para reconocer donde esta. Es el
escenario limpio del secuestro: mapeado pero fuera del mapa local.

Nota medida: un salto MENOR (p.ej. 79->110, la vuelta sobre la ida) NO fuerza
reloc — la covisibilidad absorbe el hueco y el tracking continua liso (buena
noticia de robustez, pero no ejercita la reloc). Por eso el salto es a la zona
de salida, la mas lejana en covisibilidad del extremo alcanzado.

─── Lo que verifica ───
(a) El sistema DETECTA la perdida: en < 5 frames tras el salto aparece un
    estado de coast/gate/reloc (no sigue trackeando como si nada).
(b) Se RECUPERA: vuelve a una pose correcta (error vs GT escalado < 5 cm) en
    <= 60 frames tras el salto (~2 s a 30 fps).
(c) La trayectoria POSTERIOR a la recuperacion es consistente (ATE < 5 cm).

Un sistema que solo integra movimiento (velocidad constante) no puede pasar
(b): sin re-medir contra el mapa, el salto queda dentro para siempre.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vslam.core.camera import PinholeCamera
from vslam.evaluation import load_tum_positions, umeyama_alignment
from vslam.frontend.features import create_extractor
from vslam.frontend.matching import create_matcher
from vslam.frontend.tracker import PnPTracker
from vslam.io.dataset import ImageSequenceLoader

DATA = Path("data/synthetic_loop")
PRE, TOTAL = 80, 200                  # ida 0..79; luego teletransporte a la salida
HIJACK_BACK_TO = 3                    # frame de la zona de salida al que se salta
RECOVERED_STATES = {"RELOC", "TRACK", "TRACK+KF", "TRACK+KF+LOOP"}
LOST_STATES = {"COAST", "GATE-REJECT", "RELOC"}


def _ensure_data() -> None:
    """Regenera la secuencia del corredor si no existe (esta en .gitignore)."""
    if (DATA / "images").is_dir() and (DATA / "groundtruth.txt").is_file():
        return
    print("[setup] generando", DATA, "...")
    subprocess.run([sys.executable, "scripts/make_synthetic_sequence.py",
                    "--output", str(DATA), "--motion", "loop",
                    "--frames", str(TOTAL)], check=True)


def main() -> int:
    _ensure_data()
    camera = PinholeCamera.from_file(str(DATA / "calib.txt"))
    gt = load_tum_positions(str(DATA / "groundtruth.txt"))

    frames = list(ImageSequenceLoader(str(DATA / "images")))
    # Orden secuestrado: ida completa + teletransporte a la zona de salida.
    order = list(range(PRE)) + list(range(HIJACK_BACK_TO, PRE))
    jump = PRE                                  # indice alimentado del salto

    tracker = PnPTracker(camera, extractor=create_extractor("orb"),
                         matcher=create_matcher("ratio"),
                         local_window=4, local_ba=True, loop_closure=True)

    est = np.zeros((len(order), 3))
    states = []
    first_track = None
    for fed, src in enumerate(order):
        _, gray = frames[src]
        T, info = tracker.process_frame(gray)
        est[fed] = T[:3, 3]
        states.append(info["state"])
        if first_track is None and info["state"] == "INIT-OK":
            first_track = fed
    first_track = first_track or 0
    gt_order = gt[order]                          # GT alineado a los frames alimentados

    # Gauge: alinear el tramo PRE-salto (bien trackeado) contra su GT y aplicar
    # esa similitud a TODA la trayectoria. Asi el error post-salto es honesto.
    pre = slice(first_track, jump)
    s, R, t = umeyama_alignment(est[pre], gt_order[pre])
    aligned = (s * (R @ est.T)).T + t
    err = np.linalg.norm(aligned - gt_order, axis=1)

    # (a) Deteccion de la perdida en < 5 frames tras el salto.
    window = states[jump:jump + 5]
    detected = any(st in LOST_STATES for st in window)
    n_reloc = len(tracker.reloc_events)

    # (b) Recuperacion: primer frame post-salto ya trackeando con error < 5 cm.
    recovery = None
    for fed in range(jump, len(order)):
        if states[fed] in RECOVERED_STATES and err[fed] < 0.05:
            recovery = fed
            break
    recovered_in = None if recovery is None else recovery - jump

    # (c) ATE del tramo posterior a la recuperacion.
    if recovery is not None:
        seg = err[recovery:]
        ate_post = float(np.sqrt((seg ** 2).mean()))
    else:
        ate_post = float("inf")

    print(f"    frames alimentados: {len(order)} (secuestro en el {jump}: "
          f"frame {PRE-1}->{HIJACK_BACK_TO})")
    print(f"    eventos de reloc: {n_reloc}  {tracker.reloc_events}")
    print(f"    (a) perdida detectada en <5 frames: {detected}  (estados: {window})")
    print(f"    (b) recuperado en: {recovered_in} frames tras el salto "
          f"(limite 60); pose correcta: {recovery is not None}")
    print(f"    (c) ATE post-recuperacion: {100*ate_post:.1f} cm (limite 5)")

    ok = (detected
          and recovered_in is not None and recovered_in <= 60
          and ate_post < 0.05)
    if ok:
        print("OK: el test de secuestro pasa (deteccion + reloc + consistencia).")
        return 0
    print("FALLO: el sistema no se recupero del secuestro segun el criterio.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
