"""Evaluación de trayectorias: ATE con alineación de similitud (Umeyama).

Es la métrica estándar para comparar SLAM/VO contra ground truth (la misma
que calcula `evo_ape ... -as`). Se implementa aquí para que el benchmark de
frontends no dependa de herramientas externas.

─── La matemática: alineación de Umeyama (1991) ──────────────────────────────
Problema: la VO monocular devuelve la trayectoria en OTRA escala, rotación y
origen que el ground truth (todo ello inobservable para una cámara sola). Para
comparar formas hay que resolver primero la similitud óptima:

    (s, R, t)* = argmin  Σ_i ‖ g_i − (s·R·p_i + t) ‖²

Solución cerrada: centrar ambas nubes (p̃ = p − μ_p, g̃ = g − μ_g) y tomar la
SVD de la matriz de correlación cruzada

    C = (1/n)·Σ_i p̃_i·g̃_iᵀ = U·D·Vᵀ

    R = V·S·Uᵀ            con S = diag(1, 1, det(V·Uᵀ))  (evita espejos:
                          fuerza det R = +1, rotación propia)
    s = tr(D·S) / σ²_p    con σ²_p = (1/n)·Σ‖p̃_i‖²  (varianza de la fuente)
    t = μ_g − s·R·μ_p

Con la trayectoria alineada, el ATE (Absolute Trajectory Error) es el RMSE de
las distancias punto a punto: mide la consistencia GLOBAL de la trayectoria
(la deriva acumulada), a diferencia del RPE que mide el error local por paso.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_tum_positions(path: str | Path) -> np.ndarray:
    """Carga las posiciones (N, 3) de un archivo de trayectoria formato TUM."""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 1:4]


def umeyama_alignment(source: np.ndarray, target: np.ndarray,
                      with_scale: bool = True
                      ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Similitud (s, R, t) que mejor mapea `source` sobre `target` (fórmulas arriba).

    `with_scale=False` (v0.6): alineación RÍGIDA (s fijado a 1) — la métrica
    de un sistema MÉTRICO (RGB-D/estéreo): si el mapa está en metros de
    verdad, no hay escala que regalar al alineador. R y t salen de la misma
    SVD (el óptimo rotacional no depende de s); solo cambia t = μ_g − R·μ_p.
    """
    src, dst = np.asarray(source, float), np.asarray(target, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_s, dst - mu_d

    C = src_c.T @ dst_c / len(src)                    # correlación cruzada 3x3
    U, D, Vt = np.linalg.svd(C)
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    if with_scale:
        var_src = (src_c ** 2).sum() / len(src)
        s = float(np.trace(np.diag(D) @ S) / var_src)
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate(estimated: np.ndarray, groundtruth: np.ndarray,
        with_scale: bool = True) -> Dict[str, float]:
    """ATE tras alineación de similitud (o RÍGIDA con with_scale=False — la
    métrica honesta de un sistema métrico, v0.6). Unidades del GT.

    Returns:
        dict con: rmse, mean, max (metros del GT), scale (escala recuperada;
        1.0 exacto si with_scale=False), y rmse_pct (rmse como % de la longitud
        del recorrido GT — para comparar secuencias de distinta longitud).
    """
    est, gt = np.asarray(estimated, float), np.asarray(groundtruth, float)
    if len(est) != len(gt):
        raise ValueError(f"Longitudes distintas: est={len(est)} gt={len(gt)} "
                         "(¿se compararon corridas con --max-frames?)")
    s, R, t = umeyama_alignment(est, gt, with_scale=with_scale)
    aligned = (s * (R @ est.T)).T + t
    err = np.linalg.norm(aligned - gt, axis=1)
    path_len = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    rmse = float(np.sqrt((err ** 2).mean()))
    return {
        "rmse": rmse,
        "mean": float(err.mean()),
        "max": float(err.max()),
        "scale": s,
        "rmse_pct": 100.0 * rmse / path_len if path_len > 0 else float("nan"),
    }
