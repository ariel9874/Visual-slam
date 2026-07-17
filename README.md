# Visual SLAM — Laboratorio Educativo y Arquitectura Híbrida

[![CI](https://github.com/ariel9874/Visual-slam/actions/workflows/ci.yml/badge.svg)](https://github.com/ariel9874/Visual-slam/actions/workflows/ci.yml)

Repositorio de **Visual SLAM (vSLAM)** con doble propósito:

1. **Recurso educativo**: código legible y documentación en español que explica, paso a paso,
   cómo funciona un sistema de localización y mapeo visual — desde la odometría visual clásica
   hasta el mapeo con Gaussian Splatting.
2. **Punto de partida para una arquitectura híbrida de alto rendimiento**: la tesis del proyecto
   es que los mejores sistemas actuales (Photo-SLAM, NeRF-SLAM, SplaTAM) combinan un
   *frontend* de tracking rápido (clásico o aprendido), un *backend* de optimización sobre
   grafos de factores, y un módulo de mapeo denso intercambiable. La estructura del repo está
   diseñada para evolucionar hacia eso, con implementaciones en Python (prototipado/didáctica)
   y C++ (rendimiento), e integración futura con ROS 2.

## ¿Empiezas desde cero? → [aprende-vslam](https://github.com/ariel9874/aprende-vslam)

Este repositorio es un **sistema**: se lee bien si ya sabes de qué va el SLAM.
Si vienes sin conocimientos previos, existe un curso hermano que lo descompone en
niveles independientes y autoejecutables — de *"una imagen es una matriz de números"*
hasta un SLAM completo con bundle adjustment y cierre de bucle, cada nivel con su
examen y su número esperado. **Este repo es el destino; aquel es el camino.**

## Documentación

| Documento | Contenido |
|---|---|
| [docs/01_estado_del_arte.md](https://github.com/ariel9874/Visual-slam/blob/main/docs/01_estado_del_arte.md) | Investigación comparativa: métodos clásicos (ORB-SLAM3, DSO), deep learning (DROID-SLAM, TartanVO) y mapeo de nueva generación (NeRF-SLAM, 3DGS-SLAM). |
| [docs/02_arquitectura.md](https://github.com/ariel9874/Visual-slam/blob/main/docs/02_arquitectura.md) | Diseño del repositorio: contratos de datos, módulos intercambiables, estrategia C++/Python y plan de integración con ROS 2. |
| [docs/03_detectores_y_matchers.md](https://github.com/ariel9874/Visual-slam/blob/main/docs/03_detectores_y_matchers.md) | Catálogo razonado de 12 detectores y 6 matchers (clásicos y aprendidos): idea matemática, costos y guía de selección. |
| [docs/04_hoja_de_ruta_v1.md](https://github.com/ariel9874/Visual-slam/blob/main/docs/04_hoja_de_ruta_v1.md) | El plan completo hacia v1.0: etapas, criterios de aceptación medibles, riesgos y lo que deliberadamente queda fuera. |
| [docs/05_estado_y_plan_de_continuacion.md](https://github.com/ariel9874/Visual-slam/blob/main/docs/05_estado_y_plan_de_continuacion.md) | Documento de traspaso: estado exacto con números, metodología, las 46 lecciones medidas, deuda técnica y el siguiente paso detallado. |
| [docs/06_mapa_denso_3dgs.md](https://github.com/ariel9874/Visual-slam/blob/main/docs/06_mapa_denso_3dgs.md) | El mapa denso 3DGS (v0.7): visita guiada del rasterizador diferenciable, el `GaussianSplattingMapper` y la cadena de ablaciones hasta 21.0 dB (lecciones 39-42). |

## Estructura

```
Visual-slam/
├── docs/          # Investigación y decisiones de diseño
├── vslam/         # Paquete Python (implementación de referencia)
│   ├── core/      #   Tipos comunes: Frame, PinholeCamera, Trajectory
│   ├── frontend/  #   Tracking: registros intercambiables de detectores y matchers
│   ├── backend/   #   Optimización: interfaz de grafo de factores (GTSAM/g2o)
│   ├── mapping/   #   Mapeo intercambiable: disperso hoy, 3DGS/NeRF mañana
│   └── io/        #   Carga de datasets y calibración
├── cpp/           # Núcleo C++ (ruta de rendimiento, espeja a vslam/)
├── ros2/          # Wrappers ROS 2 (planificado, ver ros2/README.md)
├── examples/      # Puntos de entrada educativos, numerados por dificultad
├── scripts/       # Utilidades: generación de datos sintéticos, descargas
└── tests/         # Tests unitarios (geometría, convenciones de pose)
```

## Inicio rápido

Requisitos: Python ≥ 3.10 con `numpy` (<2) y `opencv-python` (`matplotlib` opcional para gráficas).

```bash
# 1) Instalar la librería desde PyPI…
pip install vslam-edu
#    …o en modo editable desde la raíz del repo (los ejemplos y scripts de
#    abajo viven en el repo: clónalo para seguir el recorrido completo)
pip install -e ".[viz]"

# 2) Generar una secuencia sintética con ground truth (no necesitas descargar nada)
python scripts/make_synthetic_sequence.py --output data/synthetic

# 3) Ejecutar la odometría visual monocular sobre la secuencia
python examples/01_monocular_vo.py --images data/synthetic/images --calib data/synthetic/calib.txt --output output/synthetic

# 4) Resultados: output/synthetic/trajectory.txt (formato TUM) y trajectory.png

# 5) Cambiar el frontend por configuración (opciones: docs/03) y comparar con datos:
python examples/01_monocular_vo.py --detector akaze --matcher crosscheck --images data/synthetic/images --calib data/synthetic/calib.txt
python scripts/benchmark_frontends.py     # tabla: matches, inliers, FPS y ATE por frontend

# 6) El salto a SLAM de verdad: tracking 3D-2D (PnP) contra un mapa disperso (v0.2)
python examples/02_pnp_tracking.py --images data/synthetic/images --calib data/synthetic/calib.txt --output output/pnp --gt data/synthetic/groundtruth.txt
python scripts/benchmark_frontends.py --trackers essential,pnp --detectors orb,sift   # 2D-2D vs 3D-2D

# 7) El backend: grafo de poses + cierre de bucle (v0.3, no necesita imágenes)
python examples/03_pose_graph_loop.py --output output/pose_graph

# 8) El sistema completo: mapa local + BA + cierre de bucle visual en una
#    secuencia de corredor que re-visita el inicio (v0.35)
python scripts/make_synthetic_sequence.py --output data/synthetic_loop --motion loop --frames 200
python examples/04_loop_closure.py
```

Los frontends aprendidos (SuperPoint, DISK, LightGlue) son opcionales:
`pip install -e ".[deep]"` + `pip install git+https://github.com/cvg/LightGlue.git`.

También funciona con cualquier carpeta de imágenes ordenadas alfabéticamente (KITTI, TUM,
tus propios videos exportados a frames) si le pasas la calibración `fx fy cx cy` en un `.txt`.

## Punto de entrada educativo

[examples/01_monocular_vo.py](https://github.com/ariel9874/Visual-slam/blob/main/examples/01_monocular_vo.py) implementa el ciclo completo de
odometría visual monocular en un solo archivo comentado:

```
imágenes → ORB (características) → matching (ratio test) → matriz esencial (RANSAC)
        → pose relativa (R, t) → composición de trayectoria (hasta escala)
```

Cada bloque del ejemplo indica a qué módulo de `vslam/` corresponde en la arquitectura real.

## Hoja de ruta

- [x] **v0.1** — Esqueleto: VO monocular 2D-2D, contratos de datos, interfaces de backend/mapper.
- [x] **v0.1.5** — Frontend configurable: 6 detectores clásicos + adaptadores aprendidos (SuperPoint/DISK/LightGlue), y benchmark con ATE ([scripts/benchmark_frontends.py](https://github.com/ariel9874/Visual-slam/blob/main/scripts/benchmark_frontends.py)).
- [x] **v0.2** — Triangulación (DLT) + tracking 3D-2D (PnP) contra mapa disperso persistente, con keyframes e inicialización validada por tercera vista ([vslam/frontend/tracker.py](https://github.com/ariel9874/Visual-slam/blob/main/vslam/frontend/tracker.py)). En la secuencia sintética: ATE 0.2 cm con SIFT (vs 4.8 cm del 2D-2D).
- [x] **v0.3** — Backend real: álgebra de Lie SE(3) ([vslam/core/lie.py](https://github.com/ariel9874/Visual-slam/blob/main/vslam/core/lie.py)) + optimizador de grafo de poses en NumPy puro (Gauss-Newton/LM con kernel Huber, [vslam/backend/pose_graph.py](https://github.com/ariel9874/Visual-slam/blob/main/vslam/backend/pose_graph.py)) + demo de cierre de bucle con re-anclaje del mapa ([examples/03_pose_graph_loop.py](https://github.com/ariel9874/Visual-slam/blob/main/examples/03_pose_graph_loop.py)): ATE 1.09 m → 0.05 m con un solo factor de bucle.
- [x] **v0.35** — Backend integrado al tracker: **BA local** con jacobianos analíticos y complemento de Schur ([vslam/backend/bundle_adjustment.py](https://github.com/ariel9874/Visual-slam/blob/main/vslam/backend/bundle_adjustment.py)) — ORB pasa de 6.9 a **2.6 cm** de ATE; **mapa local** (costo acotado), keyframes con intervalo máximo y piso de salud, y **cierre de bucle visual** (reconocimiento de lugar + verificación PnP + corrección de similitud CON escala, [examples/04_loop_closure.py](https://github.com/ariel9874/Visual-slam/blob/main/examples/04_loop_closure.py)): 8.4 → 6.7 cm en la secuencia de corredor con re-visita.
- [x] **v0.4a** — Consistencia: álgebra **Sim(3)** ([vslam/core/lie.py](https://github.com/ariel9874/Visual-slam/blob/main/vslam/core/lie.py)) + grafo de poses genérico por grupo (el experimento de Strasdat reproducido en tests: la deriva de escala que SE(3) no puede corregir), **mapa local por covisibilidad** — el gran salto: el corredor con re-visita pasa de 8.4 a **2.2 cm** (criterio de v0.4 cumplido) — filtro anti-duplicados y cierre de bucle Sim(3) con puente de covisibilidad.
- [x] **v0.4b** — Relocalización (recuperación de secuestro < 2 s), culling de puntos.
- [x] **v0.45** — Datos reales TUM: **fr2_xyz 0.4 / fr1_xyz 1.8 / fr2_desk 2.1 cm** (nivel ORB-SLAM), SuperPoint/LightGlue integrados.
- [x] **v0.5** — Tiempo real: matching en C++ (pybind11) + GTSAM/iSAM2 + hilo de mapeo + BoW → **46.7 fps** en fr2_desk (pedía 30).
- [x] **v0.6** — Métrico: RGB-D **fr1_desk 2.8 / fr2_xyz 1.5 cm** (escala ≈ 1) y estéreo real EuRoC **V1_01 6.9 cm** (residuo [u,v,u_R] en el BA, también en GTSAM).
- [x] **v0.7** — La tesis cumplida: `GaussianSplattingMapper` detrás de `MapperBase` — **21.0 dB en fr1/desk full-res** (paridad SOTA; gsplat en Docker) y EN VIVO en proceso propio sin robar frames ([docs/06](https://github.com/ariel9874/Visual-slam/blob/main/docs/06_mapa_denso_3dgs.md)).
- [x] **v0.8** — ROS 2: 4 nodos (lifecycle), TF REP-105, demo RViz en vivo (TUM y EuRoC) sin tocar el núcleo.
- [x] **v0.9** — Endurecimiento: config declarativa, reset de mapa, épocas de concurrencia, API congelada (16 nombres), MIT.
- [x] **v1.0** — Empaquetado PyPI (`vslam-edu`), LICENSE, CONTRIBUTING, benchmarks publicados.

### Benchmarks (v1.0, medidos en este repo)

| secuencia | modo | resultado | nota |
|---|---|---|---|
| TUM fr2_xyz | RGB-D métrico | **ATE 1.5 cm** (escala 0.96) | 80 bucles |
| TUM fr1_desk | RGB-D métrico | **ATE 2.8 cm** (escala 1.005) | 0 perdidos |
| TUM fr2_desk | mono + `--fast` | **ATE 1.4-2.1 cm** | **46.7 fps** CPU |
| EuRoC V1_01_easy | estéreo real | **ATE 6.9 cm** (escala 1.002) | SGBM, 27 bucles |
| TUM fr1_desk | 3DGS re-render | **PSNR 21.0 dB** | paridad SOTA, 20 ms/iter |

Cada número con su comando de reproducción en [docs/05 §3.2](https://github.com/ariel9874/Visual-slam/blob/main/docs/05_estado_y_plan_de_continuacion.md).

El plan detallado — con criterios de aceptación medibles por etapa, riesgos y lo que
deliberadamente queda fuera de 1.0 — está en [docs/04_hoja_de_ruta_v1.md](https://github.com/ariel9874/Visual-slam/blob/main/docs/04_hoja_de_ruta_v1.md).

## Referencias principales

Ver la bibliografía comentada al final de [docs/01_estado_del_arte.md](https://github.com/ariel9874/Visual-slam/blob/main/docs/01_estado_del_arte.md).
