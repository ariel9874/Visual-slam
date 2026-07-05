# Visual SLAM — Laboratorio Educativo y Arquitectura Híbrida

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

## Documentación

| Documento | Contenido |
|---|---|
| [docs/01_estado_del_arte.md](docs/01_estado_del_arte.md) | Investigación comparativa: métodos clásicos (ORB-SLAM3, DSO), deep learning (DROID-SLAM, TartanVO) y mapeo de nueva generación (NeRF-SLAM, 3DGS-SLAM). |
| [docs/02_arquitectura.md](docs/02_arquitectura.md) | Diseño del repositorio: contratos de datos, módulos intercambiables, estrategia C++/Python y plan de integración con ROS 2. |
| [docs/03_detectores_y_matchers.md](docs/03_detectores_y_matchers.md) | Catálogo razonado de 12 detectores y 6 matchers (clásicos y aprendidos): idea matemática, costos y guía de selección. |
| [docs/04_hoja_de_ruta_v1.md](docs/04_hoja_de_ruta_v1.md) | El plan completo hacia v1.0: etapas, criterios de aceptación medibles, riesgos y lo que deliberadamente queda fuera. |
| [docs/05_estado_y_plan_de_continuacion.md](docs/05_estado_y_plan_de_continuacion.md) | Documento de traspaso: estado exacto con números, metodología, las 17 lecciones medidas, deuda técnica y el siguiente paso detallado. |

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

Requisitos: Python ≥ 3.9 con `numpy` y `opencv-python` (`matplotlib` opcional para gráficas).

```bash
# 1) Instalar el paquete en modo editable (desde la raíz del repo)
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

[examples/01_monocular_vo.py](examples/01_monocular_vo.py) implementa el ciclo completo de
odometría visual monocular en un solo archivo comentado:

```
imágenes → ORB (características) → matching (ratio test) → matriz esencial (RANSAC)
        → pose relativa (R, t) → composición de trayectoria (hasta escala)
```

Cada bloque del ejemplo indica a qué módulo de `vslam/` corresponde en la arquitectura real.

## Hoja de ruta

- [x] **v0.1** — Esqueleto: VO monocular 2D-2D, contratos de datos, interfaces de backend/mapper.
- [x] **v0.1.5** — Frontend configurable: 6 detectores clásicos + adaptadores aprendidos (SuperPoint/DISK/LightGlue), y benchmark con ATE ([scripts/benchmark_frontends.py](scripts/benchmark_frontends.py)).
- [x] **v0.2** — Triangulación (DLT) + tracking 3D-2D (PnP) contra mapa disperso persistente, con keyframes e inicialización validada por tercera vista ([vslam/frontend/tracker.py](vslam/frontend/tracker.py)). En la secuencia sintética: ATE 0.2 cm con SIFT (vs 4.8 cm del 2D-2D).
- [x] **v0.3** — Backend real: álgebra de Lie SE(3) ([vslam/core/lie.py](vslam/core/lie.py)) + optimizador de grafo de poses en NumPy puro (Gauss-Newton/LM con kernel Huber, [vslam/backend/pose_graph.py](vslam/backend/pose_graph.py)) + demo de cierre de bucle con re-anclaje del mapa ([examples/03_pose_graph_loop.py](examples/03_pose_graph_loop.py)): ATE 1.09 m → 0.05 m con un solo factor de bucle.
- [x] **v0.35** — Backend integrado al tracker: **BA local** con jacobianos analíticos y complemento de Schur ([vslam/backend/bundle_adjustment.py](vslam/backend/bundle_adjustment.py)) — ORB pasa de 6.9 a **2.6 cm** de ATE; **mapa local** (costo acotado), keyframes con intervalo máximo y piso de salud, y **cierre de bucle visual** (reconocimiento de lugar + verificación PnP + corrección de similitud CON escala, [examples/04_loop_closure.py](examples/04_loop_closure.py)): 8.4 → 6.7 cm en la secuencia de corredor con re-visita.
- [x] **v0.4a** — Consistencia: álgebra **Sim(3)** ([vslam/core/lie.py](vslam/core/lie.py)) + grafo de poses genérico por grupo (el experimento de Strasdat reproducido en tests: la deriva de escala que SE(3) no puede corregir), **mapa local por covisibilidad** — el gran salto: el corredor con re-visita pasa de 8.4 a **2.2 cm** (criterio de v0.4 cumplido) — filtro anti-duplicados y cierre de bucle Sim(3) con puente de covisibilidad.
- [ ] **v0.4b** — Relocalización (recuperación de secuestro), culling de puntos, adaptador GTSAM.
- [ ] **v0.45** — Datos reales: loaders TUM/EuRoC/KITTI, modelo de distorsión, benchmark batch + CI.
- [ ] **v0.5** — Tiempo real: núcleo C++ (pybind11) + adaptador GTSAM/iSAM2.
- [ ] **v0.6** — RGB-D y estéreo: escala métrica real.
- [ ] **v0.7** — La tesis cumplida: `GaussianSplattingMapper` asíncrono detrás de `MapperBase`.
- [ ] **v0.8** — ROS 2 (lifecycle nodes componibles, demo con rosbag y cámara real).
- [ ] **v0.9 → v1.0** — Endurecimiento, congelación de API, PyPI y benchmarks publicados.

El plan detallado — con criterios de aceptación medibles por etapa, riesgos y lo que
deliberadamente queda fuera de 1.0 — está en [docs/04_hoja_de_ruta_v1.md](docs/04_hoja_de_ruta_v1.md).

## Próximos pasos administrativos

- Elegir licencia (sugerencia: MIT o Apache-2.0) y añadir `LICENSE`.
- Hacer el primer commit y publicar en GitHub.

## Referencias principales

Ver la bibliografía comentada al final de [docs/01_estado_del_arte.md](docs/01_estado_del_arte.md).
