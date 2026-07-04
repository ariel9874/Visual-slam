# Arquitectura del Repositorio

> **Objetivo**: una estructura modular que hoy sirva para aprender y prototipar en Python, y
> mañana escale a un sistema híbrido de alto rendimiento (C++ + GPU) integrado en ROS 2,
> **sin reescribir las fronteras entre módulos**.

## 1. Principios de diseño

1. **Tres capas, tres relojes.** Tracking (por frame, milisegundos), optimización (por
   keyframe, decenas de ms) y mapeo denso (asíncrono, cuando haya presupuesto). Cada capa es un
   módulo con interfaz propia; se comunican por *contratos de datos*, nunca por estado
   compartido. Esto espeja tanto el diseño clásico (PTAM/ORB-SLAM) como los híbridos modernos
   (Photo-SLAM, NeRF-SLAM).
2. **Contratos de datos primero.** `Frame`, `Keyframe`, `MapPoint`, `Trajectory` y los factores
   del grafo son los tipos frontera. Están pensados para traducirse 1:1 a mensajes ROS 2 y a
   structs C++ — si el contrato no cambia, puedes reemplazar la implementación de un módulo
   entero (Python→C++, ORB→SuperPoint, disperso→3DGS) sin tocar el resto.
3. **Python como referencia, C++ como rendimiento.** `vslam/` (Python) es la implementación
   legible y didáctica; `cpp/` espeja la misma estructura de módulos para las rutas calientes,
   expuestas a Python vía pybind11. Regla: primero correcto en Python, luego rápido en C++.
4. **Todo lo intercambiable, detrás de una interfaz.** Extractores de características,
   backends de optimización y mappers implementan clases base abstractas. Cambiar de método =
   cambiar una línea de configuración.
5. **ROS 2 como cáscara, no como esqueleto.** El núcleo no importa ROS. Los nodos de `ros2/`
   solo envuelven módulos y traducen contratos ↔ mensajes. Así el mismo código corre en un
   script, un notebook, un benchmark o un robot.

## 2. Estructura de carpetas

```
Visual-slam/
├── README.md
├── pyproject.toml                  # Paquete Python instalable (pip install -e .)
├── docs/
│   ├── 01_estado_del_arte.md       # Investigación comparativa (por qué esta arquitectura)
│   └── 02_arquitectura.md          # Este documento
│
├── vslam/                          # ── PAQUETE PYTHON (referencia/prototipado) ──
│   ├── core/                       # Contratos de datos compartidos por todas las capas
│   │   ├── camera.py               #   PinholeCamera: intrínsecos K, proyección
│   │   ├── frame.py                #   Frame/Keyframe: imagen + características + pose T_w_c
│   │   ├── geometry.py             #   SE(3), triangulación DLT + filtros, PnP robusto
│   │   ├── lie.py                  #   Álgebra de Lie: Exp/Log de SO(3) y SE(3) (v0.3)
│   │   └── trajectory.py           #   Trayectoria + exportación formato TUM (evaluación)
│   │
│   ├── frontend/                   # ── CAPA 1: TRACKING RÁPIDO (por frame) ──
│   │   ├── features.py             #   Registro de detectores: orb/akaze/brisk/sift/... (docs/03)
│   │   ├── matching.py             #   Registro de matchers: ratio/crosscheck/flann/...
│   │   ├── learned.py              #   Adaptadores GPU opcionales: SuperPoint/DISK/LightGlue
│   │   └── tracker.py              #   TrackerBase + PnPTracker (3D-2D contra mapa, v0.2)
│   │
│   ├── backend/                    # ── CAPA 2: OPTIMIZACIÓN (por keyframe) ──
│   │   ├── factor_graph.py         #   FactorGraphBackend: interfaz + teoría MAP completa
│   │   └── pose_graph.py           #   GaussNewtonPoseGraph: referencia NumPy (GN/LM +
│   │                               #   Huber + gauge) — GTSAM queda como adaptador (v0.35)
│   │
│   ├── mapping/                    # ── CAPA 3: MAPEO INTERCAMBIABLE (asíncrono) ──
│   │   ├── base.py                 #   MapperBase: interfaz común
│   │   └── sparse.py               #   SparsePointMapper: puntos anclados a keyframes,
│   │                               #   re-anclaje en update_poses(), export PLY (v0.2)
│   │                               #   v0.5: GaussianSplattingMapper | futuro: NeRFMapper
│   │
│   ├── io/                         # Entrada/salida: datasets y calibración
│   │   └── dataset.py              #   ImageSequenceLoader; adaptadores TUM/KITTI/EuRoC (TODO)
│   └── evaluation.py               # ATE con alineación de Umeyama (métrica estándar)
│
├── cpp/                            # ── NÚCLEO C++ (ruta de rendimiento) ──
│   ├── CMakeLists.txt              #   Espeja la estructura de vslam/: core/, frontend/, ...
│   └── include/vslam/core/frame.hpp#   Mismos contratos de datos, en C++
│
├── ros2/                           # ── INTEGRACIÓN ROS 2 (planificada) ──
│   └── README.md                   #   Diseño de paquetes vslam_msgs / vslam_ros
│
├── examples/                       # Puntos de entrada educativos, numerados
│   ├── 01_monocular_vo.py          #   VO monocular 2D-2D en un solo archivo comentado
│   ├── 02_pnp_tracking.py          #   Tracking 3D-2D contra mapa disperso (usa el paquete)
│   └── 03_pose_graph_loop.py       #   Backend: deriva → cierre de bucle → mapa re-anclado
├── scripts/
│   ├── make_synthetic_sequence.py  #   Genera secuencia sintética + ground truth (sin descargas)
│   └── benchmark_frontends.py      #   Compara detectores/matchers: inliers, FPS, ATE
└── tests/
    ├── test_pose_recovery.py       #   Verifica convenciones de pose y geometría epipolar
    ├── test_frontends.py           #   Verifica el registro de detectores/matchers
    ├── test_triangulation_pnp.py   #   Verifica DLT, PnP robusto y re-anclaje del mapa
    └── test_pose_graph.py          #   Verifica Exp/Log de Lie y el grafo de poses
```

## 3. Flujo de datos

```
                 imágenes (cámara / dataset)
                          │
                          ▼
┌───────────────────────────────────────────────┐
│ FRONTEND  (vslam/frontend)      ~30-60 Hz     │
│  extracción → matching → estimación de pose   │
│  decide keyframes                             │
└──────────────┬───────────────────┬────────────┘
     pose por frame          keyframes (Frame con pose inicial)
       (odometría)                 │
                                   ▼
                    ┌──────────────────────────────┐
                    │ BACKEND  (vslam/backend)     │
                    │  grafo de factores:          │
                    │  odometría + bucles (+IMU)   │
                    │  → poses optimizadas         │
                    └──────┬───────────────┬───────┘
              poses corregidas      keyframes + poses
              (realimenta al               │
               frontend)                   ▼
                            ┌───────────────────────────┐
                            │ MAPPER  (vslam/mapping)   │
                            │  disperso │ 3DGS │ NeRF   │
                            │  update_poses() tras un   │
                            │  cierre de bucle          │
                            └───────────────────────────┘
```

Dos detalles de la interfaz `MapperBase` que vienen de las lecciones del estado del arte:

- `integrate_keyframe(frame)` es **asíncrono por contrato**: el mapper denso (GS/NeRF) nunca
  puede bloquear al tracking.
- `update_poses(poses_optimizadas)` existe porque los cierres de bucle **deforman el mapa**:
  con gaussianas explícitas se resuelve transformando submapas rígidamente; con campos
  implícitos es un problema abierto. La interfaz obliga a cada mapper a declarar cómo lo hace.

## 4. Contratos de datos → mensajes ROS 2

| Tipo Python (`vslam/core`) | Struct C++ (`cpp/include/vslam`) | Mensaje ROS 2 (futuro `vslam_msgs`) |
|---|---|---|
| `Frame` (id, t, kps, desc, `T_w_c`) | `vslam::Frame` | `vslam_msgs/Keyframe` |
| pose `T_w_c` (4×4, np.ndarray) | `std::array<double,16>` → Sophus::SE3d | `geometry_msgs/PoseStamped` |
| `Trajectory` | — | `nav_msgs/Path` |
| factores de odometría/bucle | `vslam::OdometryFactor` | `vslam_msgs/PoseGraphEdge` |
| mapa disperso | `std::vector<MapPoint>` | `sensor_msgs/PointCloud2` |

**Convención de poses (¡fijada para todo el repo!)**: `T_w_c` ∈ SE(3) transforma puntos del
frame de la **cámara** al frame del **mundo**; la columna de traslación es la posición de la
cámara en el mundo. Ejes de cámara estilo OpenCV: +Z hacia delante, +X derecha, +Y abajo.
En ROS 2 se convertirá a la convención REP-103/REP-105 en la capa de wrappers, nunca en el núcleo.

## 5. Estrategia de dos lenguajes

| | Python (`vslam/`) | C++ (`cpp/`) |
|---|---|---|
| Rol | referencia legible, prototipado, evaluación | rutas calientes: tracking, BA, rasterización |
| Módulos | todos | los que demuestren ser cuello de botella (perfil primero) |
| Puente | — | pybind11: `import vslam_cpp as fast` con la MISMA interfaz |
| Optimización | interfaz `FactorGraphBackend` | GTSAM (primera opción) / g2o / Ceres |
| GPU | PyTorch para mappers GS/NeRF | CUDA del rasterizador 3DGS |

La regla de oro: **una clase Python y su gemela C++ implementan la misma interfaz**, y los
tests de `tests/` corren contra ambas para garantizar equivalencia.

## 6. Plan de integración ROS 2 (resumen; detalle en [ros2/README.md](../ros2/README.md))

- `vslam_msgs`: mensajes de los contratos de datos.
- `vslam_ros`: tres *lifecycle nodes* componibles (frontend, backend, mapper) → composición en
  un solo proceso con transporte intra-proceso (cero copias) cuando haga falta rendimiento.
- TF: `map → odom → base_link` (REP-105). El frontend publica `odom→base_link` a alta
  frecuencia; el backend corrige `map→odom` a baja frecuencia.

## 7. Hoja de ruta técnica

| Versión | Entregable | Módulos que toca |
|---|---|---|
| v0.1 (actual) | VO monocular 2D-2D educativa + interfaces | examples, core, frontend, io |
| v0.2 | Triangulación, PnP 3D-2D, keyframes, mapa disperso | frontend, mapping |
| v0.3 | Grafo de poses con GTSAM + cierre de bucle (BoW) | backend |
| v0.4 | Frontend C++ (ORB/KLT) + pybind11 | cpp |
| v0.5 | `GaussianSplattingMapper` (rasterizador diferenciable) | mapping |
| v0.6 | Nodos ROS 2 + demo en robot/rosbag | ros2 |
