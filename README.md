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

## Estructura

```
Visual-slam/
├── docs/          # Investigación y decisiones de diseño
├── vslam/         # Paquete Python (implementación de referencia)
│   ├── core/      #   Tipos comunes: Frame, PinholeCamera, Trajectory
│   ├── frontend/  #   Tracking: extracción de características, matching
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
```

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
- [ ] **v0.2** — Triangulación + tracking 3D-2D (PnP), selección de keyframes, mapa disperso persistente.
- [ ] **v0.3** — Backend real: grafo de poses con GTSAM, cierre de bucle (bolsa de palabras).
- [ ] **v0.4** — Núcleo C++ del frontend (KLT/ORB) con bindings pybind11.
- [ ] **v0.5** — Mapper de Gaussian Splatting (rasterizador diferenciable) detrás de la interfaz `MapperBase`.
- [ ] **v0.6** — Nodos ROS 2 (frontend/backend/mapper como lifecycle nodes componibles).

## Próximos pasos administrativos

- Elegir licencia (sugerencia: MIT o Apache-2.0) y añadir `LICENSE`.
- Hacer el primer commit y publicar en GitHub.

## Referencias principales

Ver la bibliografía comentada al final de [docs/01_estado_del_arte.md](docs/01_estado_del_arte.md).
