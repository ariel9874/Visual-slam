# Integración ROS 2 (planificada — v0.6)

El núcleo de `vslam/` y `cpp/` **no importa ROS**: estos paquetes serán una cáscara fina que
traduce los contratos de datos del repo a mensajes y TF (docs/02_arquitectura.md §6).

## Paquetes previstos

### `vslam_msgs`
| Mensaje | Contenido | Origen en el repo |
|---|---|---|
| `Keyframe.msg` | header, id, pose, keypoints, descriptores comprimidos | `vslam/core/frame.py` |
| `PoseGraphEdge.msg` | ids, transformación relativa, matriz de información | factores de `vslam/backend` |
| `TrackingState.msg` | nº de inliers, estado (OK/COASTING/LOST) | diagnóstico del frontend |

### `vslam_ros` — tres lifecycle nodes componibles
| Nodo | Suscribe | Publica | Frecuencia |
|---|---|---|---|
| `frontend_node` | `/camera/image_raw`, `/camera/camera_info` | `/vslam/odom` (`nav_msgs/Odometry`), `/vslam/keyframes` (`vslam_msgs/Keyframe`), TF `odom→base_link` | por frame (~30 Hz) |
| `backend_node` | `/vslam/keyframes`, `/vslam/loop_candidates` | `/vslam/optimized_path` (`nav_msgs/Path`), TF `map→odom` | por keyframe |
| `mapper_node` | `/vslam/keyframes` + poses optimizadas | `/vslam/map` (`sensor_msgs/PointCloud2` o render GS) | asíncrona |

## Decisiones de diseño

- **TF según REP-105**: `map → odom → base_link`. El frontend publica `odom→base_link`
  continuo y suave; el backend corrige `map→odom` a saltos (tras optimizar/cerrar bucle).
  Así los consumidores eligen: control usa `odom` (suave), navegación usa `map` (consistente).
- **Conversión de ejes en la frontera**: el núcleo usa la convención óptica de OpenCV
  (+Z delante); los wrappers convierten a REP-103 (x delante, z arriba). Nunca dentro del núcleo.
- **Composición intra-proceso**: los tres nodos serán *components* cargables en un solo
  proceso para transporte con cero copias (imágenes y keyframes son pesados).
- **Lifecycle nodes**: configurar/activar/desactivar limpiamente es esencial en un robot real
  (p. ej. reiniciar el SLAM sin reiniciar los drivers de cámara).
- **QoS**: imágenes con `SensorDataQoS` (best effort); keyframes y grafo con `reliable`,
  porque perder un keyframe corrompe el mapa.
