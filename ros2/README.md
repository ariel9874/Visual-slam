# Integración ROS 2 (v0.8 — IMPLEMENTADA; lección 43 en docs/05)

El núcleo de `vslam/` y `cpp/` **no importa ROS**: estos paquetes son una cáscara fina que
traduce los contratos de datos del repo a mensajes y TF (docs/02_arquitectura.md §6).

## Uso (contenedor docker/, ver docker/README.md)

```bash
# compilar el workspace (una vez, cache en volúmenes)
docker compose -f docker/docker-compose.yml run --rm vslam-ros \
    bash -c "cd /workspace/ros2 && colcon build --symlink-install"

# demo TUM con RViz (WSLg en Windows 11; rviz:=false para headless)
docker compose -f docker/docker-compose.yml run --rm vslam-ros \
    bash -c "source /workspace/ros2/install/setup.bash; \
             ros2 launch vslam_ros tum_demo.launch.py rate:=10.0 rviz:=true"

# smoke de integración (verifica odom/keyframes/Path/PointCloud2/TF)
#   (con el launch corriendo) python3 ros2/vslam_ros/test/smoke_pipeline.py 35
```

OJO al sourcear en una línea: `source setup.bash && ros2 launch ... &` pone en
background LA LISTA ENTERA (el shell queda sin overlay y el CLI no resuelve los
tipos custom). Usar `source setup.bash; ros2 launch ... &`.

## Paquetes

### `vslam_msgs`
| Mensaje | Contenido | Origen en el repo |
|---|---|---|
| `Keyframe.msg` | header, id, pose, keypoints, descriptores comprimidos | `vslam/core/frame.py` |
| `PoseGraphEdge.msg` | ids, transformación relativa, matriz de información | factores de `vslam/backend` |
| `TrackingState.msg` | nº de inliers, estado (OK/COASTING/LOST) | diagnóstico del frontend |

### `vslam_ros` — cuatro nodos (frontend/backend/mapper son LIFECYCLE)

Los tres nodos vslam nacen `unconfigured`; el launch los lleva a activo en
orden **consumidores → productor** (mapper, backend, frontend — al revés se
pierden los primeros keyframes; lección 44). Pausa/reanudación en caliente:
`ros2 lifecycle set /vslam_frontend deactivate` (los drivers siguen; el SLAM
ignora frames) y `activate` para reanudar; `cleanup` destruye el tracker.
Demo EuRoC estéreo: `ros2 launch vslam_ros euroc_demo.launch.py` (bf del rig
por parámetro — no viaja en CameraInfo).
| Nodo | Suscribe | Publica | Frecuencia |
|---|---|---|---|
| `dataset_node` | — (lee TUM del disco) | `/camera/image_raw` (CRUDA), `/camera/depth/image_raw`, `/camera/camera_info` | param `rate` |
| `frontend_node` | `/camera/image_raw`+depth (sincronizados), `/camera/camera_info` | `/vslam/odom` (`nav_msgs/Odometry`), `/vslam/tracking_state`, `/vslam/keyframes` (`vslam_msgs/Keyframe`), TF `odom→base_link` | por frame |
| `backend_node` | `/vslam/keyframes`, `/vslam/odom` | `/vslam/optimized_path` (`nav_msgs/Path`), TF `map→odom` = `T_map_kf·T_odom_kf⁻¹` | por keyframe |
| `mapper_node` | `/vslam/keyframes` | `/vslam/map` (`sensor_msgs/PointCloud2` xyzi, frame `map`) | param `map_period` |

(El backend REAL —BA/iSAM2/bucles— corre dentro del tracker en `frontend_node`:
separarlo por tópicos sería re-arquitectura, no cáscara. `backend_node` publica
lo que al ROBOT le corresponde del backend: el Path optimizado y la corrección
`map→odom` de REP-105. El mapper 3DGS foto-realista vive en los ejemplos 07/08
con el contenedor gsplat — este contenedor no trae CUDA.)

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
