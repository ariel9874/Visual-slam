# Contenedor de pruebas: ROS 2 + Visual-SLAM

Entorno Linux reproducible donde conviven **ROS 2** y el **núcleo Python de
`vslam/`**, para (a) correr tests/ejemplos en Linux y (b) prototipar los
wrappers `vslam_ros` / `vslam_msgs` planificados en [`../ros2/README.md`](../ros2/README.md)
(v0.6/v0.8 de la hoja de ruta).

## Qué trae

- **ROS 2** — por defecto `osrf/ros:kilted-desktop-full` (la imagen que ya está
  en disco → build instantáneo). Ubuntu 24.04, Python 3.12.
- **Núcleo vslam** — numpy 1.26, OpenCV 4.6 (con MAGSAC y SIFT), matplotlib,
  scipy, pytest, todo desde **apt** para compartir ABI con ROS (ver el porqué en
  el encabezado del [`Dockerfile`](Dockerfile)). El paquete `vslam` se importa
  vía `PYTHONPATH` montando el repo — sin `pip install`.
- **Puentes** `cv_bridge`, `vision_opencv`, `image_transport`.

Verificado (`docker compose ... run`): `import vslam` OK, `ros2` OK, y **21/21
tests de geometría pasan** dentro del contenedor.

## Uso

Desde la raíz del repo:

```bash
# Construir (rápido: la base kilted ya está local)
docker compose -f docker/docker-compose.yml build

# Shell interactiva (--rm: contenedor efímero; el repo se monta en /workspace)
docker compose -f docker/docker-compose.yml run --rm vslam-ros
```

Dentro del contenedor (el repo vive en `/workspace`, montado en vivo):

```bash
# Los tests de geometría (21)
for t in test_pose_recovery test_frontends test_triangulation_pnp \
         test_pose_graph test_bundle_adjustment; do python3 tests/$t.py; done

# Datos sintéticos + ejemplo del corredor (regenera data/, está en .gitignore)
python3 scripts/make_synthetic_sequence.py --output data/synthetic_loop \
        --motion loop --frames 200
python3 examples/04_loop_closure.py       # salidas a output/ (visibles en Windows)

# ROS 2 disponible de una
ros2 pkg list | wc -l
```

Editas los archivos en Windows y se reflejan al instante dentro (bind mount).
Lo escrito en `output/` y `data/` aparece en tu carpeta del repo.

## Variante LTS (Jazzy, soporte hasta mayo 2029)

Kilted es non-LTS. Para el largo plazo, una sola línea:

```bash
docker compose -f docker/docker-compose.yml build \
    --build-arg ROS_IMAGE=osrf/ros:jazzy-desktop-full
```

(o edita `ROS_IMAGE` en [`docker-compose.yml`](docker-compose.yml)). Descarga
la imagen jazzy la primera vez.

## Notas de entorno

- **Sin host networking**: DDS usa memoria compartida dentro del contenedor
  (`ipc: host`, `ROS_LOCALHOST_ONLY=1`). En Docker Desktop/Windows el host
  networking es limitado; para comunicar con nodos ROS *fuera* del contenedor
  habría que ajustar la red (documentar cuando llegue v0.8).
- **GUI (rviz2)**: comentado en el compose. En Windows 11 con WSLg, descomenta
  `DISPLAY` en `docker-compose.yml`. Los ejemplos hacen `savefig()` a `output/`
  (backend `Agg`), así que para los tests no hace falta display.
- **GPU / frontend aprendido**: el runtime `nvidia` está disponible, pero la
  imagen desktop-full no trae CUDA ni torch, así que SuperPoint/LightGlue no
  corren aquí. Requeriría una imagen base CUDA + torch (bloque `deploy` de GPU
  ya esbozado en el compose).
- **OpenCV 4.6** (apt) vs `pyproject` (`>=4.8`): 4.6 cubre lo que usa el
  tracker (USAC_MAGSAC ≥ 4.5, SIFT en el módulo principal). Si algún día se
  necesita una feature de 4.8+, `pip install opencv-python-headless` — pero ojo,
  arrastra numpy>=2 y rompería los bindings de ROS; mejor una imagen aparte.
```
