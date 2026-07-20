# Estado del Proyecto y Plan de Continuación

> **Para quién es este documento**: cualquier sesión futura (humana o de un
> asistente) que continúe el desarrollo. Contiene TODO lo necesario para
> retomar el trabajo sin re-descubrir nada: contexto, metodología, estado
> exacto con números, lecciones medidas, deuda técnica y el siguiente paso
> detallado. Última actualización: julio 2026, **v1.1 (VIO) EN PROGRESO** —
> hito 1 (preintegración IMU, lección 47) HECHO y verde. v1.0 committeada — release:
> pyproject `vslam-edu` 1.0.0, licencia MIT, CONTRIBUTING, tabla de benchmarks
> en el README (fr2_xyz 1.5 / fr1_desk 2.8 cm métrico, fr2_desk 46.7 fps,
> V1_01 6.9 cm estéreo, 3DGS 21.0 dB). Pendientes MANUALES de Ariel:
> `python -m build` + `twine upload` (vslam-edu aún NO está en PyPI),
> tag `v1.0.0` + GitHub Release, video demo. Historial reciente: v0.7 3DGS
> (lecciones 39-42), v0.8 ROS 2 (43-44), v0.9 endurecimiento (45-46);
> v0.6 RGB-D fr1 2.8/fr2 1.5 cm, estéreo V1_01 6.9 cm (35-38).

---

## 1. Qué es este proyecto y para quién

Repositorio de **Visual SLAM monocular** con doble tesis (ver README y docs/01 §5):
1. **Educativo**: cada técnica lleva su matemática como comentario en el código
   (bloques `─── La matemática ───`) y cada decisión de diseño lleva su
   justificación MEDIDA. El repo debe poder leerse como un curso.
2. **Arquitectura híbrida**: frontend intercambiable + backend de grafos de
   factores + mapeo denso intercambiable (3DGS en el futuro), camino a
   producción (C++/ROS 2). La definición completa de v1.0 y sus etapas están
   en [docs/04_hoja_de_ruta_v1.md](04_hoja_de_ruta_v1.md).

**El dueño**: Ariel Vazquez — Ph.D. en CS, ingeniero de visión 3D profesional
(LiDAR-inertial, GTSAM, fotogrametría). Es EXPERTO: el tono correcto es de
colega arquitecto, no de tutorial. Trabaja en español; el código usa
identificadores en inglés y comentarios/docs en español. Aprueba por fases
("me gusta, continúa") y espera que el trabajo avance solo siguiendo la hoja
de ruta acordada.

---

## 2. Metodología de trabajo establecida (NO negociable)

Estas reglas emergieron del propio desarrollo y han demostrado su valor:

1. **Nada se da por hecho sin ejecutarlo.** Cada pieza se verifica corriendo
   el código de verdad: tests unitarios con geometría sintética exacta +
   ejemplos sobre secuencias generadas + métrica ATE contra ground truth.
2. **Sin métrica no hay ingeniería.** Todo cambio se reporta con números
   antes/después. Los umbrales se calibran con barridos de parámetros
   (ejemplo real: `MIN_INIT_FLOW_PX` se eligió midiendo ATE con 25/40/60 px),
   nunca a ojo.
3. **Las lecciones medidas se documentan EN EL CÓDIGO**, junto a la línea que
   las encarna, con los números medidos. Ver §5: hay 46 y son oro educativo.
   Los enfoques probados y DESCARTADOS también se documentan (con su medición)
   como comentario donde habrían vivido.
4. **Ciclo de depuración**: cuando algo falla, NO adivinar en el código —
   escribir un script de diagnóstico en el scratchpad de la sesión que aísle
   el paso exacto (hay muchos precedentes: matching → E → recoverPose;
   PnP por etapas; divergencia frame a frame entre dos corridas).
5. **Interfaces primero** (docs/02): `TrackerBase`, `FactorGraphBackend`,
   `MapperBase` y los contratos de datos (`Frame`, `T_w_c` 4×4) no se rompen.
6. **Convenciones fijadas** (violarlas rompe tests):
   - `T_w_c`: transforma puntos de cámara a mundo; subíndices se "cancelan":
     `T_w_c2 = T_w_c1 · T_c1_c2`. Ejes OpenCV (+Z delante, +Y abajo).
   - Vector tangente: `[ρ, ω]` (traslación primero); Sim(3): `[ρ, ω, λ]`.
   - Gauge monocular: profundidad mediana = 1.0 en la inicialización.
   - Formato de trayectorias: TUM (`t tx ty tz qx qy qz qw`).

### Particularidades del entorno (Windows 11, Python 3.13)

- **Consola cp1252**: NUNCA usar caracteres fuera de cp1252 en `print()` de
  scripts (─, →, ═ rompen con UnicodeEncodeError). En comentarios/docstrings
  sí se puede (los archivos son UTF-8).
- **GTSAM**: no hay wheel en PyPI para Windows, PERO conda-forge SÍ tiene
  `gtsam` para win-64 (4.2.2). Instalado en el env conda `vslam` (julio 2026);
  el adaptador GTSAM ya es viable en esta máquina. La referencia NumPy sigue
  siendo la implementación local (educativa). gtsam exige `numpy<2`.
- PowerShell: `Select-Object -First N` sobre un pipe de Python causa exit 255
  (BrokenPipeError) — usar `-Last` o archivos. Los heredocs de comillas
  escapadas dentro de `python -c "..."` fallan: usar scripts en el scratchpad.
- **Entorno de trabajo (julio 2026)**: env conda `vslam` (Miniforge en
  `C:\Users\ariel\miniforge3`), Python 3.11. Usar su python:
  `C:\Users\ariel\miniforge3\envs\vslam\python.exe`. Instalado y verificado:
  gtsam 4.2.2, torch 2.6.0+cu124 (CUDA ve la RTX 4070), lightglue (SuperPoint+
  LightGlue, cvg), kornia 0.8.3, opencv 4.11, numpy 1.26.4, matplotlib, pytest.
  Los tests siguen con runner `__main__` (`python tests/test_X.py`). El python
  del sistema (3.13, WindowsApps) NO tiene torch/gtsam: no usarlo.
- **Contenedor ROS 2 (julio 2026)**: entorno Linux reproducible en `docker/`
  para correr tests/ejemplos en Linux y prototipar los wrappers ROS de v0.6/v0.8.
  Base `osrf/ros:kilted-desktop-full` (ya en disco; el distro es un `ARG` →
  cambiar a `osrf/ros:jazzy-desktop-full` para LTS es una línea). Deps del núcleo
  desde apt (numpy 1.26, opencv 4.6 con MAGSAC+SIFT, matplotlib, scipy, pytest)
  para compartir ABI con ROS y respetar `numpy<2`; el paquete `vslam` entra por
  `PYTHONPATH` (repo montado en `/workspace`), sin `pip install`. Verificado:
  `import vslam` OK, `ros2` OK, **21/21 tests de geometría pasan** dentro.
  Build/uso en `docker/README.md`. Limitaciones conscientes: la imagen no trae
  CUDA/torch (SuperPoint/LightGlue NO corren ahí — requerirían base CUDA aparte);
  GUI (rviz2) comentada (WSLg en Win 11); sin host networking (DDS por memoria
  compartida, `ipc: host` + `ROS_LOCALHOST_ONLY`).

---

## 3. Estado exacto por versión (v0.1 → v1.0)

### 3.1 Qué existe (por versión, con sus números)

| Versión | Contenido | Resultado medido |
|---|---|---|
| v0.1 | VO monocular 2D-2D (examples/01), contratos de datos, docs 01-02 | ATE 13.1 cm (secuencia forward, ORB) |
| v0.1.5 | Registro de detectores/matchers (6 clásicos + adaptadores aprendidos), benchmark, docs/03 | SIFT 4.8 / ORB 12.9 / BRISK 31.9 cm (2D-2D) |
| v0.2 | PnP contra mapa disperso (PnPTracker), triangulación DLT, init validada por 3ª vista | ORB 6.9 / SIFT 0.2 cm |
| v0.3 | Álgebra SE(3) (lie.py), grafo de poses NumPy (GN/LM/Huber), examples/03 | bucle simulado: 1.09 m → 0.05 m |
| v0.35 | BA local (Schur), mapa local, observaciones, cierre de bucle visual, examples/04, secuencia corredor | ORB forward 2.6→3.1 cm; corredor 8.4/6.7 cm |
| v0.4a | Álgebra Sim(3), grafo genérico por grupo, bucle Sim(3), **covisibilidad**, filtro anti-duplicados | **corredor 2.2 cm** (criterio < 3 cumplido) |
| v0.4b | **Relocalización** (PnP global), **compuerta de movimiento** (emparejada), **culling de puntos**, helper `_match_against_kf` compartido, test de secuestro | corredor **2.0 cm**; secuestro recuperado en **2 frames**; culling **-33.9%** del mapa |
| v0.45 CERRADA | **Datos REALES**: distorsión Brown-Conrady, loader TUM RGB-D + EuRoC, benchmark batch, CI, **matching guiado**, **BA global offline**, métrica de trayectoria final de KFs, **SuperPoint+LightGlue** integrados | **TUM movimiento moderado (final de KFs): fr2_xyz 0.4 / fr1_xyz 1.8 / fr2_desk 2.1 cm**, 0 colapsos. Límites medidos (lecciones 28-29): fr1 handheld (SuperPoint rescata: 560→140) y fr3 deriva. Sintético: 02 2.4, 04 1.7, secuestro 1.1 cm |
| v0.5 CERRADA | **Tiempo real**: perfilado dirigido (regla 3), BA GTSAM batch + **iSAM2 incremental**, **matching guiado en C++** (pybind11, `vslam_cpp`), **hilo de mapeo** (async, delta pendiente), **BoW** (k-medias Hamming + tf·idf). Stack rápido opt-in (`--fast`); referencia NumPy intacta | **Criterio CUMPLIDO: 46.7 fps en fr2_desk** (pedía 30) a 640×480 en CPU, mediana 17 ms, p99 73 ms, ATE-KF 1.4 cm (paridad). Trayectoria: 4.3→9.5→18.7→25.7→46.7 fps (lecciones 30-34: medir, mover, eliminar) |
| v0.6 hito 1 | **RGB-D MÉTRICO**: profundidad en loader TUM, init instantánea por retro-proyección (`_metric`), puntos de KF desde profundidad, Umeyama RÍGIDO (`--depth`), **bucle métrico en SE(3)** (lección 35: Sim(3) re-escalaba el mapa métrico y componía — 22 cm/escala 2.09 → fix), test del cebo de escala | **fr2_xyz 4.7 cm MÉTRICO (escala 1.036), 0 perdidos — criterio <5 CUMPLIDO** (sin bucles: 1.1 cm). fr1_desk 6.7 cm en aquel momento — que resultó medido sobre un mapa MIXTO (ver hito 2) |
| v0.6 hito 2 | **Residuos de PROFUNDIDAD en el BA** (estéreo virtual ORB-SLAM2: u_R = u − bf/z, residuo [u,v,u_R], `STEREO_BF=40`) + **bug raíz de fr1_desk**: su stream depth arranca tarde → init MONOCULAR accidental → mapa mixto gauge/metros con `_metric=False` (escala 1.008 de casualidad); fix: el driver espera profundidad + invariante "puntos desde depth SOLO en mapa métrico" (lección 36) | **CRITERIO v0.6 CUMPLIDO en ambas**: **fr1_desk 2.8 cm MÉTRICO** (escala 1.005, 0 perdidos, online 3.4) y **fr2_xyz 1.5 cm** (escala 0.96, 80 bucles; antes 4.7). Ablación fr1 sin residuo: 12.8 cm, 244 perdidos — el residuo ES lo que cruza el episodio 200-340 |
| v0.6 hito 3 | **ESTÉREO REAL** (EuRoC): `EuRoCStereoRig` (rectificación cv2.stereoRectify, bf desde P2) + `EuRoCStereoLoader` (disparidad StereoSGBM → profundidad densa, MISMA firma que RGB-D); `examples/06 --stereo`. La cámara derecha virtual se vuelve real: u_R medido, mismo residuo del BA (lección 37) | **V1_01_easy ESTÉREO (final de KFs): 6.9 cm rmse, escala similitud 1.002** (234 KFs, 27 bucles SE(3), 34 perdidos) — métrico real sobre dron 6-DoF. Rig verificado: baseline 11.01 cm, bf 48.0. Tests: tests/test_stereo.py (2, sin el dataset) |
| v0.7 hito 1-2 | **MAPA DENSO 3DGS**: rasterizador diferenciable EWA (gaussian_render.py, PyTorch puro) + `GaussianSplattingMapper` detrás de MapperBase (siembra desde la nube dispersa, optimize=renderiza-y-compara, update_poses rígido por submapa). Ejemplo 07 sobre fr1/desk (lección 39) | Rasterizador y mapper verdes: sobreajuste de vista **PSNR > 30 dB**, multi-vista **> 30 dB**, gradiente por diferencias finitas, update_poses rígido exacto. fr1/desk real: EN PROGRESO (render a resolución reducida; full-res = gemela gsplat pendiente). Tests: test_gaussian_render.py (3), test_gaussian_mapper.py (2) |

(La tabla llega hasta v0.7 hitos 1-2; v0.7 completa, v0.8 ROS 2, v0.9
endurecimiento y v1.0 release están detallados con sus números en §7 y en
las lecciones 39-46 de §5.)

### 3.2 Números de referencia actuales (para detectar regresiones)

Comandos y valores esperados (tolerancia ~±20% por aleatoriedad de RANSAC):

```bash
# Tests: 21 de geometría + 1 de secuestro (script reproducible), todos OK
python tests/test_pose_recovery.py        # 3 tests
python tests/test_frontends.py            # 5 tests
python tests/test_triangulation_pnp.py    # 5 tests
python tests/test_pose_graph.py           # 6 tests (incl. Sim3 + Strasdat)
python tests/test_bundle_adjustment.py    # 2 tests
python tests/test_relocalization.py       # secuestro (v0.4b): gate->coast->reloc
python tests/test_camera_distortion.py    # 3 tests distorsión de lente (v0.45)
python tests/test_euroc_loader.py         # 3 tests loader EuRoC (fixture, v0.45)

# Datos sintéticos (si data/ no existe — está en .gitignore):
python scripts/make_synthetic_sequence.py --output data/synthetic
python scripts/make_synthetic_sequence.py --output data/synthetic_loop --motion loop --frames 200

# Ejemplo 02 (forward): esperado ~2.4 cm ORB (era 3.1 antes del matching guiado
# de v0.45; el guiado mejoró init→final)
python examples/02_pnp_tracking.py --images data/synthetic/images --calib data/synthetic/calib.txt --output output/pnp --gt data/synthetic/groundtruth.txt

# Ejemplo 04 (corredor): esperado ~1.7 cm con y sin bucle (2 bucles; era 2.2
# pre-culling, 2.0 pre-guiado — cada mejora de v0.4b/v0.45 bajó un poco el ATE)
python examples/04_loop_closure.py

# Test de secuestro (v0.4b): teletransporta 79->3 y verifica gate+coast+reloc.
# Esperado: perdida detectada en <5 frames, reloc contra KF0 en ~2 frames,
# ATE post-recuperación ~2 cm. Regenera data/synthetic_loop si falta.
python tests/test_relocalization.py

# Datos REALES TUM RGB-D (v0.45). Bajar en data/tum/ (está en .gitignore):
#   curl -LO https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
#   (extraer con tar -xzf). Números esperados (trayectoria FINAL de KFs, la
#   columna ATE-KF; la online es peor y NO es la métrica — lección 25):
#   fr1_xyz 1.8 cm | fr2_desk 2.1 cm | fr2_xyz 0.4 cm | ≤5 perdidos, 0 relocs
python examples/05_tum_rgbd.py --root data/tum/rgbd_dataset_freiburg1_xyz
python scripts/benchmark_tum.py --data data/tum          # tabla batch por secuencia

# RGB-D MÉTRICO (v0.6, --depth): ATE con alineación RÍGIDA (sin regalar escala)
# y la escala de similitud como chequeo (≈1.0 = mapa en metros de verdad).
# Con residuos de profundidad en el BA (hito 2, columna FINAL-KF):
#   fr1_desk 2.8 cm métrico (escala 1.005, 0 perdidos, 3 bucles; online 3.4)
#   fr2_xyz  1.5 cm métrico (escala 0.96, 80 bucles SE3, 0 perdidos)
#   El driver salta frames iniciales SIN depth ("saltados sin depth: 17" en
#   fr1_desk es lo esperado — sin eso, init monocular y mapa mixto, lección 36).
#   Ablación (STEREO_BF=0, solo init métrica): fr1_desk 12.8 cm y 244 perdidos —
#   el residuo de profundidad es lo que cruza el episodio 200-340 (lección 28);
#   no re-abrir la biestabilidad con perillas si estos números se sostienen.
python examples/05_tum_rgbd.py --root data/tum/rgbd_dataset_freiburg1_desk --depth
python examples/05_tum_rgbd.py --root data/tum/rgbd_dataset_freiburg2_xyz --depth
python tests/test_rgbd.py                 # 5 tests (Umeyama rígido, init, bucle SE3,
                                          #          BA con u_R nulo/observable, loader)
# --fast --depth (stack rápido iSAM2 con FACTOR ESTÉREO GTSAM, lección 38):
#   fr2_xyz 1.4-1.5 cm métrico — paridad con NumPy, a 30+ fps. Es el ANCLA de
#   regresión del stack rápido. OJO: fr1_desk --fast tiene varianza 2.6↔400 cm
#   entre corridas (lección 49, deuda §8) — el 2.5 histórico era una muestra;
#   para fr1_desk la referencia fiable es la ruta NumPy (2.8 cm).
python examples/05_tum_rgbd.py --root data/tum/rgbd_dataset_freiburg2_xyz --depth --fast

# ESTÉREO REAL (v0.6 hito 3, EuRoC): cam0+cam1, disparidad SGBM → profundidad
# métrica → MISMA ruta RGB-D. La izquierda RECTIFICADA es la cámara del tracker.
#   V1_01_easy (final de KFs): 6.9 cm métrico, escala similitud 1.002, 234 KFs
#   El host oficial (robotics.ethz.ch) suele estar caído; mirror ASL en HF:
#   https://huggingface.co/datasets/pepijn223/euroc-mirror (V1_01_easy.zip ~1.1GB)
#   Descomprimir en data/euroc/V1_01_easy (queda mav0/cam0, cam1, ground truth).
python examples/06_euroc.py --root data/euroc/V1_01_easy --stereo
python tests/test_stereo.py               # 2 tests (rig: bf=fx·b; depth por disparidad)

# MAPA DENSO 3DGS (v0.7, torch+CUDA): rasterizador diferenciable + mapper.
python tests/test_gaussian_render.py      # 3 (proyección, gradiente FD, sobreajuste >30 dB)
python tests/test_gaussian_mapper.py      # 2 (multi-vista >30 dB, update_poses rígido)
python examples/07_gaussian_mapping.py --root data/tum/rgbd_dataset_freiburg1_desk
#   (siembra desde la nube dispersa + optimiza; PSNR de re-render a res. reducida)

# PREINTEGRACIÓN IMU (v1.1 hito 1): referencia NumPy + equivalencia GTSAM +
# dead-reckoning sobre V1_01 real (esperado: rot mediana 0.33 grados / pos
# 4.4 cm / p90 7.7 cm en ventanas de 1 s contra el GT de estado).
python tests/test_imu_preintegration.py   # 4 (exactitud, sesgo 1er orden, gtsam, EuRoC)

# INIT VI ESTÁTICA (v1.1 hito 2): ventana quieta + b_g + dir(g) + R_wb.
#   Esperado (las 3 V1 vs GT): b_g err ≤ 2.3e-3 rad/s; dir(g) cruda ≤ 2.6
#   grados (b_a manda, lección 48); con b_a del GT ≤ 0.63 grados.
python tests/test_imu_init.py             # 4 (sintético, sin-reposo, degenerados, EuRoC)

# MODO VI DE iSAM2 (v1.1 hito 3a, requiere gtsam): el par nulo/observable.
#   Esperado: escala corrupta 1.3 → sin IMU ~1.33 (gauge), con IMU ~0.99;
#   b_g err ~5e-4; b_a err ~0.01 (desde cero); reset re-anclado, 0 fallos.
python tests/test_imu_isam2.py            # 3 (escala, sesgos, reset)
```

Notas: el modo `--no-ba` del ejemplo 04 da ~200 cm — es un modo de fallo
CONOCIDO (colapso de escala sin BA en esa secuencia), no un objetivo a
arreglar. El benchmark (`scripts/benchmark_frontends.py`) tiene números
publicados en README de ANTES de integrar el BA por defecto: si se re-corre,
los valores `pnp` saldrán mejores que los documentados.

### 3.3 Mapa de archivos (qué hay en cada uno)

```
vslam/core/      camera.py (pinhole+math), frame.py (contrato), geometry.py
                 (invert_se3, triangulate_two_views con filtros, solve_pnp),
                 lie.py (SO3/SE3/Sim3 Exp/Log — Sim3 validada vs serie),
                 trajectory.py (TUM + cuaterniones Shepperd)
vslam/frontend/  features.py (registro: orb/akaze/brisk/sift/kaze/gftt-orb),
                 place_recognition.py (BoW: k-medias Hamming/L2 + índice
                 invertido + tf·idf, v0.5),
                 matching.py (ratio/crosscheck/flann, firma con kps para
                 aprendidos), learned.py (SuperPoint/DISK/LightGlue,
                 EXPERIMENTAL, requiere [deep]), tracker.py (PnPTracker: el
                 corazón del sistema, ~850 líneas — leerlo entero antes de
                 tocarlo. v0.45: _guided_match (matching por reproyección),
                 _local_ref_kf (re-anclaje del mapa local tras reloc),
                 global_bundle_adjustment (BA global OFFLINE, lo llama el driver),
                 keyframe_trajectory (métrica final vs online),
                 _mapping_worker/wait_mapping (HILO DE MAPEO async, v0.5))
vslam/backend/   factor_graph.py (interfaz + teoría MAP), gtsam_ba.py (BA batch
                 GTSAM, ≡ referencia NumPy), gtsam_isam2.py (BA INCREMENTAL
                 iSAM2 con reset por época, v0.5), pose_graph.py
                 (GaussNewtonPoseGraph genérico se3/sim3),
                 bundle_adjustment.py (BA con Schur + jacobianos analíticos),
                 imu_preintegration.py (preintegración IMU en la variedad,
                 Forster TRO'17: referencia NumPy con jacobianos de sesgo y
                 covarianza 9×9 [φ,v,p]; residuo del factor documentado, v1.1),
                 imu_init.py (init VI estática: detector de ventana quieta +
                 b_g + gravedad en el cuerpo + R_wb yaw=0, v1.1 hito 2)
vslam/mapping/   base.py (MapperBase), sparse.py (puntos+observaciones+
                 covisibilidad+re-anclaje SE3/Sim3+apply_similarity+PLY+
                 culling v0.4b: _active/cull_points/active_count)
vslam/core/      camera.py: + distorsión Brown-Conrady (v0.45): campo
                 distortion, undistort_points (cv2), from_file parsea k1..k3
vslam/           evaluation.py (Umeyama + ATE), io/dataset.py (loader genérico +
                 TUMRGBDLoader/tum_camera/read_tum_trajectory/associate_by_timestamp
                 + EuRoCLoader/euroc_camera/read_euroc_groundtruth (parser
                 sensor.yaml sin PyYAML, GT cuerpo→cámara), v0.45
                 + read_euroc_imu/euroc_imu_params/read_euroc_state (IMU crudo,
                 ruidos del sensor.yaml y GT de ESTADO con v y sesgos, v1.1))
examples/        01 (2D-2D didáctico autocontenido), 02 (PnP+BA),
                 03 (grafo de poses simulado), 04 (corredor: bucle on/off),
                 05 (datos reales TUM RGB-D, v0.45), 06 (EuRoC MAV, v0.45)
scripts/         make_synthetic_sequence.py (forward: 3 planos; loop:
                 corredor de carteles disjuntos), benchmark_frontends.py,
                 benchmark_tum.py (tabla batch por secuencia TUM, v0.45)
cpp/             CMakeLists.txt (módulo pybind11 vslam_cpp, v0.5) +
                 src/guided_match.cpp (gemelo C++ del matching guiado) +
                 include/vslam/core/frame.hpp (contratos)
tests/           5 archivos de geometría (21 tests) + secuestro (v0.4b) +
                 distorsión/EuRoC (v0.45) + equivalencias gtsam_ba y
                 guided_match_cpp (v0.5), todos con runner __main__
docs/            01 estado del arte, 02 arquitectura, 03 detectores,
                 04 hoja de ruta a v1.0, 05 este documento
```

### 3.4 PnPTracker: flujo y umbrales actuales (todos calibrados con medición)

Flujo por frame: extraer → (INIT: buffer + E con MAGSAC + recoverPose con
`distanceThresh` + triangular + validar con 3ª vista + gauge mediana=1) o
(TRACK: matching vs mapa local por recencia∪covisibilidad → PnP → ¿keyframe?
→ triangular frescos con filtro anti-duplicados → BA local → ¿bucle?).

| Umbral | Valor | Origen |
|---|---|---|
| MIN_INIT_FLOW_PX | 40.0 | barrido: 25→ATE 10.8, 40→6.9 |
| INIT_SURVIVAL_RATIO | 0.5 | pose falsa del twisted pair sobrevive ~13% |
| INIT_VALIDATION_RATIO | 0.7 | tercera vista |
| MIN_PNP_INLIERS / MIN_MAP_MATCHES | 15 / 30 | |
| KF_MIN_GAP / KF_MAX_GAP | 3 / 15 | max-gap: sin él, 90 frames sin KF |
| KF_INLIER_RATIO / KF_MIN_INLIERS | 0.6 / 100 | |
| salud de KF (`healthy`) | ≥ 3×MIN_PNP_INLIERS = 45 | KF con 26 inliers creó 584 pts basura |
| CHEIRALITY_DIST_THRESH | 2000.0 | trampa de recoverPose (§5.1) |
| BA_WINDOW / BA_ITERATIONS | 5 / 6 | anclas = window[:2] (gauge 7 gdl) |
| LOOP_TEMPORAL_GAP/MIN_MATCHES/MIN_INLIERS/COOLDOWN | 60/200/40/40 | calibrados en corredor |
| filtro anti-duplicados | 1.5% de profundidad | solo DESCARTA, nunca asocia |
| covisible_kfs min_shared | 15 | |
| RELOC_AFTER / MIN_MATCHES / MIN_INLIERS (v0.4b) | 3 / 150 / 40 | matches más laxo que LOOP (sin riesgo de bucle sin sentido) |
| GATE_STEP_FACTOR / MIN_SAMPLES / HISTORY (v0.4b) | 6× p95 / 20 / 200 | umbral robusto por percentil, no absoluto |
| culling min_obs / min_age_kfs (v0.4b) | 3 / 3 | min_obs=2 es no-op (todo punto nace con 2 obs) |
| KF_HEALTH_INLIERS (v0.45) | 45 | piso de salud de KF; con matching guiado deja de ser sensible (lección 21) |
| GUIDED_RADIUS_PX / MAX_HAMMING (v0.45) | 15 / 64 | ventana de búsqueda del matching guiado; Hamming máx (ORB-SLAM TH_LOW=50) |
| GBA_ITERATIONS (v0.45) | 50 | BA global OFFLINE (converge lento en mapas grandes: 10 iters no basta, lecciones 26-27) |

### 3.5 Estado administrativo — ¡IMPORTANTE!

- **Git: repo AL DÍA (rama main, working tree limpio, v1.0 committeada).**
  CLAUDE.md, docs/05 y todos los tests están versionados. El repo YA es
  PÚBLICO (github.com/ariel9874/Visual-slam) → recordar a Ariel activar el
  bloque comentado de su perfil de GitHub que lo esperaba. El curso hermano
  vive en github.com/ariel9874/aprende-vslam (público).
- Licencia: **MIT** (LICENSE en la raíz desde v0.9; deps permisivas).
- Pendientes MANUALES de release (verificado 2026-07-17): `python -m build` +
  `twine upload` (vslam-edu NO está aún en PyPI), tag `v1.0.0` + GitHub
  Release, video demo (opcional).
- `data/` y `output/` están en .gitignore (regenerables con los scripts).
- Memoria persistente de sesiones: `C:\Users\ariel\.claude\projects\
  c--Users-ariel-Documents-GitHub-Visual-slam\memory\` (perfil del usuario).

---

## 4. Decisiones de arquitectura vigentes (y sus porqués)

1. **Ejemplo 01 duplica deliberadamente la lógica 2D-2D** (didáctico, un solo
   archivo legible). El benchmark lo importa vía `spec_from_file_location`.
   No "arreglar" esta duplicación.
2. **El mapa local = recencia ∪ covisibilidad** (v0.4a). La covisibilidad se
   calcula desde las observaciones del mapper; el cierre de bucle tiende el
   puente registrando sus pares verificados como observaciones del KF actual.
3. **El grafo del bucle es Sim(3) con el segmento antiguo CONGELADO**
   (≤ KF del bucle): mover la referencia deja la historia emitida en otro
   marco (medido: 6.7→87 cm). El factor de bucle lleva pose (PnP) + escala
   (Umeyama 3D-3D sobre puntos duplicados entre gauges).
4. **La escala vive en el MAPA, no en las poses**: tras correcciones Sim(3)
   las poses se re-normalizan a SE(3) (`update_poses_sim3`, `apply_similarity`).
5. **BA ancla 2 keyframes** (gauge monocular de 7 gdl) y solo optimiza puntos
   con ≥ 2 observaciones en la ventana.
6. **En el costo del BA, las observaciones no proyectables PAGAN** una
   penalización mayor que cualquier residuo físico (~2000 px-equivalentes).

---

## 5. Lecciones medidas (NO re-descubrirlas — están documentadas en el código)

1. `cv2.recoverPose` básico limita quiralidad a 50× el baseline → con
   depth/baseline > 50 los inliers colapsan a ~0. Usar `distanceThresh`.
   [examples/01, tests/test_pose_recovery.py]
2. Escenas cuasi-planas: E tiene la ambigüedad del twisted pair y AMBAS
   soluciones pasan quiralidad; RANSAC/MAGSAC caen erráticamente (2° u 80° de
   error). Defensas: validación con TERCERA vista + tasa de supervivencia de
   triangulación (correcta ~90%, falsa ~13%). [tracker._initialize_step]
3. `mapper or SparsePointMapper()` es un bug: un mapper vacío con `__len__`
   es falsy. Usar `is not None`. [tracker.__init__]
4. El gauge monocular del BA tiene **7 gdl**: fijar 1 cámara deja la escala
   libre (firma: error relativo idéntico en poses y puntos). Fijar 2.
   [bundle_adjustment.py]
5. **Agujeros de costo**: omitir del costo las observaciones de puntos detrás
   de la cámara enseña al optimizador a esconder puntos ahí (volaron a 15 000
   unidades); una penalización tímida (100 px) refina la trampa (esconder
   detrás de la cámara del outlier). Penalizar > residuo físico máximo.
6. Huber NO anula outliers: los degrada a empuje lineal; queda sesgo
   proporcional a la tasa (~0.2% con 10%). El test correcto compara contra
   kernel cuadrático. Y la corrupción de test debe ser de dirección ALEATORIA
   (un sesgo consistente desplaza el mínimo legítimamente).
7. Puntos con UNA observación se deslizan por su rayo en el BA (C_p rango 2):
   registrar la observación de AMBOS extremos de la triangulación y excluir
   del BA los puntos con < 2 obs en ventana. (El BA EMPEORABA hasta esto.)
8. **Nunca crear mapa desde pose incierta**: KF con 26 inliers → 584 puntos
   basura → teleports de 6 unidades. Piso de salud para insertar KFs.
9. Sin intervalo máximo de KFs, una escena siempre co-visible nunca dispara
   el "hambre": 90 frames sin KF (sin BA, sin base de bucles). KF_MAX_GAP=15.
10. La deriva monocular incluye ESCALA (14% medido sin BA): un grafo SE(3) la
    reparte como error de traslación y EMPEORA (35.9→94 cm). Sim(3) (Strasdat)
    reproducido en tests: SE3 no puede, Sim3 sí. [test_pose_graph.py]
11. En el grafo del bucle, congelar TODO el segmento antiguo: con solo el
    nodo 0 fijo, el grafo movía la referencia (6.7→87 cm).
12. **Fusión por proyección envenena**: en textura densa los descriptores
    vecinos están correlacionados → asociaciones cruzadas → obs desplazadas →
    BA colapsa (8→202 cm). Descartar creación es seguro; crear asociaciones no.
13. **Compuerta de movimiento sin relocalización corta por ambos lados**:
    bloquea teletransportes Y recuperaciones (8.4→37.7 cm); con referencia de
    ventana reciente se auto-congela tras una pausa (→202 cm). Retirada;
    re-introducir SOLO junto con relocalización (v0.4b). [comentario en
    tracker._track_step]
14. **La biestabilidad del PnP era síntoma**: la enfermedad era el mapa local
    por recencia, que al re-visitar excluye los puntos originales → duplicados
    desplazados por deriva → dos modos coherentes. La covisibilidad lo hace
    imposible (8.4→2.2 cm) y convierte cada re-visita en cierre implícito.
15. La escena "pared global" hacía todo co-visible y los bucles disparaban a
    mitad de camino sin significado → la secuencia loop usa un CORREDOR de
    carteles disjuntos (`--motion loop`).
16. Trayectorias de ida-y-vuelta se solapan en planta: graficarlas como serie
    temporal (x(t) + error(t) con los bucles marcados). [examples/04]
17. MAGSAC++ > RANSAC clásico para E en escenas cuasi-planas (pero no
    infalible: por eso la validación de 3ª vista).
18. **La compuerta de movimiento SÍ funciona — pero solo con una salida**
    (v0.4b, confirma la lección 13). Rechazar un paso anómalo sin relocalización
    ciega el sistema; con reloc, el rechazo cae a coast, el contador dispara la
    reloc, y ésta decide con verificación GLOBAL a qué pose volver. Medido en el
    test de secuestro: teletransporte 79→3 → 2×GATE-REJECT → RELOC contra KF0 →
    TRACK, recuperado en 2 frames. Umbral por percentil (6× p95), no absoluto.
19. **Todo punto nace con exactamente 2 observaciones** (ambos extremos de su
    triangulación se registran siempre — lección 7). Por eso el culling con
    `min_obs=2` es un NO-OP: el umbral honesto es 3 (un punto que ningún KF
    re-observó tras su par fundacional). Con min_obs=3/min_age=3: **-33.9%** del
    mapa del corredor (2697→1784) sin degradar el ATE (2.2→2.0 cm). Los puntos
    podados siguen accesibles por id (loop/reloc) y una re-observación los REVIVE
    → el puente de covisibilidad (lección 14) no se rompe.
20. **La covisibilidad absorbe saltos moderados** (v0.4b): un secuestro de 30
    frames sobre la vuelta del corredor (79→110) NO fuerza reloc — el mapa local
    ya cubre esa zona y el tracking sigue liso. Para EJERCITAR la reloc en el
    test hay que teletransportar a una zona mapeada pero DISJUNTA en covisibilidad
    (la de salida: 79→3). Buena noticia de robustez; ojo al diseñar tests.
21. **El piso de salud de KF es un trade-off por-secuencia, no una constante**
    (v0.45). En sintético 45 (=3×MIN_PNP_INLIERS) protege de KFs basura (lección
    8). En datos reales el óptimo se INVIERTE según la secuencia y ambos extremos
    fallan: fr2_desk con 45 sufre INANICIÓN de KFs (el tracking sano ronda 20-52
    inliers → no se insertan KFs → el mapa se congela → colapso, 1347 frames
    perdidos; bajar a 25 → 278). fr1_xyz con 25 sufre lo contrario: KFs desde
    poses marginales → puntos basura → tormenta de reloc (ATE 6.9→18.4 cm).
    Diagnóstico: NO fue la compuerta (2 disparos/1700f) ni rotación (GT: 0°/frame
    en el colapso) ni la selección local (el mapa GLOBAL daba 0 inliers) — se
    aisló por-frame que el mapa dejaba de crecer. La cura real no fue una
    constante mágica ni gestión adaptativa de KFs: fue el MATCHING GUIADO
    (lección 24), que sube los inliers y hace la inanición imposible — con él,
    45 y 25 dan el MISMO resultado en fr2_desk (el piso deja de ser sensible).
22. **Matching por descriptor contra un mapa grande real da 0 inliers** (v0.45):
    en fr2_desk (mapa ~10k puntos activos) el matching GLOBAL producía ~100-150
    matches pero 0 inliers geométricos (ORB ambiguo a esa escala); el mapa LOCAL
    (~650 pts) daba los únicos inliers reales. Confirma que el mapa local no es
    solo eficiencia — es CORRECCIÓN; y que a escala real hace falta BoW + matching
    guiado por reproyección (deuda §8). [diagnóstico local-vs-global]
23. **La deriva de escala monocular domina los recorridos largos** (v0.45):
    fr2_desk sin colapso (health=25) aún da ATE ~116 cm porque la escala Umeyama
    deriva 1.82→1.05 entre mitades — ninguna alineación global única encaja (el
    tramo 0-400, que aislado da 3.6 cm, bajo la escala global muestra 48). Los 15
    bucles cerrados son todos LOCALES (gap 60): dan consistencia local pero no
    atan la escala entre segmentos lejanos. El bucle GRANDE (volver al inicio) no
    se cierra porque para entonces el sistema ya está perdido. Es la razón de ser
    del cierre de bucle Sim(3) a escala de sesión, no solo local (trabajo futuro).
    (Con matching guiado, lección 24, fr2_desk ya no colapsa y baja a 21.9 cm; la
    escala sigue siendo el residual dominante — este es el próximo sub-problema.)
24. **El matching GUIADO por reproyección es la palanca de robustez en real**
    (v0.45): predecir la pose (velocidad constante), proyectar el mapa local y
    buscar cada punto solo en una ventana de ~15 px — en vez del matching global
    por descriptor, que a escala real es ambiguo (lección 22). El prior de pose
    restringe la asociación → suben los inliers verdaderos → deja de haber
    inanición de KFs Y mejoran las poses. Medido, un solo cambio:
    - fr2_desk: 104.9 cm / 1347 frames perdidos (COLAPSO) → **21.9 cm / 0
      perdidos**. fr2_xyz: 35/475 → **29/5**. fr1_xyz: 6.9/9 → **4.9/0**.
    - Sintético mejoró también: 02 3.1→2.4, 04 2.0→1.7, secuestro 2.1→1.1 cm.
    - **Deja obsoleta la perilla de la lección 21**: con inliers altos, el piso
      de salud 45 = 25 (fr2_desk idéntico). La cura era el matching, no el umbral.
    - Fallback: si el prior es malo (tras reloc/salto/inicio) el guiado rinde
      poco y cae al matching global. Y destapó un gap: **la reloc no re-anclaba
      el mapa local** en la zona reconocida (seguía en la recencia de antes del
      salto) → tracking no continuaba. Corregido con `_local_ref_kf` (el mapa
      local se ancla al KF relocalizado y su covisibilidad hasta el próximo KF):
      el secuestro pasó de 4 relocs a 1. [tracker._guided_match, _local_kfs]
25. **Evaluar la trayectoria FINAL de keyframes, no la ONLINE** (v0.45 — el
    hallazgo más caro de no ver antes). El ATE se estaba midiendo sobre las
    poses emitidas frame a frame, que se CONGELAN al emitirse: cuando el cierre
    de bucle o el BA global corrigen el mapa en el frame 2700, NO reescriben las
    poses ya reportadas de los frames 0..2699. Resultado: la métrica no veía NADA
    del backend. Medido en fr2_desk: online 21.9 cm vs **trayectoria final de
    keyframes 4.8 cm** (los 21 bucles llevaban corrigiendo todo el tiempo). Es la
    métrica que reporta ORB-SLAM (poses de KFs optimizadas). El "problema de
    deriva de escala" (lección 23) era en gran parte este artefacto de medición.
    [tracker.keyframe_trajectory; driver y benchmark reportan ambas].
26. **BA GLOBAL OFFLINE propaga la escala; online descarrila** (v0.45): el
    grafo de poses Sim(3) del bucle corrige poses pero no re-estima los puntos →
    la escala intermedia queda mal. El BA global (todos los KFs + puntos, 2
    anclas de gauge) sí, y las OBSERVACIONES PUENTE que registra el cierre de
    bucle (KF actual ve puntos del segmento viejo) atan los extremos → reparte la
    corrección por la cadena. PERO probarlo ONLINE (tras cada bucle grande)
    descarrila el tracking: el BA didáctico sobre un mapa grande (~240 KFs)
    sacude el mapa y provoca tormentas de reloc (fr2_xyz 5→346 frames perdidos);
    ni cooldown ni re-anclar el mapa local lo salvan — el problema es correr ese
    BA repetido en caliente. Solución: como solo evaluamos la trayectoria FINAL
    (lección 25), correr UN BA global OFFLINE al terminar. Es el "full BA" offline
    de ORB-SLAM. [tracker.global_bundle_adjustment, lo llama el driver/benchmark
    tras la secuencia]. Medido (final-KF): fr2_desk sin GBA 4.8 → **2.1 cm**;
    fr2_xyz 13 → **0.4 cm**; fr1_xyz **1.8 cm**.
27. **El BA global NO había convergido con 10 iteraciones** (v0.45): en mapas
    grandes el LM-Schur didáctico converge LENTO. El "límite" de fr2_xyz (246 KFs,
    81 bucles) no era estructural — era falta de iteraciones. Barrido medido
    (mismo mapa, GBA offline): 0→13.0, 10→12.0, 25→3.5, **50→0.4**, 100→0.3 cm.
    Con 10 el BA se quedaba a mitad de camino; a 50 converge. Como es offline y
    corre UNA vez, 50 iteraciones son baratas. Moraleja: antes de creer que un
    residual es de fondo (deriva de escala…), verificar que el optimizador
    convergió. [GBA_ITERATIONS = 50]
28. **El envelope de operación del frontend mínimo** (v0.45, 6 secuencias TUM):
    excelente en movimiento MODERADO (fr1_xyz 1.8 / fr2_xyz 0.4 / fr2_desk 2.1 cm,
    0 perdidos), pero dos límites medidos y comprendidos:
    - **fr1_desk / fr1_room se PIERDEN** (560/613 y 988/1362 frames perdidos):
      son handheld con rotación rápida + motion blur → el matching ORB (aun
      guiado) no engancha. Es el caso que pide KLT/directo o features aprendidas
      (SuperPoint, ya instalado) o un IMU — trabajo futuro, no un umbral.
    - **fr3_long_office trackea entero (0 perdidos) pero deriva a 78.5 cm**: NO
      es convergencia (GBA plateau 25→200 iters = 78.5) ni falta de bucles (los
      grandes cierran, saltos 2082-2130). Es SISTEMÁTICO: la calibración fr3
      (dist=0, la "ROS default" de TUM) sesga la geometría, y/o un recorrido
      largo complejo cuyos bucles solo atan el final. Verificar si TUM publica
      distorsión real para fr3; si no, es límite del modelo pinhole ideal.
29. **SuperPoint + LightGlue RESCATAN las fr1 handheld** (v0.45): el frontend
    aprendido (learned.py, verificado por fin en GPU — RTX 4070) transforma
    fr1_desk de FALLO a funcional. Medido (ATE-KF / frames perdidos, GBA incl.):
    - ORB+ratio:            2.2* / **560**   (*sobre 7 KFs, sin sentido)
    - SuperPoint+ratio:     3.2  / 168       (56 KFs → trackea el grueso)
    - SuperPoint+LightGlue: 4.9  / **140**   (menos perdidos + 6 bucles vs 2)
    Los descriptores de SuperPoint (256-D float, robustos a blur/rotación) hacen
    que el matching guiado enganche (~330 matches/frame) donde ORB fallaba, y
    LightGlue (atención espacial 2D-2D) añade robustez en init/loop/reloc.
    Integración: LightGlue solo empareja 2D-2D (necesita kps de ambos lados) →
    el tracker usa `self.matcher` (LightGlue) para init/KF-a-KF/loop-reloc y un
    `_desc_matcher` de ratio para el matching 3D-2D contra el mapa y la 3ª vista
    (sin kps del mapa). Con ORB+ratio, `_desc_matcher` es el mismo → sin cambio.
    Coste: SuperPoint ~224 ms/frame + LightGlue ~144 ms (GPU). Uso:
    `examples/05 --detector superpoint --matcher lightglue`.
    **Los 140 perdidos NO son de umbral** (medido, contra la intuición): cuando
    SuperPoint trackea, los inliers son ALTOS (p10=91 ≫ KF_HEALTH=45), así que
    recalibrar no rescata — y se aisló que los 140 son UN episodio contiguo
    (frames 200-340; el resto 0 perdidos), no rotación (GT: 1.0°/frame ahí, MENOS
    que en la zona sana). Es estructural: tracking se pierde al entrar a una zona,
    que —ya perdido— no se mapea y no admite reloc hasta volver a territorio
    conocido. Pide robustez de movimiento (KLT/IMU) o re-mapeo, no un umbral.
    Moraleja (otra vez): medir antes de recalibrar — el lever "obvio" no lo era.
30. **Perfilar refuta la intuición de rendimiento** (v0.5): el criterio 30 fps
    exigía saber DÓNDE se va el tiempo. docs/04 apostaba por extracción+matching+
    PnP; el perfilador dijo otra cosa (fr2_desk/ORB, 4.3 fps): **BA local 57%**
    (pico 2 s/KF), **matching guiado 37%** (TRACK 93 ms), y cv2 (ORB/BF/PnP)
    solo **8%** — porque cv2 YA es C nativo; lo caro es el código Python/NumPy
    (BA con 1.67M evals de residuo/jacobiano; guiado con su bucle por punto).
    Portar cv2 habría sido esfuerzo tirado. Primer port según el dato: el BA →
    **adaptador GTSAM** (mismo problema, test de equivalencia exacto): KF-frame
    1382→385 ms (3.6×), fps 5.8→9.5, mismo ATE. Regla 3 del repo en acción:
    solo se reescribe lo que el perfilador señala, y la gemela pasa los tests.
31. **El primer módulo C++ (pybind11) pone el TRACKING en tiempo real** (v0.5):
    `cpp/src/guided_match.cpp` (módulo `vslam_cpp`) es el gemelo EXACTO del
    matching guiado — misma matemática, y la misma semántica de desempate que
    np.argmin (ante empate de Hamming gana el índice menor; sin eso la
    equivalencia par a par falla). Verificado por tests/test_guided_match_cpp.py
    (5 escenas, uint8/Hamming y float32/L2, casos límite). Medido en fr2_desk
    (con BA GTSAM): frames TRACK **110→29 ms** (3.7×, ya bajo el presupuesto de
    33 ms = 30 fps), fps global 7.6→**18.7**, ATE idéntico (1.6 cm). El tracker
    usa la ruta C++ AUTOMÁTICAMENTE si el .pyd existe (`self.use_cpp`; False
    fuerza la referencia Python). Compilación: cpp/CMakeLists.txt (VS Build
    Tools/MSVC en Windows; el .pyd queda en la raíz del repo, gitignored).
    El cuello restante son los frames KF (~380 ms: BA GTSAM síncrono) → iSAM2
    incremental o BA en hilo aparte es el siguiente lever hacia 30 fps sostenidos.
32. **iSAM2 hace el BA incremental casi gratis — y desnuda al VERDADERO cuello
    del keyframe** (v0.5). Backend `ISAM2LocalBA` (`ba_backend="isam2"`): un solo
    grafo de sesión; cada KF solo re-linealiza las cliques afectadas del árbol
    de Bayes. Integración con 4 reglas medidas: (a) los puntos ENTRAN con ≥2 obs
    (con 1, IndeterminantLinearSystemException — lección 19 en el backend; buffer
    de pendientes); (b) consumo incremental por CURSORES sobre mapper._obs crudo
    (la vista filtrada encoge con el culling y rompería índices); (c) gauge por
    priors en 2 poses por época; (d) RESET tras cada cierre de bucle (la Sim(3)
    externa invalida la linealización) con re-siembra anclada por priors
    (historia congelada, lección 11). Medido: corredor KF 2060→82 ms (25× vs
    numpy), 49 fps; fr2_desk paridad de ATE (1.5 cm), 0 fallos con 2 bucles.
    PERO en fr2_desk el KF-frame apenas bajó (330→320 ms): el perfil muestra que
    el BA ya era marginal (gtsam.update: 34 ms/KF) — el cuello real es el
    RECONOCIMIENTO DE LUGAR del bucle (knnMatch contra TODA la _kf_db: ~9.4 s de
    17.6 s del _insert_keyframe) + culling/filtros. Moraleja: optimizar el BA
    otra vez habría sido inútil; el siguiente lever es el HILO DE MAPEO
    (bucle+BA+culling fuera del hilo de tracking, arquitectura ORB-SLAM) y/o
    BoW para el reconocimiento de lugar (deuda de §8 desde v0.35).
33. **El HILO DE MAPEO compra CONSISTENCIA de latencia, no throughput** (v0.5,
    `async_mapping=True`): el bloque pesado del KF (BA + bucle + culling) va a
    un worker; el tracking solo triangula e inserta. Medido en fr2_desk (isam2 +
    C++ guiado): **p99 400→126 ms, max 677→197 ms** — desaparecen los picos de
    KF, que es el criterio real de tiempo real — con paridad de ATE (1.6 cm),
    mismos bucles y 0 fallos del worker. El precio: la MEDIANA sube (24.6→34.6
    ms) por contención de GIL/CPU (gtsam no suelta el GIL en update/optimize;
    cv2 y vslam_cpp sí — a este le añadimos gil_scoped_release). Diseño clave:
    (a) lock SOLO en las secciones de lectura/escritura del mapa, cómputo
    pesado fuera; (b) el worker NUNCA toca T_w_c (heredar una pose vieja sería
    teletransporte); las correcciones del bucle llegan como DELTA pendiente
    (T_nuevo·T_viejo⁻¹, rígido) que el tracking aplica al inicio del siguiente
    frame — el mismo patrón que reloc/GBA; (c) el job lleva SU kf_id/mp (el
    _kf del tracker ya avanzó); (d) wait_mapping() drena antes de leer
    resultados (el GBA offline lo llama solo). [tracker._mapping_worker;
    test_async_mapping.py]
34. **BoW cierra el criterio de v0.5: 46.7 fps** (place_recognition.py). El
    reconocimiento de lugar por fuerza bruta (knnMatch contra TODA la _kf_db)
    era el coste dominante del keyframe (lección 32); el hilo lo escondía pero
    pagaba GIL. BoW lo ELIMINA: vocabulario de 512 palabras entrenado EN SESIÓN
    (k-medias en espacio de Hamming — el centroide binario es el VOTO DE
    MAYORÍA por bit, la mediana coordenada a coordenada; ~50 ms una vez con los
    primeros 5 KFs), cuantización por BFMatcher (C++, ~2 ms), índice invertido
    + tf·idf coseno (Sivic & Zisserman 2003). Query: **2.7 ms**, recall top-2
    1.00 (test sintético); solo TOP-5 candidatos pagan verificación geométrica.
    Medido en fr2_desk (isam2 + C++ guiado): sync 23.9→**43.7 fps**; async
    25.7→**46.7 fps**, mediana 34.6→17.0 ms (el worker dejó de acaparar el
    GIL), p99 73 ms, ATE-KF **1.4 cm** (paridad). Con esto, **el criterio de
    v0.5 (30 fps a 640×480 en CPU, mismo ATE) queda CUMPLIDO en fr2_desk**:
    4.3→9.5→18.7→25.7→46.7 fps por perfilado dirigido, nunca a ciegas.
    Matiz de rigor del módulo: la norma tf·idf del documento usa el idf de
    TODAS sus palabras (no solo las del query) o el coseno queda mal normalizado.
    use_bow=False restaura la fuerza bruta de referencia. [lección 32→33→34:
    medir, mover, eliminar]
35. **El grupo del cierre de bucle depende de QUIÉN fija la escala** (v0.6,
    RGB-D). El primer criterio RGB-D falló espectacularmente: fr2_xyz completa
    daba ATE MÉTRICO 22.1 cm con escala de similitud **2.09** (el humo de 200
    frames, sin bucles aún, daba 0.7 cm/0.96). Diagnóstico en dos patas:
    (a) ABLACIÓN — sin cierre de bucle, la misma secuencia da **1.1 cm /
    escala 0.977** (la escala métrica NO deriva: por ventanas de 400 frames se
    queda en 0.90-1.09 los 3669 frames); (b) MECANISMO — espía sobre el
    Umeyama del bucle: los 77 bucles empiezan midiendo s_rel ≈ 1 (1.001,
    0.971...) y DEGENERAN hasta 0.03, porque cada corrección Sim(3) re-escala
    el mapa viejo mientras los puntos nuevos siguen naciendo métricos desde el
    sensor → el siguiente bucle mide la discrepancia que el anterior CREÓ y
    la "corrige" otra vez: composición de error (escala por ventanas 0.92 →
    3.29, escalón a escalón con los bucles). La moraleja de fondo: en
    monocular la escala es GAUGE y el bucle debe medirla y redistribuirla
    (Sim(3), Strasdat — lección de v0.4, sigue siendo correcta AHÍ); en RGB-D
    la escala es una MEDICIÓN y no se negocia — el bucle métrico va en
    **SE(3)** (s_rel = 1, información 6×6), que es exactamente la decisión de
    ORB-SLAM2. Fix: flag `_metric` (lo enciende la init RGB-D) elige el grupo
    en `_try_close_loop`; `update_poses_sim3` con similitudes de s=1 ya es
    rígido solo (det(R)^⅓ = 1), el resto del pipeline no se toca. Resultado:
    fr2_xyz completa **4.7 cm MÉTRICO / escala 1.036** con los mismos 80
    bucles, 0 perdidos — criterio (<5 cm) CUMPLIDO. Test formal:
    test_metric_loop_is_rigid (bucle fabricado con nube duplicada a 1.2× de
    cebo: la rama monocular DEBE re-escalar; la métrica NO mueve un punto).
    Matiz honesto medido: en fr2_xyz sin bucles queda 1.1 cm — con deriva tan
    pequeña, el bucle corrige menos de lo que ensucia (bridge obs + grafo
    sobre 246 KFs); posible refinamiento futuro (cerrar solo con deriva
    detectada), no se persigue ahora.

36. **El mapa métrico necesita que el BA MIDA metros — y un resultado
    bit-idéntico es un bug, no una meseta** (v0.6 hito 2, la lección que cerró
    el criterio). Dos historias entrelazadas. (a) EL RESIDUO: el BA de
    reproyección solo ve píxeles — re-teje la estructura pero no tiene de
    dónde corregir la deriva métrica (fr1_desk estancado en 6.7 cm con error
    REPARTIDO: p50 7.8 cm, el peor 10% de frames solo aporta el 38% del error
    cuadrático — estructura global, no un episodio). El fix es el de ORB-SLAM2:
    la profundidad entra al BA como cámara derecha VIRTUAL, u_R = u − bf/z
    (`STEREO_BF = 40` ≈ fx·b del Kinect), residuo [u, v, u_R] — todo en
    píxeles (misma Huber, mismo Schur) y el peso de la profundidad decae con
    z² solo, como el ruido del sensor. Test del mecanismo:
    test_ba_depth_residual_makes_scale_observable (con UNA cámara fija la
    escala es espacio nulo del BA 2D — el par nulo/observable discrimina).
    (b) EL BUG: la primera medición con el residuo dio fr1_desk 6.7 cm
    BIT-IDÉNTICO al baseline (mismos bucles, mismos decimales) — en un sistema
    caótico eso no es "mejora nula", es "el código nuevo no se ejecuta". La
    sonda lo confirmó: `_metric=False`. El stream depth de fr1_desk arranca
    ~6 frames tarde (asociación rgb↔depth sin pareja → depth=None), el tracker
    caía a la init MONOCULAR y nacía un mapa MIXTO: init a escala gauge +
    puntos posteriores desde profundidad en metros — con la escala 1.008 de
    PURA CASUALIDAD (la mediana del escritorio ≈ 1 m ≈ gauge mediana=1), y ni
    bucle SE(3) ni residuo de profundidad activos. Fix doble: el driver espera
    al primer frame CON depth para inicializar (examples/05), e invariante en
    el tracker: puntos desde profundidad SOLO en mapa métrico. (c) ATRIBUCIÓN
    (ablación con el MISMO driver, STEREO_BF=0): init métrica sola = 12.8 cm
    con 244 perdidos y reloc — cae en la cuenca colapsada del episodio 200-340
    (lección 28); con el residuo = **2.8 cm, 0 perdidos**: el anclaje métrico
    por observación es lo que CRUZA el episodio, no una perilla. Resultado
    final: **fr1_desk 2.8 cm (escala 1.005) y fr2_xyz 1.5 cm (escala 0.96,
    antes 4.7) — criterio v0.6 (<5 cm métrico en ambas) CUMPLIDO**. Deuda
    anotada: los adaptadores GTSAM (batch e iSAM2) proyectan con uv[:2] —
    paridad monocular intacta, factor estéreo pendiente para --fast RGB-D.

37. **La cámara derecha VIRTUAL de RGB-D se vuelve REAL sin tocar el BA**
    (v0.6, estéreo EuRoC). El residuo estéreo del hito 2 estaba pensado para
    esto: en RGB-D sintetizábamos u_R = u − bf/z desde un sensor de
    profundidad; en estéreo, u_R se MIDE (u_R = u_L − d, el match en la imagen
    derecha). Mismo residuo [u, v, u_R], misma ruta métrica del tracker — solo
    cambia de dónde sale z. Piezas nuevas, todas en `vslam/io/dataset.py`:
    (a) `EuRoCStereoRig` rectifica el par (cv2.stereoRectify desde los dos
    sensor.yaml + la pose relativa cam0←cam1) → rectas epipolares = filas
    (búsqueda 1D), cámara izquierda pinhole SIN distorsión, y bf = −P2[0,3]
    (verificado en V1_01: baseline 11.01 cm, bf 48.0, los valores de EuRoC).
    (b) `EuRoCStereoLoader` triangula profundidad densa por disparidad
    (StereoSGBM) y la entrega con la MISMA firma `(ts, gray, depth)` que
    TUMRGBDLoader → el tracker RGB-D métrico funciona sin cambios. La belleza
    del truco: el ruido de la profundidad estéreo crece con z² (∂z/∂d = −bf/d²)
    y el peso del residuo u_R decae con z² — se cancelan, igual que en RGB-D.
    Resultado en **V1_01_easy (ESTÉREO, trayectoria final de KFs): 6.9 cm rmse,
    escala similitud 1.002** (234 KFs, 27 bucles SE(3), 34 perdidos, 1 reloc) —
    métrico de verdad SIN convención de gauge, sobre un dron rápido 6-DoF. El
    ATE online es 63.8 cm (excursiones per-frame en el vuelo agresivo + coast);
    la métrica del sistema es la final de KFs (lección 25). Tests sin el dataset
    (~1.1 GB): tests/test_stereo.py — geometría del rig (bf = fx·b, rectificar
    un par ya rectificado ≈ identidad) y profundidad por disparidad de un plano
    fronto-paralelo (SGBM recupera d → Z). NOTA de datos: el host oficial de
    EuRoC (robotics.ethz.ch) estaba caído; V1_01_easy se bajó del mirror ASL
    `pepijn223/euroc-mirror` en HuggingFace (formato .zip idéntico).

38. **El residuo de profundidad también en el stack RÁPIDO (GTSAM)** (v0.6,
    cierre de la deuda). El residuo métrico vivía solo en la referencia NumPy;
    los adaptadores GTSAM (batch e iSAM2) proyectaban con uv[:2] → bajo `--fast`
    el BA local no anclaba la escala (solo el mapa nacía métrico y el GBA la
    ataba al final). Portado con las piezas nativas de GTSAM: la profundidad es
    un `GenericStereoFactor3D` sobre `Cal3_S2Stereo(baseline = bf/fx)`, con la
    medición `StereoPoint2(u_L, u_R, v)` — el mismo residuo [u, v, u_R], ruido
    3D + Huber, resuelto por el motor de C++. Las obs (2,) o con u_R = NaN caen
    al factor monocular (mismo criterio que la referencia). En iSAM2 la
    calibración estéreo se construye perezosamente al llegar el primer bf (el
    tracker lo pasa por keyframe; es constante por secuencia). Test:
    test_gtsam_stereo_factor_makes_scale_observable (el par nulo/observable —
    con UNA cámara fija el BA 2D deja el 15% de escala, el estéreo lo recupera —
    y ADEMÁS GTSAM ≡ NumPy). Validado en tiempo real (`--fast --depth`, ruta
    iSAM2): **fr2_xyz 1.4 cm (escala 0.963) y fr1_desk 2.5 cm (escala 1.005)**,
    0 perdidos — paridad con la referencia NumPy (1.5 / 2.8) a 30+ fps. La deuda
    del §8 queda saldada para la ruta de rendimiento.

39. **El mapa denso es "renderiza y compara" — y la referencia densa es
    O(N·H·W)** (v0.7 hito 1-2, `GaussianSplattingMapper`). El SLAM geométrico ya
    da la ESTRUCTURA (poses métricas + nube dispersa, v0.6); 3DGS solo la vuelve
    foto-realista, sin tocar frontend ni backend (la tesis de docs/01 §3.2).
    (a) RASTERIZADOR (gaussian_render.py, PyTorch puro, la referencia legible
    como la NumPy del BA): proyección → covarianza 2D por EWA (Σ' = J·W·Σ·Wᵀ·Jᵀ)
    → α-blending front-to-back por transmitancia (producto acumulado exclusivo).
    Diferenciable de punta a punta; test por diferencias finitas del gradiente
    respecto a la media + sobreajuste de una vista a PSNR > 30 dB. (b) MAPPER
    (gaussian.py, detrás de MapperBase): `add_points` siembra gaussianas desde
    la nube dispersa (media = punto, color = muestra de la imagen ancla),
    `integrate_keyframe` guarda la vista (barato, no bloquea — contrato de
    base.py), `optimize` hace descenso de gradiente contra los keyframes, y
    `update_poses` re-ancla RÍGIDAMENTE cada submapa por el delta de su keyframe
    (D = T'·T⁻¹: la media va con R_D·μ+t_D, la orientación de la covarianza rota
    con R_D — la generalización densa del re-anclaje de la nube dispersa). Test:
    multi-vista con geometría correcta + color desconocido → PSNR medio > 30 dB,
    y update_poses rígido exacto. LÍMITE MEDIDO: el rasterizador denso guarda un
    tensor (N, H, W, 2) → inviable a 640×480 con miles de gaussianas (OOM); el
    ejemplo 07 renderiza a resolución reducida (`--scale`). La ruta full-res es
    la gemela gsplat (tiles + CUDA), pendiente como lo fue el C++ del matching o
    el GTSAM del BA (regla 3). Tests: test_gaussian_render.py (3),
    test_gaussian_mapper.py (2).

40. **gsplat NO enlaza en Windows (mangling nvcc↔MSVC) — y la vía corta es
    DOCKER; de paso, la gemela destapó un bug de MEDIO PÍXEL en nuestra
    referencia** (v0.7 hito 4). Tres verdades medidas en una tarde:
    (a) EL MURO: los 30 kernels CUDA de gsplat COMPILAN en Windows (CUDA 12.4
    por conda sin admin + VS Build Tools), pero el LINK falla con 38 LNK2019:
    el frontend host de nvcc (cudafe++) y cl.exe divergen al comprimir las
    back-references del mangling en plantillas de ~28 argumentos (mismo símbolo
    legible, `...V34@000000@Z` vs `...V34@V12@33333@Z`). NO depende del toolset
    (probado MSVC 14.44 y 14.39) ni de flags (/Zc:preprocessor arregló las
    cabeceras CCCL pero no esto). En Linux/gcc ese mangling no existe: en el
    contenedor (docker/Dockerfile.gsplat, imagen pytorch 2.6-cuda12.4-devel,
    idea de Ariel) gsplat compila al primer import (88 s, cacheado en el volumen
    gsplat-cache) y corre con la GPU (--gpus all, RTX 4070 visible). Además el
    rasterizador POR TILES (gaussian_render_tiled.py, PyTorch puro, culling a
    3.5σ + blending por tile con orden global) rompe el techo de MEMORIA de la
    referencia (lección 39) manteniendo equivalencia >40 dB — pero NO el de
    velocidad: 516 ms/iter con 15k gaussianas a 213×160, mientras gsplat da
    ~15 ms/iter con 300k a 640×480. El kernel CUDA no es una optimización: es
    la pieza HABILITANTE del mapa denso. (b) EL BUG que encontró el test de
    equivalencia: daba 25 dB (esperado >45) con el error concentrado en el
    NÚCLEO de las gaussianas, no en las colas del recorte. Con UNA gaussiana,
    el pico de gsplat = exp(−0.5·0.5/σ²)·el nuestro en todo el barrido de
    escalas — la firma exacta de MEDIO PÍXEL: nuestra referencia muestreaba la
    rejilla en la esquina entera (i, j) y la convención estándar (3DGS, gsplat,
    OpenGL) es el CENTRO (i+0.5, j+0.5). Fix de una línea (+0.5) en referencia
    y tiled → 60 dB de equivalencia (max diff 0.009). La gemela rápida volvió a
    auditar a la referencia legible, como GTSAM↔NumPy en el BA; queda
    test_pixel_center_convention para cazar la regresión sin gsplat. Tests:
    test_gaussian_tiled.py (3), test_gaussian_gsplat.py (4, corren en el
    contenedor).

41. **El techo del PSNR en datos reales es la CONSISTENCIA FOTOMÉTRICA, no la
    capacidad del mapa — y el criterio de v0.7 se recalibró a paridad SOTA**
    (v0.7 hito 4, fr1/desk 640×480, gsplat en Docker). La cadena de ablaciones,
    cada hipótesis medida y las tres primeras FALSADAS:
    | experimento | PSNR |
    |---|---|
    | 300k gaussianas full-res, siembra ingenua (escala fija 3 cm) | 15.5 dB |
    | + escala por punto step·z/fx (la huella de la celda; mediana 0.4 cm) | 15.8 |
    | + delta SE(3) POR KEYFRAME + ganancia/sesgo de exposición | 16.4 |
    | + 30k iters con DECAY del lr de medias (×0.01, 3DGS original) | **20.9** |
    | + densificación/poda (clone/split al 5% de mayor gradiente; 300k→493k) | 21.0 |
    Conclusiones: (a) subir 20× las gaussianas y 4× la resolución NO movió el
    número (15.0→15.5): la capacidad no era el cuello — dos veces medido (la
    densificación tampoco: +0.1). (b) El factor dominante fue el PRESUPUESTO +
    SCHEDULE de optimización (16.4→20.9): sin decay, el paso fijo de las medias
    es un jitter perpetuo que se paga como blur. (c) El refinamiento de poses
    es obligatorio en real: el ATE de ~cm del SLAM son PÍXELES a 1 m (1 cm ≈
    5 px con fx≈520) y la fusión fotométrica exige sub-píxel — es el lazo de
    MonoGS/SplaTAM: el mapa denso devuelve corrección a las poses (T' = T·exp(ξ),
    ξ ∈ se(3), horneado en el keyframe al terminar). La exposición afín por
    keyframe compensa el auto-exposure de TUM. (d) El RESIDUO es del dataset:
    fr1/desk es handheld rápido con motion blur y rolling shutter — los propios
    targets están emborronados distinto por vista. La dispersión por keyframe
    (min 17.1 / mediana 20.9 / max 29.9) es el diagnóstico: en las vistas bien
    condicionadas el mapa YA toca 30 dB. (e) CRITERIO RECALIBRADO (decisión de
    Ariel con la literatura delante): los >30 dB de docs/04 son territorio de
    datasets sintéticos (Replica); en fr1/desk lo publicado es Photo-SLAM ~21,
    SplaTAM ~22, MonoGS RGB-D ~23-25. Criterio v0.7: **PSNR ≥ 21 dB en fr1/desk
    (paridad SOTA) — CUMPLIDO: 21.0 dB** (12.5 de siembra), a 20 ms/iter con
    ~500k gaussianas. Margen extra (SSIM, ponderar KFs por blur) documentado
    como opcional, no bloqueante.

42. **En Python el tercer hilo de ORB-SLAM es un PROCESO — el GIL convierte el
    "hilo de mapeo denso" en un impuesto del 78% al tracking** (v0.7 hito 5,
    examples/08, fr1/desk en el contenedor). `DenseMappingThread/Process`
    (dense_thread.py): el tracking solo ENCOLA el keyframe (submit = una copia,
    ~µs — test lo exige <5 ms) y el worker integra + siembra + optimiza por
    CHUNKS con el presupuesto sobrante. La medición del criterio (ON vs OFF,
    mismos 596 frames y 80 KFs en todos los casos):
    | modo | mediana tracking | iters de mapa en vivo |
    |---|---|---|
    | sin mapper (baseline) | 36.1 ms | — |
    | HILO | 64.2 ms (+78%) | 6800 |
    | PROCESO | 46.6 ms (+29%) | 8400 |
    | proceso + torch 1 core en el hijo | **45.0 ms (+25%)** | 8500 |
    (a) El hilo NO sirve aunque el trabajo viva en la GPU: cada iter hace
    cientos de llamadas Python→torch que retienen el GIL y el tracking
    (Python+numpy) lo pierde la mitad del tiempo — ni set_num_threads(1) ni
    dormir entre chunks lo arreglan (medido). El PROCESO (mp spawn, torch/CUDA
    solo en el hijo, keyframes de ~77 KB por mp.Queue) recupera casi todo, y
    además el mapa recibe MÁS presupuesto (8400 vs 6800 iters: el worker
    tampoco es estrangulado). Es lo que hace MonoGS; base.py ya decía
    "hilo/proceso". El +25% residual es contención de CPU/memoria del portátil
    (driver CUDA, WSL2), no GIL. (b) CARRERA real cazada: update_poses desde el
    driver muta los tensores del mapa en medio de un backward del worker →
    acceso CUDA ilegal. Fix estructural: las correcciones de pose VIAJAN POR LA
    COLA del worker (serializadas entre chunks) — sin locks largos, sin carrera
    por construcción. (c) CRITERIO v0.7 (2ª mitad) CUMPLIDO: el tracking
    procesa LOS MISMOS frames con el mapper ON (596/596, 80/80 KFs integrados,
    0 fallos), con presupuesto medido: +9 ms de mediana. Tests:
    test_dense_thread.py (3: submit barato, consumo+optimización, proxy de
    poses; smoke del proceso). Ejemplo: examples/08_live_dense_mapping.py.

43. **La cáscara ROS 2 no contamina — y la arquitectura de v0.7 mapea 1:1 a
    nodos** (v0.8 hitos 1-4, contenedor vslam-ros). `vslam_msgs` (Keyframe con
    imagen+depth+pose ÓPTICA+K escalada; PoseGraphEdge; TrackingState) +
    `vslam_ros` con 4 nodos rclpy FINOS: dataset_node (TUM → cámara simulada,
    imagen CRUDA + CameraInfo, como un driver real), frontend_node (arma el
    PnPTracker desde la CameraInfo, publica odom + TF odom→base_link +
    keyframes), backend_node (Path + TF map→odom) y mapper_node (keyframes →
    nube retro-proyectada → PointCloud2). Claves: (a) la CONVERSIÓN DE EJES
    óptico↔REP-103 vive SOLO en conversions.py y es por CONJUGACIÓN
    (T_ros = R̃·T_opt·R̃⁻¹ — rotar un solo lado deja el mundo inconsistente y
    RViz muestra la trayectoria "de lado"); el núcleo no importa ROS (regla 4
    verificada). (b) El mapper-como-consumidor del hito 5 de v0.7 ES un nodo:
    el tópico sustituye a la mp.Queue y la lección 42 (proceso, no hilo) viene
    de serie — cada nodo ROS es un proceso. (c) backend_node materializa
    REP-105: T_map_odom = T_map_kf·T_odom_kf⁻¹, la corrección a saltos; el
    control consume odom (suave), la navegación map (consistente). MEDIDO
    (test/smoke_pipeline.py, 35 s a 10 Hz): 148 odom, keyframes→Path(3) +
    PointCloud2 (4901 pts), árbol TF map→odom→base_link completo, metric=True.
    RViz vía WSLg (Docker Desktop: mounts /run/desktop/mnt/host/wslg +
    LIBGL_ALWAYS_SOFTWARE=1) confirmado visualmente — CRITERIO de v0.8
    cumplido. Gotcha de bash que costó una hora: `source X && cmd &` pone en
    background LA LISTA ENTERA (el shell nunca sourceó el overlay y el CLI
    decía "message type invalid" — parecía un bug de typesupport y era un `;`).
    Restante v0.8: lifecycle nodes, demo con rosbag de EuRoC, webcam.

44. **Lifecycle: los CONSUMIDORES se activan antes que el productor — y EuRoC
    entra por la misma puerta que TUM** (v0.8 hitos 5-6). Los tres nodos vslam
    son LifecycleNode (configure arma pub/sub, activate/deactivate PAUSAN el
    procesamiento con los drivers vivos, cleanup destruye el tracker → el
    siguiente ciclo es un SLAM nuevo); verificado: deactivate → 0 msgs de
    odom, activate → vuelve a fluir. La LECCIÓN del bringup: activarlos en
    orden frontend→backend→mapper pierde los primeros keyframes (medido:
    map=0, path=1 — el QoS reliable protege el TRANSPORTE, no al suscriptor
    tardío); el orden correcto es consumidores→productor (mapper, backend,
    frontend) — smoke verde: 18 nubes/9958 pts, Path completo. EuRoC estéreo:
    dataset_node con param `dataset:=euroc` reutiliza EuRoCStereoLoader (misma
    firma que RGB-D, lección 37); el bf del rig NO viaja en CameraInfo → va
    por parámetro al frontend (euroc_demo.launch). Smoke EuRoC: metric=True,
    bf=48.02, 9 KFs→Path, 41k puntos. WEBCAM (pendiente, decisión anotada):
    Docker Desktop/Windows no expone /dev/video0 — requiere usbipd-win para
    adjuntar la cámara USB a WSL2; además el frontend hoy exige depth
    sincronizada (la webcam sería MONOCULAR: rama gray-only del tracker).

45. **Config declarativa sin duplicar la verdad + reset de mapa — y el frame
    CIEGO que nunca habíamos visto** (v0.9 hitos 1-2). (a) CONFIG
    (vslam/config.py): las constantes de clase SIGUEN siendo la documentación
    (cada umbral con su porqué medido); el YAML/JSON solo las sobreescribe POR
    INSTANCIA al final del __init__ (`PnPTracker(..., config=load_config(p))`).
    Config vacía = bit-idéntico (garantía de no-regresión); un typo falla en
    el ARRANQUE listando las claves válidas; la plantilla se GENERA desde las
    clases (`python -m vslam.config`, 45 perillas: tracker+isam2+pose_graph) —
    una sola fuente de verdad, sin YAML que envejece. Tests: test_config (5).
    (b) DEGRADACIÓN ELEGANTE: tras LOST_RESET_AFTER=90 frames en coast sin
    reloc, el mapa es irrecuperable → `_reset_map()` archiva la trayectoria
    (keyframe_trajectory devuelve archivadas+actual, en orden), vacía TODO
    (mapper, BoW, iSAM2, init) y la siguiente vista re-inicializa una sesión
    NUEVA anclada en la pose extrapolada — el init RGB-D ahora ancla en
    self.T_w_c (= I en el arranque: bit-idéntico; = pose coasted tras reset:
    continuidad sin salto). Las sesiones NO se fusionan (multi-mapa/Atlas
    fuera de 1.0, docs/04). (c) El test del APAGÓN (frames negros) cazó un bug
    real preexistente: `_guided_match` con CERO keypoints crashea (np.array de
    lista vacía es (0,) y no broadcastea contra (2,)) — nunca visto porque TUM
    siempre tiene esquinas; una oclusión total en un robot real habría tirado
    el tracker. Fix: retorno temprano. REGRESIÓN verificada tras tocar init y
    matching: fr1_desk --depth 2.8 cm rmse / escala 1.005 / 81 KFs / 0
    perdidos — los números EXACTOS de referencia. Tests: test_map_reset (1),
    test_rgbd (5) verdes.

46. **La concurrencia se audita con ÉPOCAS — y la API se congela pequeña**
    (v0.9 hitos 3-5). (a) El test de estrés (test_concurrency: async_mapping +
    lectores en caliente estilo nodo ROS + apagón con reset en pleno vuelo)
    cazó la última carrera: un job del worker de mapeo puede pertenecer a una
    SESIÓN MUERTA (reset de la lección 45). Fix estructural: época de mapa —
    `_map_epoch` viaja con cada job; el worker DESCARTA jobs de épocas viejas
    y una excepción con la época cambiada a mitad del job es la muerte
    esperada de la sesión, no un fallo (map_failures ya no la cuenta). El
    mismo patrón del epoch-check de los sistemas de colas. (b) docs/06 (mapa
    denso 3DGS): la visita guiada de las lecciones 39-42 con la cadena de
    ablaciones y los comandos; el "docs/05 (RGB-D)" de la hoja de ruta quedó
    cubierto por las lecciones 35-38 de ESTE documento (la numeración de docs
    derivó: 05 es el traspaso). (c) API FREEZE (vslam/__init__.py, v0.9.0):
    16 nombres públicos en __all__ — contratos (PinholeCamera/Frame/
    Trajectory), el tracker y sus factories, MapperBase/SparsePointMapper,
    loaders y config. Los pesados (torch/gtsam) quedan FUERA del import raíz
    (perezosos, se importan de sus módulos): `import vslam` no arrastra GPU ni
    C++ — verificado. Lo no listado puede cambiar entre minors.

47. **La preintegración se valida con el GT de ESTADO — y la gemela GTSAM no
    es bit-exacta porque es OTRA formulación** (v1.1 hito 1,
    imu_preintegration.py). (a) `predict()` es un REORDENAMIENTO EXACTO del
    dead-reckoning: la gravedad (g·Δt, ½·g·Δt²) y el arrastre de v_i salen de
    la suma telescópica de la integración — no hay aproximación; el test lo
    exige a 1e-12 y el residuo del futuro factor es 0 en el estado verdadero.
    Lo aproximado es SOLO la corrección de sesgo (1er orden: error medido
    2e-5 con |δb|=1e-3, crece ×100 al ×10 — 2º orden como debe). (b) La wheel
    de GTSAM (conda 4.2.2) usa preintegración TANGENTE, no la de variedad del
    paper de Forster: la equivalencia es de 2º orden (7e-5 rad / 1.2e-4 m tras
    2 s con |Δv|~20 m/s), NO bit-exacta — el test usa tolerancias con ~10× de
    margen sobre lo medido, no igualdad. Tres trampas de su API documentadas:
    `preintMeasCov` ordena (θ, p, v) (nosotros [φ, v, p]),
    `ConstantBias(ACELERÓMETRO, gyro)` — al revés de lo natural — e
    `integrateMeasurement(acc, omega, dt)` también. (c) EuRoC
    `state_groundtruth_estimate0` trae velocidades y SESGOS → el
    dead-reckoning por ventanas es el test de CONVENCIONES perfecto antes de
    meter nada al grafo: 1 s de IMU real (V1_01) predice rot 0.33° / pos
    4.4 cm (medianas; p90 7.7 cm). Un signo de gravedad mal daría ½·g·t² ≈
    4.9 m; un frame mal, metros — este humo convierte "¿están bien q_RS,
    fuerza específica y g = −z?" en un número.

48. **"Quieto" no es std pequeña — y la dirección de g hereda el sesgo del
    acelerómetro** (v1.1 hito 2, imu_init.py). (a) En EuRoC el dron en
    reposo VIBRA con los motores: std_acc 0.3-1.1 m/s² parado (2.4 en
    vuelo); un detector estricto de varianza declara "nunca estático" en
    V1_01/V1_02 (medido). El detector correcto: umbrales LAXOS que separan
    reposo-vibrando de vuelo (std_gyro < 0.06, std_acc < 1.0) + |f̄| ≈ g +
    CONSISTENCIA entre mitades de la ventana (una deriva lenta — lo
    levantan en mano — pasa la std y falla ahí; V1_01 salta sus primeros
    0.5 s por eso). La vibración es de MEDIA CERO: b_g sale con error
    1.9-2.3e-3 rad/s en las tres V1 aun vibrando (el GT confirma reposo:
    |v| ≤ 0.015 m/s). (b) En reposo f̄ = −Rᵀ·g + b_a: la dirección de g
    queda ENTRELAZADA con b_a (error ≈ |b_a⊥|/g). Medido: 2.60° en V1_01
    (¡|b_a| = 0.55 m/s² — el ADIS16448 arranca torcido!) y 0.44-0.80° en
    V1_02/03; corrigiendo con el b_a del GT: 0.35-0.63° en las TRES. El
    criterio "dir(g) < 1°" lo cumple el MÉTODO, pero b_a no es observable
    en reposo (ni se distingue de inclinar g) — se refina en el grafo
    (hito 3), como en todo VIO. (c) El yaw TAMPOCO es observable (g es
    invariante a girar sobre la vertical): attitude_from_gravity devuelve
    la rotación mínima (yaw = 0 por convención). Consecuencia para el hito
    3: el grafo VI tiene gauge de 4 gdl (posición + yaw), no los 6/7 del
    visual puro — los priors deben reflejarlo.

49. **El primer sensor que DISCREPA del mapa desenmascara los priors
    fantasma — y una referencia de una sola corrida puede ser una muestra
    afortunada** (v1.1 hito 3a, gtsam_isam2.py). (a) EL BUG (latente desde
    v0.5): `_session_pids` se poblaba al VER un pid — incluidos los que
    esperaban en pendientes — así que la 2ª obs de cada punto que cruzaba
    llamadas parecía "re-siembra de otra época" y recibía un
    PriorFactorPoint3 (σ=2 cm) a su posición del mapper. La visión pura
    NUNCA lo notó (dos versiones enteras): el prior es consistente con la
    solución visual — mismo gauge — y solo estorba cuando otro sensor
    discrepa del mapa. El test nulo/observable del modo VI lo cazó: la
    escala corrupta ×1.3 quedaba CLAVADA en 1.25 por 28 priors fantasma
    (batch LM sobre el grafo espiado de la clase: mínimo en 1.244 — no era
    convergencia, era el CONTENIDO del grafo). Fix: un pid entra a
    _session_pids solo al INSERTARSE. Con él: escala 1.334→0.993, b_g err
    4.7e-4, b_a err 0.008 (desde 0 — lo que la init estática no ve, lección
    48, el grafo lo recupera). (b) LA REGRESIÓN A/B del fix (fr1_desk
    --depth --fast, la secuencia biestable): fix 4.6/4.7/18.9 cm; viejo
    400.8 (¡colapso, escala 0.029!) / 2.6. Moraleja doble: el "2.5 cm" de
    la lección 38 era UNA corrida de una distribución con varianza 2.6↔400
    (--fast = iSAM2 + hilo async: no determinista) — el ancla de regresión
    del stack rápido es fr2_xyz (estable: 1.5 vs 1.4, paridad ✓) y fr1_desk
    --fast pasa a deuda §8; y las referencias de configs no deterministas
    deben medirse con VARIAS corridas antes de fijarse como número. (c) De
    camino, dos verdades de iSAM2 en modo VI: con el default
    relinearizeSkip=10 el grafo se queda linealizado en el valor inicial
    (la escala ni se movía) → relinealizar SIEMPRE + umbral 0.05 + DOGLEG
    (GN puro no navega correcciones del 30%); y el gauge VI es 1 prior de
    pose (el 2º del modo visual congela la escala del valor inicial y pelea
    con el IMU). Solo en VI: la sintonía visual medida en v0.5 no se toca.

50. **La varianza del stack rápido era el WORKER — el pipeline síncrono es
    determinista AL BIT** (v1.1 hito 3b, el experimento discriminador).
    V1_01 estéreo con `--ba isam2` SÍNCRONO: dos corridas sin IMU dan
    EXACTAMENTE 4.6 cm/escala 1.006/106 perdidos/28 bucles — bit-idénticas
    — y dos con IMU, exactamente 5.4/1.005/105/26. Tres consecuencias:
    (a) TODA la varianza de `--fast` (mediana ~83 cm, rango 7-153) es el
    scheduling del hilo de mapeo: en un dron rápido, el KF procesado TARDE
    deja al tracking con mapa viejo justo cuando más lo necesita (en el
    escritorio de TUM no dolía: fr2_xyz --fast estable — el sobre de
    operación del ASYNC también tiene borde, no solo el del frontend).
    (b) iSAM2 síncrono (4.6) SUPERA al NumPy (6.9) en V1_01 — el backend y
    los factores están sanos; **el VIO síncrono (5.4, escala 1.005) queda
    en PARIDAD: el criterio del hito 3 CUMPLIDO**. (c) El determinismo
    bit-exacto del modo síncrono es una herramienta de regresión que no
    sabíamos que teníamos (adiós tolerancias del ±20% AHÍ) — y explica por
    qué fr1_desk --fast varía 2.6↔400 (lección 49) mientras el resto de la
    suite es reproducible. Anotado sin perseguir: el online con IMU es
    peor (59.5 vs 21.5) con final-KF igual — la métrica es final-KF
    (lección 25). El frente async (prioridad de cola, o el prior IMU del
    hito 4 que compensa el mapa viejo exactamente) va a deuda §8.

51. **El IMU cierra el sobre en V1_01 pero NO rescata V1_02/V1_03 — el
    cuello es el FRONTEND bajo blur, no el backend inercial** (v1.1 hitos
    4-5, dos resultados negativos que redefinen el criterio de v1.1).
    HITO 4 (prior IMU en el matching guiado + dead-reckoning en el coast,
    en vez de velocidad constante): el estado del cuerpo (R,v,p) vive en el
    frontend, se preintegra el gap entre frames con el sesgo VIGENTE del
    grafo, y la visión re-ancla (R,p) en cada pose aceptada / el grafo
    refresca (v,sesgos) en cada KF. **ÉXITO rotundo en V1_01: 5.0 cm,
    escala 1.005, perdidos 105→28** (mejor que 6.9 sin IMU y que el 5.4 del
    VIO-sync del hito 3) — el prior IMU engancha exactamente los frames que
    la velocidad constante perdía en las curvas. En sintético (vuelo
    agresivo) el prior de velocidad constante yerra 1.1 cm / 0.56° por
    frame (~5 px, se come la ventana de 15 px del guiado) y el prior IMU
    queda en precisión de máquina (test_imu_frontend, 4 tests). PERO en
    **V1_02/V1_03 el objetivo (< 10 cm, escala≈1, sin colapso) NO se
    alcanza, y el grafo IMU las EMPEORA**. Los 5 discriminadores en V1_02
    (todos síncronos), leídos juntos:
    | config | final-KF | escala | perdidos | resets | fallos |
    |---|---|---|---|---|---|
    | control iSAM2 sin IMU (mejor corrida) | **72.8** | **0.889** | 381 | 3 | 0 |
    | control iSAM2 sin IMU (otra corrida) | 142.0 | 0.633 | 486 | 3 | 0 |
    | grafo IMU + prior CV (ablación h3b) | 1108.5 | 0.039 | 662 | — | — |
    | grafo IMU + prior IMU (hito 4) | 539.9 | 0.112 | 487 | 4 | 2 |
    | grafo IMU + prior IMU, SIN reset (hito 5) | 20166 | 0.006 | 1584 | 3 | 0 |
    Cuatro verdades medidas: (a) **el culpable NO es mi prior del frontend
    — es el GRAFO IMU bajo resets**: mi prior MITIGA (539 < 1108 de la
    ablación con prior CV; perdidos 487 < 662), pero los perdidos son
    IDÉNTICOS con y sin prior (487 vs 486) → en V1_02 los frames no se
    pierden por mal prior de movimiento, se pierden porque el matching muere
    bajo blur y ahí NINGÚN prior inventa correspondencias. (b) **El grafo
    IMU colapsa la escala ACTIVAMENTE**: el control llega a 0.889 sin IMU y
    el CombinedImuFactor la hunde a 0.112 — la firma son los 2 fallos de
    update (sistemas indeterminados) que el control (0 fallos) no tiene: en
    cada reset el grafo VI hereda velocidad/sesgo del tramo ciego y se
    re-configura mal condicionado. (c) HITO 5 ("coast IMU sin reset": subir
    LOST_RESET_AFTER a 400 con IMU, que el dead-reckoning puentee y la reloc
    re-enganche el MISMO mapa) **FALLA 4×**: el dead-reckoning solo puentea
    apagones CORTOS (a 1 s deriva 4.4 cm, lección 47; a 20 s deriva
    cuadrática de metros), el dron se DESPLAZA durante el blur, el mapa deja
    de ser visible y el sistema queda en coast eterno (1584 perdidos, 8
    KFs). El reset es un mal MENOR — REVERTIDO. (d) De paso: **el iSAM2
    síncrono NO es bit-determinista en secuencias DIFÍCILES** (control V1_02
    142/486 vs 72.8/381 entre corridas) — casi seguro el RANSAC del PnP sin
    semilla, invisible en V1_01 (lección 50) porque ahí el tracking es tan
    sano que converge al mismo inlier-set; deuda §8. **VEREDICTO: el IMU
    cierra el sobre en movimiento suave-agresivo (V1_01) pero el blur severo
    de V1_02/V1_03 es un problema de FRONTEND** (ORB no detecta features
    estables) — la palanca es LightGlue/deblur/máscara de blur, no más
    backend inercial. El criterio de v1.1 se redefine: VIO validado y en
    paridad-o-mejor donde el frontend vive; V1_02/V1_03 quedan como límite
    MEDIDO del enfoque disperso (§7).

---

## 6. v0.4b — CERRADA (plan original abajo, como referencia de lo hecho)

**Estado: completada.** Criterio de aceptación (docs/04): *recuperación de un
"secuestro" en < 2 s de video*, sin degradar los números de referencia — CUMPLIDO
(recuperación en 2 frames, corredor 2.0 cm). Lo entregado:
- Relocalización (`_relocalize`) + estado RELOC + `self.reloc_events`, sobre el
  helper compartido `_match_against_kf` (extraído de `_try_close_loop`).
- Compuerta de movimiento por percentil (6× p95), emparejada con reloc vía el
  contador `_coast_count` (`_coast` ahora recibe gray/kps/desc e intenta reloc).
- Culling de puntos (`SparsePointMapper.cull_points`, flag `_active`, revivir al
  re-observar) llamado al final de `_insert_keyframe`.
- Test de secuestro reproducible: `tests/test_relocalization.py`.
- Lecciones 18-20 añadidas al código y a §5.

**El SIGUIENTE PASO es v0.45 (datos reales) — ver §7.** El plan detallado que
sigue se conserva como registro de diseño de lo implementado.

### Plan original de v0.4b (referencia histórica)

Criterio de aceptación (docs/04): *recuperación de un "secuestro" (saltar 30
frames) en < 2 s de video*, sin degradar los números de referencia (§3.2).

### 6.1 Relocalización (la pieza principal)

**Diseño** (en `vslam/frontend/tracker.py`):
1. Estado nuevo: `self._coast_count` (frames consecutivos en coast; resetear
   a 0 en cada TRACK exitoso y al inicializar).
2. En `_coast()`: incrementar contador. Si `>= RELOC_AFTER` (sugerido: 3),
   intentar `_relocalize(gray, kps, desc, info)` ANTES de aplicar velocidad
   constante (el coast como fallback si la reloc falla).
3. `_relocalize()`: es el mismo mecanismo del bucle SIN filtro temporal:
   - matching de `desc` contra cada entrada de `self._kf_db` (todas);
     quedarse con el mejor por número de matches; umbral sugerido
     `RELOC_MIN_MATCHES = 150` (más laxo que LOOP_MIN_MATCHES: aquí no hay
     riesgo de "bucle sin significado", solo de PnP fallido).
   - verificación geométrica: PnP contra los puntos 3D del candidato (vía
     `old["mp"]` — el mismo código de `_try_close_loop` pasos 1-2; **extraer
     ese bloque a un helper compartido** `_match_against_kf(old, kps, desc)
     -> (pairs, T_pnp, inliers)` para no duplicarlo).
   - si inliers ≥ `RELOC_MIN_INLIERS = 40`: aceptar — `self.T_w_c = T_pnp`,
     `self._T_prev = T_pnp`, `self._T_rel = np.eye(4)` (¡importante: la
     velocidad acumulada del coast ya no vale!), `self._coast_count = 0`,
     `info["state"] = "RELOC"`.
   - registrar el evento en `self.reloc_events` (lista, como loop_events).
4. Tras una reloc exitosa NO insertar keyframe inmediatamente (dejar que el
   flujo normal decida), pero sí considerar: los frames en coast largo
   corrompen `_recent`... (no hay estado de velocidad aparte de _T_rel ✓).

**Re-introducir la compuerta de movimiento** (ahora sí, emparejada):
- El código retirado está descrito en el comentario "NOTA (compuerta de
  movimiento...)" en `_track_step`. Diseño corregido: si el paso PnP >
  6 × percentil 95 del historial de pasos aceptados (≥ 20 muestras, guardar
  últimos 200), NO aceptar la pose → `_coast()` → que el contador dispare la
  relocalización (que SÍ puede decidir a qué modo volver, con verificación
  geométrica global). Sin la pareja reloc, la compuerta demostró ser dañina.

**Test de secuestro** (nuevo: `tests/test_relocalization.py` o dentro del
ejemplo — PREFERIR un script de verificación reproducible):
1. Correr el tracker sobre `data/synthetic_loop` alimentando frames 0..79 y
   luego SALTAR a 110..199 (simula oclusión/teleport de 30 frames).
2. Criterios: (a) el sistema entra en coast/reloc en <5 frames tras el salto;
   (b) recupera TRACK con pose correcta (error vs GT×gauge < 5 cm) en ≤ 60
   frames tras el salto (≈2 s a 30 fps); (c) el ATE del tramo posterior a la
   recuperación < 5 cm.
3. Ojo: tras el salto, el frame 110 ve una zona YA mapeada (la vuelta pasa
   por donde la ida) — es el escenario realista. La reloc debe encontrar el
   KF de la ida correspondiente.

**Riesgos conocidos**: el matching de reloc contra TODA la db es O(KFs); con
~15 KFs es trivial, documentar que a escala real esto es BoW (docs/03 §3).

### 6.2 Culling de puntos

**Diseño** (en `vslam/mapping/sparse.py`):
1. Añadir `self._active: List[bool]` (paralelo a _positions). `snapshot()`,
   `get_map()`, `observations()` filtran por activo. NO borrar entradas
   (los ids son índices estables — invariante del diseño).
2. `cull_points(min_obs=2, min_age_kfs=3)`: desactivar puntos cuyo total de
   observaciones < min_obs Y cuyo anchor quedó ≥ min_age_kfs keyframes atrás
   (necesita saber el orden de KFs: pasarle `kf_ids` o el id actual).
3. Llamarlo desde el tracker al final de `_insert_keyframe` (tras el BA).
4. Medir: el mapa del corredor tiene ~4200 pts al final; esperar reducción
   de 20-40% sin degradar ATE (si el ATE empeora >20%, revisar criterios).
5. NO implementar todavía la fusión de duplicados por descriptor (lección 12);
   el filtro anti-duplicados + covisibilidad ya contienen el problema.

### 6.3 Al cerrar v0.4b

1. Correr TODA la verificación de §3.2 + el test de secuestro.
2. Actualizar README (marcar v0.4b, números nuevos) y docs/02 si cambió el
   árbol; añadir las lecciones nuevas al código y a §5 de este documento.
3. **Proponer (otra vez) el primer commit** — a estas alturas es urgente.

---

## 7. Después de v0.4b: el orden establecido

El detalle completo está en [docs/04_hoja_de_ruta_v1.md](04_hoja_de_ruta_v1.md).
Resumen operativo de lo inmediato:

- **v0.45 — Datos reales** (EN PROGRESO — la etapa grande y la MÁS importante):
  HECHO:
  - ✅ Distorsión Brown-Conrady en `PinholeCamera` (campo `distortion`,
    `undistort_points`, `from_file` parsea k1..k3) + test. La geometría del repo
    asume el modelo ideal → el driver PRE-RECTIFICA (cv2.undistort).
  - ✅ Loader TUM RGB-D (`TUMRGBDLoader`, `tum_camera` con intrínsecos fr1/2/3,
    `read_tum_trajectory`, `associate_by_timestamp` mocap↔rgb). Driver
    `examples/05_tum_rgbd.py` + `scripts/benchmark_tum.py` (tabla batch).
  - ✅ **MATCHING GUIADO por reproyección** (lección 24): predecir pose +
    proyectar mapa local + buscar en ventana de 15 px. Eliminó los colapsos en
    las 3 secuencias (frames perdidos fr2_desk 1347→0). Curó la inanición de KFs
    (el piso de salud `KF_HEALTH_INLIERS` deja de ser sensible, lección 21;
    default 45). Incluye re-anclaje del mapa local tras reloc (`_local_ref_kf`).
  - ✅ **Métrica correcta + BA global offline** (lecciones 25-26): el ATE se
    medía sobre la trayectoria ONLINE (poses congeladas al emitirse), que no ve
    las correcciones del backend. Sobre la trayectoria FINAL de keyframes
    (`keyframe_trajectory`) + un BA global OFFLINE al terminar
    (`global_bundle_adjustment`): **fr1_xyz 1.8 / fr2_desk 3.7 / fr2_xyz 12.0 cm**
    (antes online 4.9/21.9/29), ≤5 frames perdidos, 0 relocs en las 3. El GBA
    online se probó y descarrilaba (lección 26) → offline. Y el BA no convergía
    con 10 iteraciones (lección 27): a 50, **fr2_xyz 12→0.4, fr2_desk 3.7→2.1 cm**.
  ESTADO TUM (trayectoria final de KFs, GBA 50 iters): fr1_xyz **1.8** / fr2_desk
  **2.1** / fr2_xyz **0.4** cm; 0 colapsos, ≤5 perdidos, 0 relocs — nivel ORB-SLAM.
  PENDIENTE (lo que queda de v0.45):
  - ✅ **CI en GitHub Actions** (`.github/workflows/ci.yml`): Ubuntu limpio,
    Python 3.11, numpy<2 + opencv-headless<5 (sin gtsam/torch — el núcleo es
    NumPy+OpenCV). Corre los 21 tests de geometría + distorsión + humo sintético
    (ejemplos 02/04 + secuestro). Verificado local; el primer run verde depende
    del push a GitHub. OJO: KITTI odometry son ~22 GB (no hay descarga por
    secuencia) → NO meter datasets en CI; el CI es solo sintético.
  - ✅ **Loader EuRoC MAV** (`EuRoCLoader`, `euroc_camera`, `read_euroc_groundtruth`
    + `examples/06_euroc.py`): parser del `sensor.yaml` sin PyYAML (intrínsecos,
    distorsión radtan, T_BS), timestamps ns→s, y GT del frame del CUERPO→CÁMARA
    con el extrínseco (docs advertían la trampa del brazo de palanca). Validado
    con fixture (`tests/test_euroc_loader.py`, 3 tests). PENDIENTE la corrida
    sobre una secuencia REAL: el host de EuRoC (robotics.ethz.ch) estaba caído
    (HTTP 000) al intentar bajar MH_01_easy (~1.5 GB) — reintentar / que Ariel
    la baje. El loader queda listo para correr en cuanto haya datos.
  - Loader KITTI odometry (image_0/ + calib P0 + poses 3×4 — descarga ~22 GB, el
    gray completo; sin descarga por-secuencia). Estresará la escala (exterior).
  - Criterio: ≥ 6 secuencias públicas SIN PERDERSE, tabla reproducible.
    **6 TUM medidas** (trayectoria final de KFs, ATE en cm / frames perdidos):
    ```
    fr2_xyz   0.4 / 5      fr1_xyz  1.8 / 0     fr2_desk 2.1 / 0     ← ✅ excelentes
    fr3_long 78.5 / 0      fr1_room 13.8/988    fr1_desk 2.2 / 560   ← límites (lección 28)
    ```
    4/6 trackean sin perderse (las 3 buenas + fr3, que trackea pero deriva); las
    fr1 handheld se pierden con ORB. **Con SuperPoint+LightGlue fr1_desk pasa de
    560 a 140 perdidos (lección 29)** — el frontend aprendido es la vía para las
    fr1. El residual (140) es un episodio estructural, NO umbrales (medido).
    Falta: re-correr el benchmark completo con learned; resolver fr3 (calibración/
    deriva); EuRoC cuando vuelva el host; para el residual de fr1, KLT/IMU.
- **v0.5 — Tiempo real (EN PROGRESO)**. Criterio: 30 fps a 640×480, mismo ATE
  ±5%. HECHO:
  - ✅ **PERFILADO** (regla de oro): fr2_desk/ORB da **4.3 fps** (232 ms/frame).
    Reparto — BA local **57%** (pico 2 s/KF síncrono), matching guiado **37%**
    (frames TRACK 93 ms), cv2 (ORB/match/PnP) solo **8%**. Refuta la conjetura de
    docs/04 (los candidatos eran extracción/matching/PnP): lo caro es el BA y el
    guiado, código Python/NumPy — NO cv2, que ya es C. [lección 30]
  - ✅ **Adaptador GTSAM del BA** (`vslam/backend/gtsam_ba.py`, `ba_backend="gtsam"`
    en el tracker): mismo problema que la referencia NumPy, test de equivalencia
    (`tests/test_gtsam_ba.py`). Medido en fr2_desk: KF-frame **1382→385 ms**
    (BA 3.6×), pico 3356→614, fps **5.8→9.5**, ATE equivalente (~1-2 cm).
  - ✅ **Matching guiado en C++** (`cpp/src/guided_match.cpp` → módulo pybind11
    `vslam_cpp`, lección 31): equivalencia exacta con la referencia Python
    (tests/test_guided_match_cpp.py). Frames TRACK **110→29 ms** (bajo los 33 ms
    de 30 fps); fps global **18.7** (4.3 al inicio de v0.5). Toolchain Windows:
    VS Build Tools 2022 + pybind11/cmake del env (comando en cpp/CMakeLists.txt).
  - ✅ **BA incremental iSAM2** (`ba_backend="isam2"`, `gtsam_isam2.py`, lección
    32): corredor KF 2060→82 ms (25×), 49 fps; fr2_desk paridad ATE (1.5 cm),
    0 fallos. Test formal: tests/test_isam2_ba.py (incl. reset/re-siembra).
  - ✅ **HILO DE MAPEO** (`async_mapping=True`, lección 33): bucle+BA+culling en
    worker; correcciones al tracking vía delta pendiente. fr2_desk: **p99
    400→126 ms, max 677→197 ms**, paridad ATE, 0 fallos. GIL release en vslam_cpp.
    Test: tests/test_async_mapping.py.
  - ✅ **BoW para el reconocimiento de lugar** (place_recognition.py, lección
    34): vocabulario en sesión (k-medias Hamming, voto de mayoría) + índice
    invertido + tf·idf. Query 2.7 ms; solo top-5 pagan verificación. fr2_desk:
    **46.7 fps, mediana 17 ms, p99 73 ms, ATE-KF 1.4 cm**.
  - ✅ **CRITERIO DE v0.5 CUMPLIDO en fr2_desk** (640×480, CPU): 30 fps pedidos,
    **46.7 medidos** (stack: guiado C++ + isam2 + hilo de mapeo + BoW; el stack
    rápido es opt-in: `ba_backend="isam2", async_mapping=True`; la referencia
    didáctica NumPy sigue siendo el default). ATE en paridad (±ruido RANSAC).
  **v0.5 CERRADA** (julio 2026, decisión de Ariel). Deuda menor trasladada a §8:
  validar --fast en las demás secuencias TUM; GTSAMBackend del grafo de poses.
- **v0.6 — RGB-D y estéreo (EN PROGRESO)**. Criterio (docs/04): TUM fr1_desk y
  fr2_xyz con **ATE < 5 cm MÉTRICO** (metros de verdad: alineación rígida SIN
  escala, o escala Umeyama ≈ 1.0). Estado del primer hito (todo MEDIDO):
  - ✅ Profundidad en `TUMRGBDLoader` (`with_depth=True`; depth.txt asociado
    por timestamp, PNG 16 bits, factor 1/5000 → metros; 0 = sin dato).
  - ✅ INIT RGB-D instantánea (`_initialize_rgbd`): retro-proyección métrica
    desde el frame 0 (sin danza de 2 vistas, sin gauge mediana=1); enciende
    `_metric`. Puntos de KF desde profundidad (nacen con 1 obs — BA los
    excluye hasta la 2ª; el buffer de iSAM2 ya lo manejaba por diseño).
  - ✅ Evaluación métrica: `ate(..., with_scale=False)` (Umeyama rígido) +
    escala de similitud como chequeo (≈1.00 = mapa de verdad en metros).
    `examples/05 --depth`. Rectificación de profundidad con INTER_NEAREST
    (interpolar profundidad a través de una discontinuidad inventa valores).
  - ✅ Bucle MÉTRICO en SE(3) (lección 35 — la lección grande del hito):
    Sim(3) re-escalaba el mapa métrico y componía el error (fr2_xyz 22 cm,
    escala 2.09). Con el fix: **fr2_xyz 4.7 cm MÉTRICO (escala 1.036) —
    criterio CUMPLIDO**; sin bucles 1.1 cm (techo de la secuencia).
  - ✅ HITO 2 — residuos de PROFUNDIDAD en el BA (lección 36): estéreo
    virtual u_R = u − bf/z (`STEREO_BF = 40`), residuo [u, v, u_R] en la
    referencia NumPy (BA local + GBA cuando `_metric`); observaciones (3,)
    en el mapper (el almacén no interpreta). De camino cayó el bug RAÍZ de
    fr1_desk: depth arranca tarde → init monocular → mapa MIXTO gauge/metros
    (`_metric=False`, escala 1.008 de casualidad) — fix: driver espera depth
    + invariante "puntos desde profundidad SOLO en mapa métrico".
    **CRITERIO v0.6 CUMPLIDO: fr1_desk 2.8 cm (escala 1.005, 0 perdidos) y
    fr2_xyz 1.5 cm (escala 0.96, antes 4.7)**. Ablación sin residuo: fr1
    12.8 cm/244 perdidos — el residuo cruza el episodio biestable 200-340.
    Levers refutados en su momento (contexto pre-fix, mapa mixto):
    SuperPoint+LightGlue (7.9 cm), DEPTH_MAX 8→4, GBA 50→100.
  - ✅ HITO 3 — ESTÉREO REAL (EuRoC, lección 37): `EuRoCStereoRig`
    (rectificación + bf) + `EuRoCStereoLoader` (disparidad SGBM → profundidad,
    misma firma RGB-D) + `examples/06 --stereo`. La cámara derecha VIRTUAL de
    RGB-D se vuelve REAL: u_R medido en vez de sintetizado, mismo residuo del
    BA. **V1_01_easy: 6.9 cm métrico, escala 1.002** (final de KFs, 234 KFs, 27
    bucles). Tests sin el dataset: tests/test_stereo.py (2).
  - ✅ FACTOR ESTÉREO EN GTSAM (lección 38): batch e iSAM2 usan
    `GenericStereoFactor3D`/`Cal3_S2Stereo` cuando llega u_R. `--fast --depth`
    en paridad con NumPy (fr2_xyz 1.4, fr1_desk 2.5 cm) a tiempo real.
  - ⏳ Restante de v0.6: más secuencias EuRoC (MH_*, V2_*) para medir el techo
    del enfoque; --fast sobre estéreo EuRoC (example 06 aún no expone --fast).
- **v0.7 — MAPA DENSO 3DGS** (EN PROGRESO — la etapa de la tesis, docs/01 §3.2):
  - ✅ HITO 1: rasterizador 3DGS diferenciable (`gaussian_render.py`, PyTorch
    puro): proyección + covarianza EWA + α-blending por transmitancia. Tests:
    proyección, gradiente por diferencias finitas, sobreajuste de vista >30 dB.
  - ✅ HITO 2: `GaussianSplattingMapper` detrás de MapperBase (`gaussian.py`):
    siembra desde la nube dispersa, `optimize` (renderiza y compara),
    `update_poses` rígido por submapa. Test multi-vista >30 dB + rigidez exacta.
  - ✅ HITO 3: medición REAL en fr1/desk (ejemplo 07, siembra densa por
    profundidad): 15.0 dB a 160×120 — el techo de la referencia densa (memoria)
    y de PyTorch puro (velocidad: tiled da 516 ms/iter con 15k gaussianas).
  - ✅ HITO 4 — gemela rápida y CRITERIO (lecciones 40-41): rasterizador POR
    TILES (equivalencia >40 dB, rompe la memoria) + **gsplat en DOCKER**
    (docker/Dockerfile.gsplat; en Windows nativo el link es imposible por el
    mangling nvcc↔MSVC). El test de equivalencia destapó el bug del MEDIO PÍXEL
    (+0.5, 25→60 dB). Cadena fotométrica completa (escala por punto, delta
    SE(3)+exposición por KF, 30k iters con decay, densificación/poda):
    **21.0 dB en fr1/desk full-res** a 20 ms/iter con ~500k gaussianas.
    Criterio recalibrado a PARIDAD SOTA (≥21 dB; el >30 de docs/04 es de
    dataset sintético) — **CUMPLIDO al filo**. Comando de referencia:
    `docker run --rm --gpus all -v <repo>:/workspace -v gsplat-cache:/root/.cache/torch_extensions
    vslam-gsplat python -u examples/07_gaussian_mapping.py --root data/tum/rgbd_dataset_freiburg1_desk
    --backend gsplat --scale 1 --max-points 300000 --iters 30000 --refine-poses
    --exposure --densify-every 500` → 21.0 dB (por KF: 17.1/20.9/29.9).
  - ✅ HITO 5 — integración EN VIVO (lección 42): `DenseMappingProcess`
    (dense_thread.py) — el mapper denso corre en PROCESO propio (el GIL hace
    inviable el hilo: +78% de latencia al tracking; el proceso la deja en
    +25% con MÁS presupuesto de mapa) y las correcciones de pose viajan por
    la cola del worker (carrera CUDA real cazada y eliminada). Criterio
    (2ª mitad) CUMPLIDO: mismos 596/596 frames con mapper ON, 80/80 KFs,
    0 fallos (examples/08, tabla en la lección 42).
  - **v0.7 COMPLETA** (committeada). Opcional no bloqueante: margen de
    PSNR (SSIM, ponderar KFs por blur), color RGB (el pipeline va en gris),
    Replica para el criterio >30 dB sintético.
- **v0.8 — ROS 2** (EN PROGRESO — criterio principal CUMPLIDO, lección 43):
  - ✅ HITOS 1-4: `vslam_msgs` + `vslam_ros` (4 nodos rclpy finos) compilan y
    corren en el contenedor; smoke headless completo (test/smoke_pipeline.py);
    **demo RViz vía WSLg confirmada** (trayectoria + nube + TF REP-105) sin
    tocar el núcleo. Comandos: `colcon build` en /workspace/ros2 y
    `ros2 launch vslam_ros tum_demo.launch.py rate:=10.0 rviz:=true`.
  - ✅ HITOS 5-6 (lección 44): los 3 nodos vslam son LIFECYCLE (pausa/reanuda
    verificado; bringup consumidores→productor) y `dataset:=euroc` corre el
    estéreo real por ROS (bf por parámetro; smoke: metric=True, 41k pts).
  - ⏳ Restante: webcam/RealSense (requiere usbipd-win + rama monocular del
    frontend — decisión anotada en lección 44); rosbag nativo (`ros2 bag
    record/play` de nuestros tópicos funciona out-of-the-box si se quiere).
- **v0.9 — ENDURECIMIENTO** (EN PROGRESO, lección 45):
  - ✅ HITO 1 — config declarativa: vslam/config.py (sobrescritura por
    instancia, typo falla en arranque, plantilla generada de las clases con
    `python -m vslam.config`), `--config` en examples/05. Tests: 5.
  - ✅ HITO 2 — degradación elegante: LOST_RESET_AFTER=90 → _reset_map()
    (archiva trayectoria, sesión nueva anclada en la pose extrapolada) + fix
    del frame CIEGO en _guided_match. Regresión fr1_desk intacta (2.8 cm).
  - ✅ HITO 3 — concurrencia (lección 46): test de estrés (async_mapping +
    lectores en caliente + reset en vuelo) + fix de ÉPOCAS de mapa en el
    worker. ✅ HITO 4 — docs/06 (3DGS, visita guiada de lecciones 39-42).
    ✅ HITO 5 — API freeze: vslam/__init__.py v0.9.0, 16 nombres públicos,
    import raíz sin torch/gtsam (verificado).
  - ✅ LICENCIA: **MIT** (decisión de Ariel; GTSAM es BSD-3 y todas las deps
    son permisivas — solo se importan, no imponen nada). LICENSE en la raíz;
    package.xml/setup.py de ROS actualizados. **v0.9 COMPLETA.**
- **v1.0 — COMMITEADA**: pyproject `vslam-edu` 1.0.0 (semver), LICENSE MIT,
  CONTRIBUTING, tabla de benchmarks en el README. Pendiente MANUAL de Ariel:
  `python -m build` + `twine upload` (PyPI), tag `v1.0.0` + GitHub Release,
  video demo (opcional).
- **v1.1 — VISUAL-INERCIAL (VIO)** (EN PROGRESO — decisión de Ariel, jul 2026).
  Contexto: comparando contra ORB-SLAM3, el gap real del repo no es el ATE en
  secuencias moderadas (factor 2-3×) sino el SOBRE DE OPERACIÓN — y la palanca
  número uno es el IMU (las lecciones 28-29 ya lo pedían: los fallos de
  movimiento agresivo no eran de umbral). docs/04 lo dejaba para post-1.0 con
  "GTSAM nos deja la puerta abierta". Es también el hito técnico que refuerza
  el posicionamiento "plataforma VIO" del posible paper (SoftwareX).
  - ✅ HITO 1 — preintegración de referencia (lección 47):
    `vslam/backend/imu_preintegration.py` (NumPy, Forster TRO 2017: ΔR/Δv/Δp,
    jacobianos de sesgo, covarianza 9×9 [φ,v,p], `residual` del futuro factor
    documentado, `preintegrate_between` por intervalo de frames) + lectores
    `read_euroc_imu`/`euroc_imu_params`/`read_euroc_state` (io/dataset.py).
    VERIFICADO (tests/test_imu_preintegration.py, 4 tests; en el job extras
    del CI): predict == dead-reckoning exacto (1e-13); sesgo a 1er orden;
    equivalencia GTSAM (deltas + predict + covarianza permutada); V1_01 real:
    rot 0.33° / pos 4.4 cm mediana en ventanas de 1 s contra el GT de estado.
  - ✅ HITO 2 — init VI ESTÁTICA (lección 48): `vslam/backend/imu_init.py` —
    detector de ventana quieta (el reposo de EuRoC VIBRA: umbrales laxos
    calibrados + |f̄| ≈ g + consistencia entre mitades) que entrega b_g
    (media del gyro), dirección de g en el cuerpo, R_wb inicial (rotación
    mínima, yaw = 0 — no observable) y v = 0. MEDIDO contra el GT de estado
    en las 3 V1 (tests/test_imu_init.py, 4 tests): b_g err 1.9-2.3e-3
    rad/s; dir(g) cruda 0.44-2.60° (V1_01 limitada por su |b_a| = 0.55
    m/s²) y 0.35-0.63° corrigiendo con el b_a del GT — el criterio < 1° lo
    cumple el método; b_a se refina en el grafo (hito 3). Ventanas
    detectadas: V1_01 [0.5, 2.5] s (salta la manipulación inicial),
    V1_02/V1_03 [0, 2] s. (La alineación dinámica tipo Martinelli queda
    como alternativa anotada si alguna secuencia MH_* arranca en vuelo.)
  - 🔶 HITO 3 — el factor IMU en el grafo rápido. DECISIÓN TOMADA (Ariel,
    jul 2026): **el grafo VI vive SOLO en la gemela GTSAM** (opción B) — la
    referencia educativa queda en la preintegración + residuo del hito 1
    (documentados y testeados); el solver es CombinedImuFactor/iSAM2 (que
    además ES la implementación de referencia del paper de Forster). El VIO
    es capacidad del stack --fast; el tracker NumPy de default sigue siendo
    visual puro — decisión consciente, no deuda olvidada.
    ✅ 3a — BACKEND (lección 49): modo VI de `ISAM2LocalBA` opt-in vía
    `configure_imu(noise, gravity_map, T_cam_imu, sesgos/velocidad de la
    init estática)`; `process_keyframe(..., imu_data=(ts,gyro,accel))`
    preintegra el segmento con el sesgo VIGENTE y añade V(kf)/B(kf) +
    CombinedImuFactor; las poses SIGUEN siendo de cámara (extrínseco por
    `body_P_sensor`); gauge VI = 1 prior de pose; params iSAM2 propios del
    modo (relinearizeSkip=1, umbral 0.05, Dogleg — GN puro no movía la
    escala) + 2 updates extra por KF; la cadena re-ancla tras reset (V/B
    con prior del último estimado; el sesgo es físico y sobrevive).
    MEDIDO (tests/test_imu_isam2.py, 3 tests, en CI extras): escala
    corrupta ×1.3 → sin IMU 1.334 (gauge), con IMU 0.993; b_g err 4.7e-4
    rad/s; b_a err 0.008 m/s² DESDE CERO; reset re-anclado sin fallos. De
    camino cayó el bug del prior fantasma de re-siembra (lección 49;
    regresión: fr2_xyz --fast 1.5 vs 1.4 ✓; fr1_desk --fast resultó tener
    varianza 2.6↔400 cm — a deuda §8).
    ✅ 3b — CABLEADO tracker/driver (el primer VIO del repo corre de punta
    a punta): `PnPTracker.enable_imu(noise, g_map, T_cam_imu, sesgos,
    segment_provider)` — el DRIVER es el dueño del reloj (Frame.timestamp
    sigue siendo deuda consciente): entrega `segment_provider(kf_a, kf_b) →
    (ts, gyro, accel)` y el tracker pide el segmento desde
    `imu_chain_tail` del backend (KFs coalescidos en el worker quedan
    cubiertos: el tiempo se particiona sin huecos). `_reset_map`
    re-configura la gemela nueva HEREDANDO sesgo/velocidad (físicos, no
    gauge). `EuRoCStereoRig.T_cam_imu` — OJO: lleva la rotación R1 de
    rectificación (el frame del tracker es la izquierda RECTIFICADA, no
    cam0). examples/06 gana `--fast` (deuda de v0.6 SALDADA) e `--imu`.
    MEDIDO en V1_01 con **n=4 por brazo** (config no determinista — la
    lección 49 exigía distribuciones, no muestras):
    ```
    numpy         6.9 cm / 34 perdidos (estable entre sesiones)
    --fast        17.6 / 48.8 / 116.4 / 153.2 → mediana ~83 | perdidos 115-206 | escala 0.60-0.99
    --fast --imu   7.3 / 22.0 / 24.2 / 61.7  → mediana ~23 | perdidos  45-126 | escala 0.90-1.00
    ```
    Lectura (n=4): (a) el IMU mejora TODOS los ejes de forma CONSISTENTE
    — final-KF mediana ~83→~23 cm, perdidos ~151→~74, online ~79→~46,
    escala 0.87→0.98 — y su mejor corrida (7.3 cm, escala 1.001) TOCA la
    paridad NumPy: el techo es paridad, el enemigo es la VARIANZA. (b) El
    stack --fast SIN IMU está ROTO en EuRoC (mediana ~83 vs 6.9 NumPy,
    perdidos ~151 vs 34) — problema dominante, PRE-existente (nunca medido:
    era deuda de v0.6) y ortogonal al IMU; la primera corrida (17.6) era
    la MEJOR de cuatro — lección 49 otra vez. (c) La hipótesis de la
    dilución del GBA visual queda EN PAUSA: la corrida VIO de 7.3 pasó por
    el mismo GBA — la varianza domina, no el GBA. RESUELTO por el
    experimento discriminador (`--ba isam2` síncrono, lección 50): el
    culpable es el WORKER ASYNC; el pipeline síncrono es determinista AL
    BIT y queda **isam2-sync 4.6 cm (¡mejor que NumPy 6.9!) / VIO-sync
    5.4 cm, escala 1.005 — PARIDAD: hito 3 CERRADO**. El frente async va
    a deuda §8; el hito 4 (prior IMU en el frontend) es además su
    compensador natural.
  - ✅ HITO 4 HECHO (lección 51) — predicción IMU en el frontend:
    `_imu_advance`/`_imu_anchor` en tracker.py sustituyen la velocidad
    constante (lección 24) por la preintegración del gap con el sesgo
    vigente del grafo, en el prior del matching guiado Y en el coast
    (dead-reckoning). `enable_imu` cablea la clase; el estado (R,p) lo re-ancla
    la visión en cada pose aceptada, (v,sesgos) los refresca el grafo por KF;
    bucle/reloc/reset rotan o re-siembran el estado. test_imu_frontend (4).
    **ÉXITO en V1_01: 5.0 cm, escala 1.005, perdidos 105→28** (mejor que
    6.9 sin IMU y 5.4 del hito 3). NO rescata V1_02/V1_03: el cuello es el
    frontend bajo blur, no el prior (perdidos idénticos con/sin prior IMU).
  - ❌ HITO 5 DESCARTADO (lección 51) — "coast IMU sin reset"
    (LOST_RESET_AFTER 90→400 con IMU, que el dead-reckoning puentee el
    apagón y la reloc re-enganche el MISMO mapa). MEDIDO 4× PEOR en V1_02
    (1584 perdidos vs 487, escala 0.006): el dead-reckoning no puentea
    apagones largos (deriva cuadrática; el dron se va y el mapa deja de ser
    visible) → coast eterno. El reset es un mal MENOR. REVERTIDO.
  - ⚠️ CRITERIO de v1.1 REDEFINIDO (lección 51): el IMU cierra el sobre en
    movimiento suave-agresivo (**V1_01 VIO 5.0 cm, paridad-o-mejor**) pero
    V1_02/V1_03 son un límite de FRONTEND, no de backend inercial — ORB no
    detecta features estables bajo el blur del dron (perdidos idénticos con
    y sin prior IMU). Rescatarlas exige LightGlue/deblur/máscara de blur en
    el frontend, NO más IMU. DECISIÓN de rumbo pendiente de Ariel: (A)
    atacar el frontend bajo blur (la única palanca medida para V1_02/V1_03);
    (B) cerrar v1.1 con el VIO validado en V1_01 + este límite documentado.
    BASELINE MEDIDO (jul 2026, examples/06
    --stereo, ATE final-KF métrico; V1_02/V1_03 bajadas del mirror GlowBond
    en HF — vicon_room1.zip trae bag+zip ASL por secuencia; el host ETH
    sigue caído):
    ```
    V1_01_easy       6.9 cm  escala 1.002 |   34/2912 perdidos, 27 bucles   OK (regresión exacta de lección 37)
    V1_02_medium   363.8 cm  escala 0.175 |  468/1710 perdidos, 11 relocs   COLAPSO
    V1_03_difficult 339.7 cm escala 0.131 | 1683/2149 perdidos, 70 relocs   COLAPSO
    ```
    La escala 0.13-0.18 en un mapa "métrico" dice que el tracking se rompe
    tanto que el mapa deja de ser mapa (V1_03 pasa el 78% del tiempo
    perdido; 70 relocs = tormenta). Es el modo de fallo de las lecciones
    28-29 en su versión extrema — rotación rápida + blur, no umbrales.
    CRITERIO v1.1: paridad en V1_01 (≤ 6.9 cm) y V1_02 + V1_03 SIN colapso
    con ATE métrico < 10 cm y escala ≈ 1 (referencia SOTA: ORB-SLAM3
    estéreo-inercial reporta ~2 cm en ambas; <10 con margen es honesto para
    un primer VIO).

---

## 8. Deuda técnica conocida (consciente y aceptada, no "olvidada")

| Ítem | Nota |
|---|---|
| ~~Primer commit pendiente~~ SALDADA | Repo con commits, público y v1.0 committeada |
| `covisible_kfs` es O(KFs×obs) y se llama POR FRAME | A ~50+ KFs necesitará caché/índice invertido (pid→KFs) |
| `snapshot()` reconstruye arrays por frame | Aceptable ahora; cachear cuando se perfile |
| `_kf_db` guarda kps+desc de todos los KFs | Memoria lineal; BoW lo reemplazará |
| `_try_close_loop` matchea contra toda la db por KF | Ídem |
| Frame.timestamp = 0.0 en los KFs internos | Propagar cuando importe (datasets reales) |
| Umbrales calibrados solo en sintético | v0.45: piso de salud de KF ya es perilla (lección 21). Resto pendiente por-dataset |
| Robustez de recorrido largo en real | fr2_xyz 35 / fr2_desk ~105 cm: KFs adaptativos + matching guiado + bucle a escala de sesión (lecciones 21-23). Sub-hito de v0.45 |
| learned.py (SuperPoint/DISK/LightGlue) | v0.45: VERIFICADO en GPU + INTEGRADO (LightGlue 2D-2D vía `self.matcher`, `_desc_matcher` para 3D-2D). Rescata fr1_desk (560→140 perdidos, lección 29). Recalibrar umbrales MEDIDO como NO-lever (inliers p10=91 ≫ 45); el residual es estructural. Pendiente: benchmark completo con learned |
| Números del benchmark en README pre-BA | Re-correr y refrescar al tocar el benchmark |
| Modo --no-ba del corredor colapsa (~200 cm) | Conocido; no es objetivo |
| ~~Adaptadores GTSAM sin residuo de profundidad~~ SALDADA (v0.6, lección 38) | gtsam_ba y gtsam_isam2 usan `GenericStereoFactor3D` + `Cal3_S2Stereo` cuando llega u_R. `--fast --depth` validado: fr2_xyz 1.4 / fr1_desk 2.5 cm, paridad con NumPy |
| examples/01 y tracker comparten conceptos duplicados | Deliberado (didáctica); no unificar |
| fr1_desk `--fast` tiene varianza 2.6↔400 cm entre corridas | Descubierto en el A/B de la lección 49 (n=5; iSAM2+hilo async no determinista sobre la secuencia biestable). El "2.5" de lección 38 era una muestra. Ancla de regresión --fast: fr2_xyz (estable). Investigar (semillas/iteraciones) fuera del camino crítico de v1.1 |
| El worker ASYNC colapsa en EuRoC (dron rápido) | Lección 50: --fast mediana ~83 cm en V1_01 (síncrono: 4.6 determinista al bit). El KF procesado tarde deja mapa viejo en vuelo rápido. Curas candidatas: prior IMU del hito 4 (compensa el mapa viejo), prioridad/presupuesto de cola. v1.1 mide su criterio en modo SÍNCRONO mientras tanto |
| ~~Licencia sin decidir~~ SALDADA (v0.9) | MIT (decisión de Ariel); deps permisivas (GTSAM es BSD-3) |
| iSAM2 síncrono NO es bit-determinista en secuencias DIFÍCILES | Lección 51: control V1_02 sin IMU dio 142/486 y 72.8/381 entre corridas (V1_01 sí es bit-idéntico, lección 50). Sospechoso: el RANSAC de `cv2.solvePnPRansac` sin semilla — invisible cuando el tracking es sano (mismo inlier-set), aflora al límite. Fijar semilla del RANSAC para recuperar el determinismo como herramienta de regresión también en EuRoC |
| V1_02/V1_03 colapsan: es FRONTEND, no backend | Lección 51: el IMU (hito 4) rescata V1_01 pero NO estas — el blur del dron rompe el matching ORB (perdidos idénticos con/sin prior IMU); el grafo IMU + reset colapsa la escala (0.11 vs 0.89). Palanca MEDIDA: LightGlue/deblur/máscara de blur en el frontend. Decisión de rumbo de Ariel (§7) |

1. Leer este documento completo y el README.
2. `git status` — verificar si ya hubo commits (si no: recordar ofrecerlo).
3. Si `data/` no existe, regenerar (comandos en §3.2).
4. Correr los tests de §3.2 (23 archivos en tests/, runner `__main__`) —
   todos verdes esperados (gtsam/C++ se saltan limpio si falta la dep;
   los de torch/gsplat requieren el env `vslam` o Docker).
5. Correr ejemplos 02 y 04 y comparar contra los números de referencia.
6. Trabajo EN CURSO: **v1.1 (VIO)** — hito 1 (preintegración, lección 47)
   HECHO; seguir con el hito 2 del plan de §7 (y su primer paso: baseline
   estéreo v0.6 en V1_02/V1_03/MH_*). v1.0: pendientes manuales de release
   en §3.5 (PyPI, tag, video). Deuda de §8 y restantes de §7 (EuRoC MH_*/
   V2_*, --fast en examples/06, benchmark con learned) siguen abiertos.
   Recordar: usar el env conda `vslam` (§2), NO el Python del sistema.
