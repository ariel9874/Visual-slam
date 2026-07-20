# Visual-SLAM — instrucciones para sesiones de desarrollo

**ANTES DE TOCAR NADA**: lee [docs/05_estado_y_plan_de_continuacion.md](docs/05_estado_y_plan_de_continuacion.md)
— contiene el estado exacto del proyecto, la metodología acordada, las 46
lecciones medidas (no re-descubrirlas), la deuda técnica conocida y el plan
paso a paso del siguiente hito. Es el documento de traspaso entre sesiones y
DEBE mantenerse actualizado al cerrar cada etapa.

## Reglas esenciales (detalle y porqués en docs/05)

- Repo educativo + arquitectura seria: la matemática va EN el código (bloques
  `─── La matemática ───`, en español; identificadores en inglés). Cada
  decisión con su medición; cada enfoque descartado, documentado con números.
- Nada se fusiona sin ejecutarse: tests (`python tests/test_X.py`, 23
  archivos) + ejemplos con ATE vs ground truth. Números de referencia y comandos
  en docs/05 §3.2. Umbrales solo por barrido medido.
- Convenciones fijas: `T_w_c` (cámara→mundo, ejes OpenCV), tangente `[ρ, ω, λ]`,
  gauge mediana=1, formato TUM. Interfaces de docs/02 no se rompen.
- Windows: prints solo ASCII/cp1252; `Select-Object -First` sobre un pipe de
  Python da exit 255 espurio. Toolchain completo en el env conda **`vslam`**
  (Miniforge): gtsam (conda-forge win-64), torch CUDA, lightglue, pytest.
  Usar su python: `C:\Users\ariel\miniforge3\envs\vslam\python.exe`. Restricción:
  gtsam exige `numpy<2` (no instalar opencv-python 5.x). Detalle en docs/05 §2.
- El usuario (Ariel) es experto en SLAM (Ph.D., GTSAM/LiDAR profesional):
  tono de colega arquitecto, en español, trabajo por fases de la hoja de ruta
  ([docs/04](docs/04_hoja_de_ruta_v1.md)).
- Estado git: repo AL DÍA y PÚBLICO (rama main; CLAUDE.md, docs/05 y tests
  versionados). Ofrecer commit al cerrar cada etapa. `data/` y `output/` se
  regeneran con scripts (están en .gitignore).

## Estado actual: v1.1 (VIO) EN PROGRESO — hito 1 hecho; v1.0 pendiente de publicación

**v1.1 (decisión de Ariel, jul 2026)**: visual-inercial — la palanca del sobre
de operación vs ORB-SLAM3 (lecciones 28-29). HITO 1 HECHO (lección 47):
`vslam/backend/imu_preintegration.py` (referencia NumPy de Forster: ΔR/Δv/Δp,
jacobianos de sesgo, covarianza 9×9 [φ,v,p], residuo del factor documentado) +
`read_euroc_imu`/`euroc_imu_params`/`read_euroc_state` en io/dataset.py.
Verificado: predict == dead-reckoning EXACTO (1e-13); sesgo a 1er orden;
equivalencia GTSAM — OJO: la wheel conda usa preintegración TANGENTE (acuerdo
de 2º orden, no bit-exacto), `preintMeasCov` ordena (θ,p,v) y
`ConstantBias(acc, gyro)`/`integrateMeasurement(acc, omega, dt)` van con el
ACELERÓMETRO primero —; dead-reckoning REAL V1_01: rot 0.33°/pos 4.4 cm
mediana a 1 s (valida frames/signos: un error daría metros).
tests/test_imu_preintegration.py (4, guards gtsam/dataset) + job extras del
CI. BASELINE estéreo v0.6 MEDIDO (define el criterio del hito 5): V1_01
6.9 cm OK (regresión exacta); **V1_02 363.8 cm (escala 0.175, 468/1710
perdidos) y V1_03 339.7 cm (escala 0.131, 1683/2149 perdidos, 70 relocs) —
ambas COLAPSAN sin IMU**. Objetivo v1.1: ambas <10 cm, escala ≈1, sin
colapso. Datos: V1_02/V1_03 extraídas en data/euroc/ (mirror GlowBond HF,
vicon_room1.zip: trae bag+zip ASL). HITO 2 HECHO (lección 48):
`vslam/backend/imu_init.py` — init estática (el reposo de EuRoC VIBRA:
umbrales laxos + consistencia entre mitades; b_g err 1.9-2.3e-3 rad/s en
las 3 V1; dir(g) cruda ≤ 2.6° — entrelazada con b_a, no observable en
reposo — y ≤ 0.63° con b_a del GT; yaw no observable → R_wb con yaw=0 →
el grafo VI tendrá gauge de 4 gdl). tests/test_imu_init.py (4). HITO 3a
HECHO (lección 49; decisión de Ariel: el grafo VI vive SOLO en la gemela
GTSAM): modo VI de `ISAM2LocalBA` — `configure_imu` + V/B por KF +
CombinedImuFactor (poses de cámara, extrínseco por body_P_sensor), gauge
VI = 1 prior, params propios (relinearizeSkip=1 + Dogleg; GN puro no movía
la escala), re-anclaje tras reset. Test nulo/observable: escala corrupta
1.334 sin IMU → 0.993 con IMU; b_a desde cero a 0.008. De camino: BUG
LATENTE de v0.5 cazado (prior fantasma de re-siembra — _session_pids
registraba pids PENDIENTES; solo el IMU, primer sensor que discrepa del
mapa, lo destapó) y hallazgo de regresión: fr1_desk --fast tiene varianza
2.6↔400 cm entre corridas (el 2.5 de lección 38 era UNA muestra; ancla
--fast = fr2_xyz, en paridad 1.5/1.4). HITO 3b HECHO — primer VIO del
repo corriendo: `tracker.enable_imu` (el DRIVER es dueño del reloj:
segment_provider entre frame-ids; cadena desde imu_chain_tail del
backend), `_reset_map` hereda sesgo/velocidad, `EuRoCStereoRig.T_cam_imu`
(¡con la R1 de rectificación!), examples/06 con `--fast` (deuda v0.6
saldada, y `--ba` como en 05) e `--imu`. MEDIDO con n=4 + experimento
discriminador (lecciones 49-50): --fast en EuRoC varía 7↔153 cm (mediana
~83) y el culpable es el WORKER ASYNC (KF tarde → mapa viejo en vuelo
rápido; deuda §8); el pipeline SÍNCRONO es determinista AL BIT y queda
**isam2-sync 4.6 cm (mejor que NumPy 6.9) / VIO-sync 5.4, escala 1.005 —
PARIDAD: hito 3 CERRADO** (V1_01; el IMU además: perdidos −60% y online
2× mejor bajo async). v1.1 mide su criterio en modo síncrono. HITO 4
HECHO (lección 51): prior IMU en el frontend — `_imu_advance`/`_imu_anchor`
en tracker.py sustituyen la velocidad constante (prior del guiado + coast
dead-reckoning) por la preintegración del gap con el sesgo vigente del
grafo; la visión re-ancla (R,p), el grafo refresca (v,sesgos) por KF.
tests/test_imu_frontend.py (4). **ÉXITO en V1_01: 5.0 cm, escala 1.005,
perdidos 105→28** (mejor que 6.9 sin IMU y 5.4 del hito 3). HITO 5
DESCARTADO (lección 51): "coast IMU sin reset" (LOST_RESET_AFTER 90→400
con IMU) MEDIDO 4× PEOR (V1_02 1584 perdidos, escala 0.006 — el
dead-reckoning no puentea apagones largos, coast eterno); REVERTIDO.
**RESULTADO CLAVE de v1.1: el IMU cierra el sobre en movimiento
suave-agresivo (V1_01) pero V1_02/V1_03 NO se rescatan — es un límite de
FRONTEND (blur del dron rompe ORB: perdidos IDÉNTICOS con/sin prior IMU),
no de backend inercial. El grafo IMU + reset colapsa la escala ACTIVAMENTE
(0.11 vs 0.89 del control sin IMU; 2 fallos de update).** Discriminadores
en lección 51. RUMBO (A) elegido por Ariel — FRONTEND APRENDIDO (lección
52): SuperPoint+LightGlue (ya integrado desde la 29, `--detector superpoint
--matcher lightglue`, cero código nuevo) **RESCATA V1_02: perdidos ~486→8,
escala 0.18→1.00, 5.5 cm SIN IMU / 3.9 cm CON IMU** — confirma el
diagnóstico de la 51 (el cuello era el frontend). Hallazgo: IMU y frontend
son palancas ORTOGONALES (prior/movimiento vs features/blur) y **el IMU
solo AYUDA sobre un frontend SANO** (con ORB dañaba: 539/escala 0.11 por el
grafo IMU+resets; con LightGlue 0 resets → 3.9/escala 1.00 — el reset era
el veneno, no el IMU). **CRITERIO de v1.1 CUMPLIDO en 2/3: V1_01 VIO 5.0
cm, V1_02 LightGlue+IMU 3.9 cm** (< 10 cm, escala 1.00, sin colapso). V1_03
sigue LÍMITE: LightGlue+IMU la mejora (perdidos 1810→629, resets 14→4) pero
los 4 resets residuales colapsan la escala (0.34) — tramos irrecuperables;
cura candidata = atacar el colapso de escala del grafo IMU tras reset
(deuda §8, hito futuro). Cuadro completo 4×3 en lección 52. Plan en §7.

## v1.0 COMMITEADA — pendiente de publicación (pasos manuales)

**v1.0**: pyproject.toml actualizado (`vslam-edu` 1.0.0, deps numpy<2 +
opencv, extras deep/gtsam/dense, license MIT), `__version__` 1.0.0 (import
verificado), CONTRIBUTING.md (las 6 reglas del repo), README con checklist
completo y TABLA DE BENCHMARKS (fr2_xyz 1.5 / fr1_desk 2.8 / fr2_desk 46.7
fps / V1_01 6.9 cm / 3DGS 21.0 dB). PASOS MANUALES de Ariel: `python -m
build` + `twine upload` (PyPI), tag `v1.0.0`, y el video demo (opcional).
Verificado 2026-07-17: vslam-edu NO está aún en PyPI y no hay tag v1.0.0;
el repo SÍ es público y el curso hermano vive en ariel9874/aprende-vslam.

## v0.9 COMPLETA (endurecimiento)

**v0.9 (lección 45)**: hito 1 = **config declarativa**
(`vslam/config.py`: las constantes de clase siguen siendo la documentación; el
YAML/JSON sobreescribe POR INSTANCIA; config vacía = bit-idéntico; typo falla
en el arranque; plantilla generada con `python -m vslam.config`; `--config` en
examples/05). Hito 2 = **degradación elegante**: LOST_RESET_AFTER=90 frames en
coast → `_reset_map()` archiva la trayectoria y re-inicializa una sesión nueva
ANCLADA en la pose extrapolada (init RGB-D ancla en self.T_w_c); las sesiones
no se fusionan (Atlas fuera de 1.0). El test del apagón cazó el bug del frame
CIEGO (cero keypoints crasheaba _guided_match). REGRESIÓN verificada: fr1_desk
2.8 cm/escala 1.005/0 perdidos (números exactos de referencia). Hito 3
(lección 46): test de ESTRÉS de concurrencia (async_mapping + lectores en
caliente + reset en vuelo) → fix de ÉPOCAS de mapa (el worker descarta jobs
de sesiones muertas; excepción con época cambiada = muerte esperada, no
fallo). Hito 4: **docs/06** (3DGS, visita guiada de L39-42). Hito 5: **API
FREEZE** — vslam/__init__.py v0.9.0, 16 nombres en __all__, import raíz sin
torch/gtsam (verificado). Tests: +7 (config 5, reset 1, concurrencia 1) y el
bug del frame ciego arreglado. LICENCIA: **MIT** (LICENSE en la raíz; GTSAM
es BSD-3, todas las deps permisivas — no imponen nada). **v0.9 COMPLETA y
committeada. Siguiente: v1.0** (PyPI, README benchmarks,
CONTRIBUTING, video demo). Docs/05 §7.

## v0.8 (ROS 2) — criterio CUMPLIDO

**v0.8 (lección 43)**: `ros2/vslam_msgs` (Keyframe con imagen+
depth+pose ÓPTICA+K; PoseGraphEdge; TrackingState) + `ros2/vslam_ros` (4 nodos
rclpy finos: dataset/frontend/backend/mapper). Conversión de ejes óptico↔REP-103
por CONJUGACIÓN solo en conversions.py (regla 4: el núcleo no importa ROS). El
mapper-consumidor de v0.7 hito 5 mapea 1:1 a nodo (proceso = lección 42 de
serie); backend_node materializa REP-105 (T_map_odom = T_map_kf·T_odom_kf⁻¹).
Compila y corre en el contenedor `docker/` (colcon; comandos en ros2/README).
Smoke medido (test/smoke_pipeline.py): odom+Path+PointCloud2+TF completo,
metric=True. **Demo RViz vía WSLg confirmada — criterio de v0.8 CUMPLIDO.**
Hitos 5-6 (lección 44): frontend/backend/mapper son LIFECYCLE (pausa/reanuda
verificado; bringup CONSUMIDORES→productor o se pierden los primeros KFs —
medido) y `dataset:=euroc` corre el estéreo real por ROS (euroc_demo.launch,
bf por parámetro; smoke: metric=True, 41k pts). OJO bash: `source X && ros2
launch ... &` backgroundea la lista ENTERA (usar `;`). Restante: webcam
(usbipd-win + rama monocular, decisión anotada). Docs/05 §7.

## v0.7 COMPLETA (mapa denso 3DGS)

**v0.7 (COMPLETA)**: `GaussianSplattingMapper` detrás de `MapperBase` (la
tesis de docs/01 §3.2: cambiar la representación del mapa sin tocar frontend ni
backend). Hito 1 = **rasterizador 3DGS diferenciable** PyTorch
(`gaussian_render.py`, EWA + α-blending; muestrea en CENTROS de píxel i+0.5 —
bug del medio píxel, lección 40); hito 2 = **mapper** (`gaussian.py`): siembra
(escala por punto ~step·z/fx), `optimize` (renderiza-y-compara + delta SE(3) y
exposición POR KEYFRAME + decay de lr de medias + densificación/poda opcionales),
`update_poses` rígido; hito 3 = medición real (referencia: 15.0 dB a 160×120,
techo memoria/velocidad); hito 4 = **gemelas**: `gaussian_render_tiled.py`
(PyTorch por tiles, equivalencia >40 dB) y `gaussian_render_gsplat.py` —
**gsplat SOLO corre en Docker** (`docker/Dockerfile.gsplat` + volumen
gsplat-cache; en Windows nativo el link falla por mangling nvcc↔MSVC, lección
40; comando de referencia en docs/05 §7). **CRITERIO recalibrado a paridad SOTA
(≥21 dB en fr1/desk; el >30 de docs/04 es de sintético) — CUMPLIDO: 21.0 dB**
full-res, 20 ms/iter, ~500k gaussianas (cadena de ablaciones en lección 41: el
techo real es la consistencia fotométrica del dataset, no la capacidad).
Hito 5 = **integración EN VIVO** (`dense_thread.py`, lección 42): el mapper
denso corre en PROCESO propio (`DenseMappingProcess`; el HILO paga +78% de
latencia por el GIL — medido; el proceso +25% con más presupuesto de mapa) y
`update_poses` viaja POR LA COLA del worker (carrera CUDA real si se muta el
mapa desde fuera). Criterio 2ª mitad CUMPLIDO: 596/596 frames, 80/80 KFs
(examples/08). Tests: 15 de v0.7 (render 3 + mapper 2 + tiled 3 + gsplat 4 en
Docker + dense 3). Opcional no bloqueante: SSIM/ponderar blur, color RGB,
Replica. Lecciones 39-42, docs/05 §7.

## v0.6 — CRITERIO RGB-D CUMPLIDO + estéreo real (CERRADA)

**v0.5 CERRADA** — criterio cumplido: **46.7 fps** en fr2_desk (pedía 30) a
640×480 en CPU, ATE en paridad. Stack rápido opt-in (`--fast` = isam2 + hilo de
mapeo; C++ y BoW auto); referencia NumPy de default. Lecciones 30-34.

**v0.6**: RGB-D y estéreo. Criterio: TUM fr1_desk y fr2_xyz con
**ATE < 5 cm MÉTRICO** (alineación rígida; escala Umeyama ≈ 1.0) — **CUMPLIDO
EN AMBAS: fr1_desk 2.8 cm (escala 1.005, 0 perdidos) y fr2_xyz 1.5 cm (escala
0.96, 80 bucles)**. Hito 1 (lección 35): init RGB-D instantánea (`_metric`),
Umeyama rígido (`--depth`), bucle métrico en **SE(3)** — Sim(3) re-escalaba el
mapa métrico y componía (22 cm/escala 2.09). Hito 2 (lección 36): **residuos
de profundidad en el BA** — estéreo virtual ORB-SLAM2, u_R = u − bf/z
(`STEREO_BF=40`), residuo [u,v,u_R] en el BA NumPy — y el BUG RAÍZ de
fr1_desk: su depth arranca tarde → init MONOCULAR accidental → mapa MIXTO
gauge/metros con escala 1.008 de casualidad; se detectó porque el run salió
BIT-IDÉNTICO (en sistema caótico eso = "el código nuevo no corre"). Fix:
driver espera depth + invariante "puntos desde depth solo en mapa métrico".
Ablación: sin residuo 12.8 cm/244 perdidos — el residuo es lo que cruza el
episodio biestable 200-340. Tests: tests/test_rgbd.py (5). Hito 3 (lección 37):
**ESTÉREO REAL EuRoC** — `EuRoCStereoRig` (rectificación cv2.stereoRectify, bf
desde P2) + `EuRoCStereoLoader` (disparidad StereoSGBM → profundidad, MISMA
firma `(ts,gray,depth)` que RGB-D) + `examples/06 --stereo`. La cámara derecha
virtual de RGB-D se vuelve REAL: u_R medido, mismo residuo del BA.
**V1_01_easy: 6.9 cm métrico, escala 1.002** (final de KFs, 234 KFs, 27 bucles).
Tests: tests/test_stereo.py (2, sin el dataset). OJO datos: host EuRoC oficial
(robotics.ethz.ch) caído; mirror ASL en HF `pepijn223/euroc-mirror`.
**FACTOR ESTÉREO EN GTSAM** (lección 38, deuda §8 SALDADA): gtsam_ba (batch) e
gtsam_isam2 (incremental) usan `GenericStereoFactor3D`/`Cal3_S2Stereo` cuando la
obs trae u_R; `--fast --depth` en paridad con NumPy (fr2_xyz 1.4, fr1_desk 2.5
cm métrico) a tiempo real. Restante v0.6: más secuencias EuRoC (MH_*, V2_*);
--fast sobre estéreo EuRoC (example 06 no expone --fast aún). Plan en docs/05 §7.

## Historial: v0.5 (núcleo C++ para tiempo real)

**v0.45 CERRADA** (datos reales). Resumen: matching guiado + BA global offline +
métrica de trayectoria final de KFs + SuperPoint/LightGlue integrados. TUM
movimiento moderado: **fr2_xyz 0.4 / fr1_xyz 1.8 / fr2_desk 2.1 cm**, nivel
ORB-SLAM. Límites medidos (lecciones 28-29): fr1 handheld (SuperPoint rescata
560→140) y fr3 deriva — techo del enfoque, no se sigue puliendo. Lecciones 21-29
en docs/05 §5. OJO: GBA OFFLINE (`global_bundle_adjustment`, lección 26).

**v0.5**: núcleo C++ para tiempo real. Criterio: 30 fps a 640×480,
mismo ATE ±5%. Progreso medido en fr2_desk/ORB: 4.3 fps → 9.5 (BA GTSAM batch,
lección 30) → **~21 fps** (matching guiado en C++ `vslam_cpp`, lección 31; TRACK
25-29 ms — tiempo real) + **iSAM2 incremental** (`ba_backend="isam2"`, lección 32:
BA a 34 ms/KF; corredor 25×/49 fps; paridad de ATE). Tests de equivalencia:
test_gtsam_ba, test_guided_match_cpp, test_isam2_ba. Compilar C++: comando en
cpp/CMakeLists.txt (VS Build Tools 2022; .pyd en la raíz, gitignored, uso AUTO).
**HILO DE MAPEO** (`async_mapping=True`, lección 33) y **BoW** (place_recognition.py,
lección 34) hechos. **CRITERIO DE v0.5 CUMPLIDO en fr2_desk: 46.7 fps** (pedía 30)
a 640×480 en CPU, mediana 17 ms, p99 73, ATE-KF 1.4 cm (paridad). Trayectoria por
perfilado dirigido: 4.3→9.5→18.7→25.7→46.7 fps. Stack rápido opt-in:
`ba_backend="isam2", async_mapping=True` (o `--fast` en examples/05); la
referencia NumPy sigue de default. PENDIENTE de cierre: validar --fast en las
otras secuencias TUM; GTSAMBackend del grafo de poses (opcional). Toolchain: env
conda `vslam` (visión+GPU) y `docker/` (ROS, v0.8).
