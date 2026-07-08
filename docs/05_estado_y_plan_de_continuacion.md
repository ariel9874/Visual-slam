# Estado del Proyecto y Plan de Continuación

> **Para quién es este documento**: cualquier sesión futura (humana o de un
> asistente) que continúe el desarrollo. Contiene TODO lo necesario para
> retomar el trabajo sin re-descubrir nada: contexto, metodología, estado
> exacto con números, lecciones medidas, deuda técnica y el siguiente paso
> detallado. Última actualización: julio 2026, al cierre de **v0.4b**.

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
   las encarna, con los números medidos. Ver §5: hay ~17 y son oro educativo.
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

## 3. Estado exacto al cierre de v0.4a

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
| v0.45 (en progreso) | **Datos REALES**: distorsión Brown-Conrady, loader TUM RGB-D, driver `examples/05`, benchmark batch, **matching guiado por reproyección**, re-anclaje de mapa local tras reloc, **BA global offline** (50 iters), métrica de trayectoria final de KFs | **TUM movimiento moderado (trayectoria final de KFs): fr2_xyz 0.4 / fr1_xyz 1.8 / fr2_desk 2.1 cm**, 0 colapsos. En 6 secuencias, las fr1 handheld se pierden y fr3 deriva (límites del frontend mínimo, lección 28). Sintético mejoró: 02 2.4, 04 1.7, secuestro 1.1 cm |

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
                 matching.py (ratio/crosscheck/flann, firma con kps para
                 aprendidos), learned.py (SuperPoint/DISK/LightGlue,
                 EXPERIMENTAL, requiere [deep]), tracker.py (PnPTracker: el
                 corazón del sistema, ~850 líneas — leerlo entero antes de
                 tocarlo. v0.45: _guided_match (matching por reproyección),
                 _local_ref_kf (re-anclaje del mapa local tras reloc),
                 global_bundle_adjustment (BA global OFFLINE, lo llama el driver),
                 keyframe_trajectory (métrica final vs online))
vslam/backend/   factor_graph.py (interfaz + teoría MAP), pose_graph.py
                 (GaussNewtonPoseGraph genérico se3/sim3),
                 bundle_adjustment.py (BA con Schur + jacobianos analíticos)
vslam/mapping/   base.py (MapperBase), sparse.py (puntos+observaciones+
                 covisibilidad+re-anclaje SE3/Sim3+apply_similarity+PLY+
                 culling v0.4b: _active/cull_points/active_count)
vslam/core/      camera.py: + distorsión Brown-Conrady (v0.45): campo
                 distortion, undistort_points (cv2), from_file parsea k1..k3
vslam/           evaluation.py (Umeyama + ATE), io/dataset.py (loader genérico +
                 TUMRGBDLoader/tum_camera/read_tum_trajectory/associate_by_timestamp
                 + EuRoCLoader/euroc_camera/read_euroc_groundtruth (parser
                 sensor.yaml sin PyYAML, GT cuerpo→cámara), v0.45)
examples/        01 (2D-2D didáctico autocontenido), 02 (PnP+BA),
                 03 (grafo de poses simulado), 04 (corredor: bucle on/off),
                 05 (datos reales TUM RGB-D, v0.45), 06 (EuRoC MAV, v0.45)
scripts/         make_synthetic_sequence.py (forward: 3 planos; loop:
                 corredor de carteles disjuntos), benchmark_frontends.py,
                 benchmark_tum.py (tabla batch por secuencia TUM, v0.45)
tests/           5 archivos de geometría (21 tests) + test_relocalization.py
                 (secuestro, v0.4b), todos con runner __main__
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

- **Git: repo con commits (rama main).** OJO: `CLAUDE.md`, `docs/05` y
  `tests/test_relocalization.py` siguen SIN versionar (untracked); README,
  tracker.py y sparse.py tienen cambios de v0.4b sin commitear. Ofrecer el
  commit de cierre de v0.4b. Su perfil de GitHub tiene un bloque comentado
  esperando a que este repo sea público.
- Licencia sin decidir (sugerido MIT/Apache-2.0; pyproject sin campo license).
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
    fr1 handheld se pierden. Para cumplir "6 SIN PERDERSE" con calidad falta:
    frontend más robusto (fr1) y resolver fr3 (calibración/deriva) — o añadir
    secuencias más amables (EuRoC cuando vuelva el host, o fr1_desk2/fr2_360).
- **v0.5 — C++** (perfilar primero), **v0.6 — RGB-D**, **v0.7 — 3DGS mapper**,
  **v0.8 — ROS 2**, **v0.9 — endurecimiento**, **v1.0**.
  - Infraestructura ya lista para v0.6/v0.8: contenedor `docker/` con ROS 2 +
    el núcleo Python (ver §2 y `docker/README.md`). Los wrappers previstos
    (`vslam_ros`, `vslam_msgs`, TF REP-105) están diseñados en `ros2/README.md`.

---

## 8. Deuda técnica conocida (consciente y aceptada, no "olvidada")

| Ítem | Nota |
|---|---|
| Primer commit pendiente | Prioridad administrativa #1 |
| `covisible_kfs` es O(KFs×obs) y se llama POR FRAME | A ~50+ KFs necesitará caché/índice invertido (pid→KFs) |
| `snapshot()` reconstruye arrays por frame | Aceptable ahora; cachear cuando se perfile |
| `_kf_db` guarda kps+desc de todos los KFs | Memoria lineal; BoW lo reemplazará |
| `_try_close_loop` matchea contra toda la db por KF | Ídem |
| Frame.timestamp = 0.0 en los KFs internos | Propagar cuando importe (datasets reales) |
| Umbrales calibrados solo en sintético | v0.45: piso de salud de KF ya es perilla (lección 21). Resto pendiente por-dataset |
| Robustez de recorrido largo en real | fr2_xyz 35 / fr2_desk ~105 cm: KFs adaptativos + matching guiado + bucle a escala de sesión (lecciones 21-23). Sub-hito de v0.45 |
| learned.py (SuperPoint/DISK/LightGlue) | VERIFICADO en GPU (v0.4b): SuperPoint+LightGlue corren en la RTX 4070 (env `vslam`). Falta integrarlo/benchmark en secuencias |
| Números del benchmark en README pre-BA | Re-correr y refrescar al tocar el benchmark |
| Modo --no-ba del corredor colapsa (~200 cm) | Conocido; no es objetivo |
| examples/01 y tracker comparten conceptos duplicados | Deliberado (didáctica); no unificar |
| Licencia sin decidir | Preguntar a Ariel en el commit |

---

## 9. Checklist de arranque para la próxima sesión

1. Leer este documento completo y el README.
2. `git status` — verificar si ya hubo commits (si no: recordar ofrecerlo).
3. Si `data/` no existe, regenerar (comandos en §3.2).
4. Correr los 5 archivos de tests (§3.2) — 21 OK esperados.
5. Correr ejemplos 02 y 04 y comparar contra los números de referencia.
6. v0.4b CERRADA. Continuar con **v0.45 (datos reales)** — §7 y docs/04 —
   salvo que Ariel indique otra cosa. Recordar: usar el env conda `vslam`
   (§2), NO el Python del sistema.
