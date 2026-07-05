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

# Datos sintéticos (si data/ no existe — está en .gitignore):
python scripts/make_synthetic_sequence.py --output data/synthetic
python scripts/make_synthetic_sequence.py --output data/synthetic_loop --motion loop --frames 200

# Ejemplo 02 (forward): esperado ~3.1 cm ORB, ~0.3 cm SIFT
python examples/02_pnp_tracking.py --images data/synthetic/images --calib data/synthetic/calib.txt --output output/pnp --gt data/synthetic/groundtruth.txt

# Ejemplo 04 (corredor): esperado ~2.0 cm con y sin bucle (2 bucles cerrados;
# eran 2.2 antes del culling — la poda de espurios mejoró ligeramente el ATE)
python examples/04_loop_closure.py

# Test de secuestro (v0.4b): teletransporta 79->3 y verifica gate+coast+reloc.
# Esperado: perdida detectada en <5 frames, reloc contra KF0 en ~2 frames,
# ATE post-recuperación ~2 cm. Regenera data/synthetic_loop si falta.
python tests/test_relocalization.py
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
                 corazón del sistema, ~700 líneas — leerlo entero antes de
                 tocarlo)
vslam/backend/   factor_graph.py (interfaz + teoría MAP), pose_graph.py
                 (GaussNewtonPoseGraph genérico se3/sim3),
                 bundle_adjustment.py (BA con Schur + jacobianos analíticos)
vslam/mapping/   base.py (MapperBase), sparse.py (puntos+observaciones+
                 covisibilidad+re-anclaje SE3/Sim3+apply_similarity+PLY+
                 culling v0.4b: _active/cull_points/active_count)
vslam/           evaluation.py (Umeyama + ATE), io/dataset.py (loader genérico)
examples/        01 (2D-2D didáctico autocontenido), 02 (PnP+BA),
                 03 (grafo de poses simulado), 04 (corredor: bucle on/off)
scripts/         make_synthetic_sequence.py (forward: 3 planos; loop:
                 corredor de carteles disjuntos), benchmark_frontends.py
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

- **v0.45 — Datos reales** (la siguiente etapa grande y la MÁS importante):
  - Loaders TUM RGB-D (formato: rgb.txt/depth.txt con timestamps,
    groundtruth.txt TUM; asociación por timestamp más cercano), KITTI odometry
    (image_0/ + calib.txt P0 + poses.txt matrices 3×4) y EuRoC (cam0/data.csv,
    body↔cam extrinsics en sensor.yaml — cuidado: GT en frame del IMU).
  - Distorsión: añadir `distortion` (k1,k2,p1,p2,k3) a PinholeCamera +
    `undistort_points` (cv2) ANTES de la geometría; o pre-rectificar imágenes.
  - Los umbrales de §3.4 están calibrados en sintético: esperar re-calibración
    (hacerla con barridos y documentar).
  - Benchmark batch: extender scripts/benchmark_frontends.py o script nuevo
    con tabla por secuencia; CI en GitHub Actions (tests + humo sintético).
  - Criterio: ≥ 6 secuencias públicas sin perderse, tabla reproducible.
- **v0.5 — C++** (perfilar primero), **v0.6 — RGB-D**, **v0.7 — 3DGS mapper**,
  **v0.8 — ROS 2**, **v0.9 — endurecimiento**, **v1.0**.

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
| Umbrales calibrados solo en sintético | Re-calibrar en v0.45 |
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
