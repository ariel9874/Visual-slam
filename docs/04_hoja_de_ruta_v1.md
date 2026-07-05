# Hoja de Ruta hacia v1.0

> **Definición de 1.0** (la meta que ancla todo lo demás): un sistema vSLAM
> **híbrido** — frontend intercambiable (clásico/aprendido), backend de grafos
> de factores, mapeo denso intercambiable (3DGS) — corriendo en **tiempo real**
> sobre **datos reales**, validado en **benchmarks públicos** (TUM RGB-D,
> EuRoC, KITTI), integrado a **ROS 2**, con API estable y toda su matemática
> documentada en el código. Es decir: la tesis de docs/01 §5, materializada.

## Principios que ninguna versión puede violar

1. **Cada versión cierra con números**: un criterio de aceptación medible
   corrido por el benchmark. Lo que no se mide, no se fusiona.
2. **La matemática vive en el código**: cada técnica nueva llega con su bloque
   `─── La matemática ───` y sus lecciones medidas (el estilo v0.1–v0.35).
3. **Python primero, C++ después del perfil**: solo se reescribe en C++ lo que
   el perfilador señale, y la gemela C++ pasa los mismos tests.
4. **Los contratos de datos no se rompen** (docs/02 §4): las interfaces
   `TrackerBase` / `FactorGraphBackend` / `MapperBase` son la constitución.

## Las etapas

### v0.4 — El SLAM "correcto": consistencia y robustez del núcleo · [M]
La deuda técnica medida en v0.35, saldada:
- **Grafo de poses Sim(3)**: `sim3_exp/log` en `core/lie.py` (7 gdl) y
  `pose_graph.py` genérico por grupo — la redistribución retroactiva de bucles
  monoculares que el SE(3) no puede hacer (deriva de escala del 14%, medida).
- **Covisibilidad real**: el mapa local pasa de "últimos N keyframes" a un
  grafo de covisibilidad (puntos compartidos), como debe ser.
- **Higiene del mapa**: culling de puntos (poco observados / alta reproyección),
  fusión de duplicados tras bucles, descriptor representativo multi-vista.
- **Relocalización**: tracking perdido → búsqueda en la base de keyframes +
  PnP (la misma maquinaria del bucle, disparada por el estado COAST).
- ✅ *Criterio*: corredor sintético con bucle < 3 cm ATE; recuperación de un
  "secuestro" (saltar 30 frames) en < 2 s de video.

### v0.45 — Datos reales y evaluación seria · [M]
El salto del mundo sintético al real (aquí se rompen las suposiciones cómodas):
- Loaders TUM RGB-D / EuRoC / KITTI con sus convenciones y timestamps.
- **Distorsión**: modelo Brown-Conrady (y Kannala-Brandt) en `camera.py` —
  hasta ahora asumimos imágenes rectificadas.
- Robustez fotométrica: exposición/blur reales (aquí los frontends de docs/03
  se ganan su lugar de verdad).
- Benchmark batch: tabla ATE/RPE por secuencia, reproducible con un comando;
  CI en GitHub Actions (tests + humo sintético en cada push).
- ✅ *Criterio*: tabla reproducible en ≥ 6 secuencias públicas sin perderse;
  posicionarse honestamente frente a los números publicados de ORB-SLAM3
  (situarse, no ganarle: somos un sistema educativo sin años de tuning).

### v0.5 — Tiempo real: el núcleo C++ · [L]
- Perfilar primero (regla 3). Candidatos obvios: extracción + matching + PnP.
- `cpp/` implementa las rutas calientes con la MISMA interfaz, expuesto vía
  pybind11 (`import vslam_cpp as fast`); tests de equivalencia Python↔C++.
- Adaptador **GTSAM/iSAM2** como backend opcional (Linux/conda) — el contrato
  `FactorGraphBackend` ya lo espera; nuestra referencia NumPy queda como
  implementación didáctica y de respaldo.
- ✅ *Criterio*: 30 fps sostenidos a 640×480 en CPU de laptop, mismo ATE que
  la referencia Python (±5%).

### v0.6 — RGB-D y estéreo: escala métrica real · [M]
La ambigüedad de escala monocular (nuestra vieja conocida) se elimina con
sensores reales — y es prerrequisito práctico del mapeo denso:
- RGB-D: inicialización trivial (profundidad directa), PnP con prior de
  profundidad, mapa métrico en metros de verdad.
- Estéreo (EuRoC): triangulación por disparidad, mismo pipeline.
- ✅ *Criterio*: TUM RGB-D fr1/desk y fr2/xyz con ATE < 5 cm (métrico).

### v0.7 — La tesis cumplida: `GaussianSplattingMapper` · [L]
El módulo por el que existe la arquitectura (docs/01 §3.2, estilo Photo-SLAM):
- Mapper 3DGS detrás de `MapperBase`: keyframes con pose del tracker →
  optimización de gaussianas por rasterización diferenciable (gsplat/PyTorch).
- `integrate_keyframe` asíncrono DE VERDAD: hilo/proceso propio — aquí entra
  la concurrencia tracking/mapping (el diseño de 3 hilos de ORB-SLAM).
- `update_poses` tras bucles = transformación rígida de submapas de gaussianas
  (la generalización densa de nuestro `apply_similarity`).
- ✅ *Criterio*: PSNR de re-render > 30 dB en TUM fr1/desk; el tracking NO
  pierde frames por culpa del mapper (presupuesto medido).

### v0.8 — ROS 2 · [M]
El diseño de ros2/README.md, ejecutado:
- `vslam_msgs` + 3 lifecycle nodes componibles; TF `map→odom→base_link`
  (REP-105); QoS sensor-data para imágenes.
- Demos: rosbag de EuRoC y cámara real (webcam / RealSense).
- ✅ *Criterio*: demo en vivo con RViz (trayectoria + nube/splats) sin tocar
  el núcleo (la cáscara no contamina, regla 4).

### v0.9 — Endurecimiento y congelación de API · [M]
- Concurrencia revisada, degradación elegante (reinicio de mapa al perderse
  irrecuperablemente), configuración unificada (los umbrales didácticos de
  las clases pasan a config declarativa).
- Documentación completa: docs/05 (RGB-D), docs/06 (3DGS), tutoriales, y una
  pasada de revisión sobre todos los bloques de matemática.
- API freeze: lo que queda público en 1.0 se decide y se documenta aquí.

### v1.0 — Release · [S]
- Paquete en PyPI, versionado semántico, LICENSE definitiva, CONTRIBUTING,
  tabla de benchmarks publicada en el README, video demo.

## Lo que 1.0 deliberadamente NO incluye (y por qué)

| Fuera de 1.0 | Razón |
|---|---|
| Visual-inercial (IMU) | Es un mundo propio (preintegración, gravedad, sincronización); merece su propia serie post-1.0 — y GTSAM ya nos deja la puerta abierta. |
| Multi-mapa / Atlas | Complejidad de gestión enorme; la relocalización de v0.4 cubre el 80% del valor. |
| `NeRFMapper` | La interfaz lo admite, pero 3DGS domina el trade-off (docs/01 §3); NeRF queda como ejercicio comparativo post-1.0. |
| Escenas dinámicas / semántica | Investigación activa; el repo debe consolidar lo clásico antes. |

## Riesgos conocidos

- **v0.5 y v0.7 son las etapas grandes** ([L]): C++ multiplataforma y la
  concurrencia del mapper denso. Si hay que recortar, el orden de sacrificio
  es: estéreo (v0.6) → GTSAM adapter (v0.5) → nunca la evaluación (v0.45).
- La validación en datos reales (v0.45) puede revelar que el frontend ORB puro
  no basta en secuencias difíciles — el plan B ya está instalado: los
  adaptadores aprendidos de docs/03 (`superpoint`/`lightglue`) con GPU.
