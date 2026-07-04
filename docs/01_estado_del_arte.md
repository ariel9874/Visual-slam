# Estado del Arte en Visual SLAM (2026)

> **Objetivo del documento**: mapa comparativo de las tres grandes familias de vSLAM
> actuales — clásicos, deep learning y mapeo con campos de radiancia — para justificar
> la arquitectura híbrida que persigue este repositorio.

## 0. El problema y su anatomía

**SLAM** (Simultaneous Localization and Mapping) estima simultáneamente la trayectoria de la
cámara y un modelo del entorno. Desde PTAM (2007), casi todos los sistemas separan el problema
en dos procesos con frecuencias distintas:

- **Tracking (frontend)**: estima la pose de *cada* frame en tiempo real contra el mapa actual.
- **Mapping + optimización (backend)**: refina un subconjunto de frames (*keyframes*) y la
  estructura del mapa, a menor frecuencia, típicamente mediante *bundle adjustment* o grafos
  de factores; detecta y corrige *cierres de bucle* (reconocer un lugar ya visitado para
  eliminar la deriva acumulada).

Las familias siguientes se distinguen en **qué señal de la imagen usan** (características vs.
píxeles vs. representaciones aprendidas) y en **cómo representan el mapa** (puntos dispersos
vs. campos densos diferenciables).

---

## 1. Métodos clásicos

### 1.1 Basados en características (indirectos) — ORB-SLAM3

**Referencia**: Campos et al., *ORB-SLAM3: An Accurate Open-Source Library for Visual,
Visual-Inertial and Multi-Map SLAM*, IEEE T-RO 2021.

**Cómo funciona.** Reduce cada imagen a un conjunto disperso de características ORB
(esquinas FAST + descriptor binario BRIEF con orientación). El tracking empareja
características entre frames y minimiza el **error de reproyección** (distancia en píxeles
entre la observación y la proyección del punto 3D). Tres hilos concurrentes:

1. *Tracking*: pose por frame mediante modelo de velocidad constante + PnP contra el mapa local.
2. *Local mapping*: triangulación de nuevos puntos y bundle adjustment local sobre el grafo de
   covisibilidad (keyframes que observan puntos comunes).
3. *Loop & map merging*: reconocimiento de lugares con bolsa de palabras (DBoW2), corrección
   de bucles sobre el grafo esencial, y **Atlas**: múltiples mapas y sesiones que se fusionan
   al reencontrarse.

Soporta monocular, estéreo, RGB-D y todas sus variantes inerciales (con preintegración de IMU
en estimación MAP conjunta).

**Beneficios**
- Precisión de referencia entre los métodos clásicos; tiempo real en CPU (sin GPU).
- Ecosistema maduro: relocalización, cierre de bucle, multi-mapa, multi-sensor.
- Las características son razonablemente invariantes a iluminación y punto de vista.

**Cuellos de botella**
- **Escenas pobres en textura** (paredes lisas, pasillos) o con *motion blur*: sin esquinas no
  hay sistema. Es su modo de fallo dominante.
- El mapa resultante es una **nube dispersa de puntos**: suficiente para localizar, inútil
  directamente para navegación densa, inspección o realismo visual.
- Escenas dinámicas (personas, vehículos) violan la suposición de mundo rígido.
- Base de código compleja y muchos hiperparámetros ajustados a mano por sensor.

### 1.2 Métodos directos — DSO

**Referencia**: Engel et al., *Direct Sparse Odometry*, IEEE TPAMI 2018 (y LSD-SLAM, ECCV 2014).

**Cómo funciona.** En lugar de descriptores, opera **directamente sobre intensidades de
píxel**: selecciona píxeles con gradiente alto (no necesitan ser esquinas) y minimiza el
**error fotométrico** — la diferencia de intensidad entre el píxel original y su reproyección
en otro frame. DSO optimiza conjuntamente, en una ventana deslizante de ~7 keyframes, las
poses, las profundidades inversas de los puntos y parámetros afines de brillo, marginalizando
estados antiguos con el complemento de Schur. Se beneficia enormemente de **calibración
fotométrica** (respuesta del sensor, viñeteo, tiempo de exposición).

**Beneficios**
- Funciona donde los métodos de características fallan: bordes y gradientes suaves cuentan,
  no solo esquinas → más robusto en escenas de textura débil.
- Precisión geométrica sub-píxel y mapas semi-densos (más información que ORB-SLAM).
- Formulación elegante y eficiente; tiempo real en CPU.

**Cuellos de botella**
- La **constancia de brillo** es frágil: auto-exposición, balance de blancos o iluminación
  cambiante lo rompen si no hay calibración fotométrica.
- Sensible a *rolling shutter* y a mala calibración geométrica.
- Sin cierre de bucle ni relocalización nativos (LDSO los añade a posteriori): es odometría,
  la deriva no se corrige.
- Inicialización monocular delicada.

---

## 2. Métodos basados en Deep Learning

### 2.1 DROID-SLAM

**Referencia**: Teed & Deng, *DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D
Cameras*, NeurIPS 2021.

**Cómo funciona.** Es la síntesis "aprender a optimizar": construye un grafo de frames y, para
cada arista, una red recurrente (ConvGRU, heredera de RAFT) predice iterativamente **flujo
óptico denso con pesos de confianza**. Una capa de **Dense Bundle Adjustment diferenciable**
resuelve entonces las poses y un campo de profundidad por píxel que mejor explican ese flujo
(Gauss-Newton dentro de la red). El ciclo predicción→optimización se repite; el backend hace
BA global sobre todo el grafo. Entrenado end-to-end solo con datos sintéticos (TartanAir),
generaliza a datasets reales.

**Beneficios**
- **Robustez excepcional**: prácticamente no tiene fallos catastróficos en EuRoC/TUM/ETH3D,
  donde los clásicos se pierden; precisión que iguala o supera a ORB-SLAM3 en muchos benchmarks.
- Un único modelo cubre monocular, estéreo y RGB-D sin reentrenar.
- Produce profundidad densa por keyframe (base perfecta para mapeo denso posterior).

**Cuellos de botella**
- **Hambre de GPU**: varios GB de VRAM en inferencia (el paper usaba dos GPUs para el modo
  completo); inviable en robots embebidos pequeños.
- Tiempo real solo con hardware potente; latencia del backend global.
- Caja negra parcial: menos interpretable/depurable que un pipeline geométrico clásico.

> **Evolución relevante**: DPVO / DPV-SLAM (Teed et al., NeurIPS 2023) sustituye el flujo denso
> por *parches dispersos* aprendidos: ~3-4× más rápido y mucha menos memoria con precisión
> similar. Y desde 2024-2025, los modelos fundacionales de reconstrucción 3D de dos vistas
> (DUSt3R/MASt3R) han generado sistemas como **MASt3R-SLAM** (Murai et al., CVPR 2025), que
> obtienen geometría métrica densa *feed-forward* incluso sin calibración precisa.

### 2.2 TartanVO

**Referencia**: Wang et al., *TartanVO: A Generalizable Learning-based VO*, CoRL 2020.

**Cómo funciona.** Odometría visual (no SLAM completo) puramente aprendida: una red de flujo
óptico (estilo PWC-Net) alimenta una red de pose que regresa la transformación relativa entre
dos frames. Dos decisiones clave para generalizar: entrenamiento masivo en simulación
(TartanAir) con condiciones extremas, una **pérdida "hasta escala"** que acepta la ambigüedad
monocular, y una **capa de normalización de intrínsecos** que le permite funcionar con cámaras
distintas a las de entrenamiento.

**Beneficios**
- Robusto en condiciones donde la geometría clásica sufre (movimiento agresivo, niebla, lluvia,
  poca textura) — no depende de detectar y casar esquinas.
- Generaliza a datasets reales sin fine-tuning; pipeline simple de desplegar.

**Cuellos de botella**
- **Solo odometría**: sin mapa, sin cierre de bucle, sin relocalización → la deriva crece sin
  límite; precisión inferior a los métodos con optimización en secuencias "fáciles".
- Escala monocular ambigua (consistente pero no métrica).
- Requiere GPU; la precisión depende de cuánto se parezca el dominio al de entrenamiento.

> **Nota de diseño**: otra vía híbrida muy práctica es mantener el pipeline geométrico clásico
> pero sustituir las piezas frágiles por versiones aprendidas: características **SuperPoint** y
> emparejadores **SuperGlue/LightGlue**. Mismo backend, frontend mucho más robusto. Nuestra
> interfaz `FeatureExtractor` está pensada exactamente para ese intercambio.

---

## 3. Mapeo de nueva generación: campos de radiancia

El cambio de paradigma: el mapa deja de ser una nube de puntos y pasa a ser una
**representación densa y diferenciable optimizada por síntesis de imágenes** ("render y
compara"). La pregunta ya no es solo "¿dónde estoy?" sino "¿puedo re-renderizar el mundo?".

### 3.1 SLAM con campos neurales implícitos (NeRF-SLAM)

**Referencias**: iMAP (Sucar et al., ICCV 2021) — el pionero; NICE-SLAM (Zhu et al., CVPR 2022)
— rejillas jerárquicas de características; NeRF-SLAM (Rosinol et al., IROS 2023) — DROID-SLAM
como tracker + Instant-NGP como mapa; Co-SLAM, ESLAM (2023) — codificaciones hash/tri-planos.

**Cómo funciona.** El mapa es un campo neural (MLP puro en iMAP; rejillas de características +
MLPs pequeños después) que asigna a cada punto 3D densidad y color. Se optimiza *online*
minimizando el error de re-renderizado (volumétrico) contra los frames que llegan. El tracking
puede hacerse invirtiendo el render (optimizar la pose que mejor re-renderiza el frame), aunque
los sistemas más sólidos —como NeRF-SLAM— delegan el tracking en un frontend robusto (DROID) y
usan el campo neural solo como **backend de mapeo** supervisado con profundidades e
incertidumbres del tracker.

**Beneficios**
- Mapas **densos, continuos y completos**: interpolan y rellenan huecos que ningún método
  disperso cubre; exportables a mallas.
- Fotorrealismo y consistencia multi-vista; manejo natural de incertidumbre.
- Memoria compacta (los pesos comprimen la escena).

**Cuellos de botella**
- **Coste computacional brutal**: el render volumétrico exige cientos de muestras por rayo;
  aun con hash-grids (Instant-NGP) el mapeo online satura una GPU de gama alta.
- **Olvido catastrófico**: optimizar con los frames nuevos degrada zonas antiguas del mapa
  (se mitiga re-muestreando keyframes antiguos, pero limita la escala).
- Escala de habitación/apartamento; el tracking por render invertido es frágil (cuenca de
  convergencia pequeña) → casi todos necesitan RGB-D o un tracker externo.
- Cerrar bucles implica deformar un campo implícito: problema abierto.

### 3.2 SLAM con Gaussian Splatting (3DGS-SLAM)

**Referencias**: 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) — la representación;
SplaTAM (Keetha et al., CVPR 2024); Gaussian Splatting SLAM "MonoGS" (Matsuki et al., CVPR
2024); **Photo-SLAM** (Huang et al., CVPR 2024); RTG-SLAM, GS-ICP-SLAM (2024) — variantes
orientadas a tiempo real.

**Cómo funciona.** El mapa es un conjunto **explícito** de gaussianas 3D anisótropas (posición,
covarianza, opacidad, color) que se renderizan por **rasterización diferenciable** — órdenes de
magnitud más rápida que el ray-marching de NeRF. El tracking minimiza error fotométrico (y de
profundidad, si hay) del frame actual contra el render del mapa; el mapeo densifica, poda y
optimiza gaussianas sobre los keyframes. MonoGS demuestra el caso monocular puro; SplaTAM usa
RGB-D con siluetas para decidir dónde densificar; **Photo-SLAM ilustra la arquitectura híbrida
que este repo persigue: tracking clásico de ORB-SLAM3 + mapa fotorrealista de gaussianas**.

**Beneficios**
- Render en tiempo real → el ciclo "render y compara" por fin es interactivo.
- Representación **explícita y editable**: las gaussianas se pueden transformar rígidamente
  (submapas tras un cierre de bucle), recortar, fusionar — mucho más manejable que un MLP.
- Calidad visual estado del arte; los gradientes fluyen bien (optimización estable).

**Cuellos de botella**
- **Memoria**: millones de gaussianas por escena; crece sin límite si no se poda.
- La densificación es heurística y sensible a hiperparámetros.
- Tensión geometría vs. apariencia: las gaussianas sobreajustan el color y pueden dar
  superficies "infladas" o flotantes; la precisión de tracking pura-GS aún va por detrás de los
  frontends clásicos/aprendidos en secuencias difíciles.
- Sigue exigiendo GPU dedicada; escenas grandes y cierres de bucle son investigación activa.

---

## 4. Tabla comparativa

| Criterio | ORB-SLAM3 | DSO | DROID-SLAM | TartanVO | NeRF-SLAM | 3DGS-SLAM |
|---|---|---|---|---|---|---|
| Señal | características dispersas | intensidades (gradiente alto) | flujo denso aprendido | flujo→pose aprendido | render volumétrico | rasterización de gaussianas |
| Mapa | nube dispersa | semi-denso | profundidad densa/keyframe | ninguno | campo implícito denso | gaussianas explícitas densas |
| Precisión | ★★★★ | ★★★★ (corto plazo) | ★★★★★ | ★★★ | ★★★ | ★★★★ |
| Robustez (textura pobre, blur) | ★★ | ★★★ | ★★★★★ | ★★★★ | ★★ | ★★★ |
| Cierre de bucle | ✔ maduro | ✘ (LDSO) | ✔ (BA global) | ✘ | problema abierto | submapas (activo) |
| Hardware | CPU | CPU | GPU grande | GPU media | GPU muy grande | GPU grande |
| Tiempo real | ✔ | ✔ | ~ (GPU potente) | ✔ (GPU) | ✘/~ | ~ |
| Madurez | ★★★★★ | ★★★★ | ★★★★ | ★★★ | ★★ | ★★★ (explosión 2024-25) |
| Utilidad del mapa para navegación densa | baja | media | media-alta | — | alta | alta |

## 5. Conclusión: la tesis híbrida de este repositorio

Ninguna familia domina en todo. La convergencia del campo (Photo-SLAM, NeRF-SLAM, SplaTAM y
sucesores) apunta a una **arquitectura de tres capas con contratos claros**:

1. **Frontend de tracking rápido y robusto** — clásico (ORB/KLT) hoy, con piezas aprendidas
   (SuperPoint/LightGlue, o parches estilo DPVO) intercambiables mañana. Debe correr en
   tiempo real y degradarse con gracia.
2. **Backend de optimización sobre grafos de factores** — la maquinaria probabilística clásica
   (GTSAM/g2o) sigue siendo insustituible para consistencia global, cierres de bucle y fusión
   multisensor (IMU, GPS, odometría de ruedas).
3. **Mapeo denso intercambiable** — nube dispersa para empezar, Gaussian Splatting como objetivo
   (explícito, rápido, editable), campos neurales como alternativa. El mapa consume keyframes
   con pose optimizada; nunca bloquea al tracking.

Esa separación es exactamente la que codifica la estructura de carpetas del repo
(ver [02_arquitectura.md](02_arquitectura.md)).

## Bibliografía comentada

- Campos, Elvira, Gómez-Rodríguez, Montiel, Tardós. **ORB-SLAM3**. IEEE T-RO, 2021. *El sistema clásico de referencia.*
- Engel, Koltun, Cremers. **Direct Sparse Odometry (DSO)**. IEEE TPAMI, 2018. *La formulación directa canónica.*
- Teed, Deng. **DROID-SLAM**. NeurIPS, 2021. *Optimización diferenciable end-to-end; robustez récord.*
- Teed, Lipson, Deng. **Deep Patch Visual Odometry (DPVO)**. NeurIPS, 2023. *DROID eficiente con parches dispersos.*
- Wang, Hu, Scherer. **TartanVO**. CoRL, 2020. *VO aprendida que generaliza entre datasets.*
- Sucar, Liu, Ortiz, Davison. **iMAP**. ICCV, 2021. *Primer SLAM con mapa neural implícito.*
- Zhu et al. **NICE-SLAM**. CVPR, 2022. *Rejillas jerárquicas: campos neurales escalables a escenas.*
- Rosinol, Leonard, Carlone. **NeRF-SLAM**. IROS, 2023. *Híbrido DROID + Instant-NGP con incertidumbre.*
- Kerbl, Kopanas, Leimkühler, Drettakis. **3D Gaussian Splatting**. SIGGRAPH, 2023. *La representación que cambió el mapeo.*
- Keetha et al. **SplaTAM**. CVPR, 2024. *SLAM RGB-D sobre gaussianas con densificación guiada por siluetas.*
- Matsuki, Murai, Kelly, Davison. **Gaussian Splatting SLAM (MonoGS)**. CVPR, 2024. *El caso monocular directo sobre gaussianas.*
- Huang, Li, Hui, Fu. **Photo-SLAM**. CVPR, 2024. *Tracking ORB-SLAM3 + mapeo GS: la referencia de nuestra arquitectura.*
- Murai, Dexheimer, Davison. **MASt3R-SLAM**. CVPR, 2025. *SLAM sobre priors fundacionales de dos vistas, sin calibración fina.*
- DeTone, Malisiewicz, Rabinovich. **SuperPoint**. CVPRW, 2018; Lindenberger et al. **LightGlue**. ICCV, 2023. *Frontend aprendido enchufable en pipelines clásicos.*
