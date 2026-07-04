# Detectores de Características y Matchers: Análisis y Selección

> **Objetivo**: catálogo razonado de los detectores/descriptores y emparejadores que el
> frontend soporta (o soportará) como configuración intercambiable. Para cada uno: la idea
> matemática central, beneficios, cuellos de botella y cuándo elegirlo.
>
> Uso: `python examples/01_monocular_vo.py --detector akaze --matcher flann ...`
> Comparativa automática: `python scripts/benchmark_frontends.py`

## 1. Detectores / descriptores clásicos (CPU, incluidos en OpenCV)

### 1.1 FAST / AGAST — detector puro
**Idea**: test de segmento — p es esquina si ≥ 9 píxeles contiguos del círculo de Bresenham
de radio 3 son todos más claros/oscuros que `I(p) ± τ` (la matemática completa está en
[vslam/frontend/features.py](../vslam/frontend/features.py)). AGAST optimiza el árbol de
decisión del test. **Beneficios**: el detector más rápido que existe (µs por imagen).
**Cuellos**: no produce descriptor, ni escala ni orientación; dispara en bordes con ruido.
**Cuándo**: como detector dentro de otros (ORB lo usa) o para tracking KLT donde no hace
falta descriptor.

### 1.2 Shi-Tomasi / GFTT (“Good Features to Track”) — detector puro
**Idea**: tensor de estructura del parche `M = Σ_w ∇I·∇Iᵀ`; esquina si el menor autovalor
`min(λ₁, λ₂) > umbral` — hay gradiente fuerte en DOS direcciones independientes, así que el
parche se puede localizar sin ambigüedad. Es el criterio óptimo para el tracking
Lucas-Kanade (que invierte exactamente esa matriz M). **Beneficios**: esquinas "trackeables"
por diseño; distribución espacial controlable (`minDistance`). **Cuellos**: sin descriptor,
sin escala. **Cuándo**: frontends KLT (v0.4) o combinado con un descriptor (combo
`gftt-orb` del registro, que además enseña la separación detector/descriptor).

### 1.3 ORB — el caballo de batalla (default del repo)
**Idea**: FAST multi-escala + orientación por centroide de intensidad + BRIEF rotado
(matemática detallada en `features.py`). **Beneficios**: ~µs/feature en CPU, descriptor
binario de 32 bytes, el estándar de los SLAM en tiempo real (ORB-SLAM). **Cuellos**:
sufre con blur fuerte y cambios grandes de punto de vista; tiende a concentrar puntos en
zonas de mucha textura (ORB-SLAM lo mitiga con rejilla de reparto). **Cuándo**: default
en CPU; siempre como línea base.

### 1.4 BRISK
**Idea**: patrón de muestreo en anillos concéntricos con suavizado proporcional al radio.
Los pares de LARGA distancia estiman la orientación global del parche
(`g = (1/L)·Σ (I(p_j) − I(p_i))·(p_j − p_i)/‖p_j − p_i‖²`); los de CORTA distancia generan
los bits del descriptor (512 bits). Detección con pirámide de escalas continua (interpola
la escala entre niveles). **Beneficios**: binario, estimación de escala más fina que ORB.
**Cuellos**: 2-3× más lento que ORB; descriptor más largo (64 bytes). **Cuándo**: cuando la
escala cambia mucho entre frames (acercamientos) y ORB pierde repetibilidad.

### 1.5 AKAZE
**Idea**: espacio de escalas NO LINEAL por difusión anisotrópica:
`∂L/∂t = div( g(|∇L|)·∇L )`, donde la conductividad `g` se anula en bordes → el suavizado
progresivo difumina el interior de las regiones pero RESPETA los contornos (el espacio
gaussiano de SIFT difumina todo por igual). Detección por el determinante del hessiano en
ese espacio; descriptor M-LDB (binario, comparaciones de medias e derivadas en subceldas,
rotado). **Beneficios**: repetibilidad claramente superior a ORB con blur/iluminación;
binario; libre. **Cuellos**: 2-4× más lento que ORB (la difusión FED es iterativa).
**Cuándo**: el mejor trade-off clásico calidad/costo cuando ORB no alcanza; buen default
para RGB-D/interiores.

### 1.6 SIFT
**Idea**: extremos locales en el espacio-escala `(x, y, σ)` de la Diferencia de Gaussianas
`DoG(x, σ) = L(x, kσ) − L(x, σ)` (aproximación del laplaciano normalizado `σ²∇²G`, que es
invariante a escala). Orientación por histograma de gradientes del parche; descriptor =
rejilla 4×4 de histogramas de 8 orientaciones → 128 floats, normalizado (invarianza afín a
iluminación). **Beneficios**: 25 años siendo la referencia de robustez a escala, rotación
e iluminación; subpíxel fino. **Cuellos**: float 128D → matching L2 caro; ~10× ORB en CPU;
sin GPU no es de tiempo real con miles de puntos. **Cuándo**: mapeo offline, relocalización,
o como “juez” de calidad frente a detectores rápidos.

### 1.7 KAZE
**Idea**: mismo espacio no lineal que AKAZE pero con descriptor flotante M-SURF (sumas de
respuestas de Haar por subregiones). **Beneficios**: algo más preciso que AKAZE.
**Cuellos**: el más lento del catálogo clásico (difusión + descriptor float). **Cuándo**:
casi nunca en línea; útil en benchmarks como techo de la familia KAZE.

### 1.8 BRIEF — descriptor puro
**Idea**: 256 comparaciones de pares de píxeles → bits (detalle en `features.py`). Sin
detector, sin orientación ni escala propias: las hereda del detector que lo alimente.
**Cuándo**: la pieza pedagógica que muestra por qué importa el EMPAREJAMIENTO
detector↔descriptor (nuestro combo `gftt-orb` existe para ese experimento).

## 2. Detectores / descriptores aprendidos (GPU, extra opcional `[deep]`)

### 2.1 SuperPoint (2018)
**Idea**: una CNN con dos cabezas sobre un encoder compartido — detector (clasifica cada
celda de 8×8 en 65 clases: 64 posiciones + “sin punto”) y descriptor (256D, interpolado).
Lo notable es el entrenamiento AUTO-SUPERVISADO: se pre-entrena en esquinas sintéticas
(MagicPoint) y se generaliza a imágenes reales por *Homographic Adaptation* — agregar las
detecciones de la misma imagen bajo decenas de homografías aleatorias crea el “ground
truth” de esquinas. **Beneficios**: repetibilidad muy superior a ORB con blur, poca luz y
cambios fuertes de iluminación; puntos distribuidos uniformemente. **Cuellos**: GPU para
tiempo real; 256 floats/punto; **licencia de los pesos oficiales: solo investigación
(Magic Leap)** — vigilar en uso comercial. **Cuándo**: el reemplazo directo de ORB cuando
hay GPU y las condiciones visuales son difíciles.

### 2.2 DISK (2020)
**Idea**: entrenado con gradiente de política (REINFORCE): la “recompensa” son los matches
correctos tras el emparejamiento — optimiza directamente el objetivo final del pipeline,
no un proxy de repetibilidad. **Beneficios**: produce MUCHOS matches correctos y densos;
pesos con licencia permisiva; disponible en `kornia`. **Cuellos**: GPU; los puntos no son
“esquinas” clásicas (a veces incomoda a RANSAC con umbrales estrictos). **Cuándo**:
reconstrucción densa, escenas con textura débil.

### 2.3 ALIKED (2023)
**Idea**: descriptores con convoluciones DEFORMABLES: el soporte del descriptor se adapta
a la geometría local (aproxima invarianza afín aprendida) en una red diseñada ligera.
**Beneficios**: el mejor ratio calidad/latencia de los aprendidos; apto para GPU modesta
o edge. **Cuándo**: robots embebidos con aceleración (Jetson).

### 2.4 R2D2 (2019) — mención
**Idea**: separa dos mapas aprendidos: repetibilidad (¿este punto se re-detecta?) y
fiabilidad (¿este descriptor será discriminativo?) — aprende a NO describir texturas
repetitivas. Conceptualmente importante: el ratio test de Lowe, convertido en red.

## 3. Matchers

### 3.1 Fuerza bruta + ratio test (`ratio`, default)
Exhaustivo O(N·M) con la métrica del descriptor (Hamming/L2) + criterio de Lowe
(matemática en [vslam/frontend/matching.py](../vslam/frontend/matching.py)). Con N ~ 2000
es perfectamente viable en CPU. **El default correcto hasta que se demuestre lo contrario.**

### 3.2 Fuerza bruta + cross-check (`crosscheck`)
Acepta (i, j) solo si j es el mejor vecino de i **y** i es el mejor vecino de j (mutuo
mejor vecino). Más estricto y sin parámetro de ratio; suele dar menos matches pero muy
limpios. Alternativa al ratio test, no complemento (OpenCV no permite ambos con knn).

### 3.3 FLANN (`flann`) — vecinos aproximados
Para descriptores float: KD-trees aleatorizados (varios árboles, backtracking acotado por
`checks` — más checks = más exacto y lento). Para binarios: LSH multi-probe — funciones
hash que muestrean subconjuntos de bits, de modo que colisionar ≈ estar cerca en Hamming.
**Beneficio**: sub-lineal, importa con >5k features o matching contra mapas grandes
(relocalización). **Cuello**: aproximado — puede perder el vecino verdadero; con 2k
features NO gana nada frente a fuerza bruta.

### 3.4 SuperGlue (2020) — aprendido
**Idea**: red de grafos con auto-atención (contexto dentro de la imagen) y cross-atención
(entre imágenes) sobre keypoints + descriptores + posiciones; la asignación final se
resuelve como TRANSPORTE ÓPTIMO (iteraciones de Sinkhorn con un “dustbin” para puntos sin
pareja). Resuelve ambigüedades por CONTEXTO GLOBAL: dos ventanas idénticas se distinguen
por lo que las rodea — exactamente lo que el ratio test no puede hacer. **Cuellos**:
atención O(N²), GPU obligada, **licencia no comercial**. **Cuándo**: relocalización y
cierre de bucle con cambios extremos de punto de vista.

### 3.5 LightGlue (2023) — aprendido, el sucesor práctico
**Idea**: SuperGlue re-diseñado adaptativo: capas con *early-exit* (si la confianza es alta
tras pocas capas, para), poda de puntos no emparejables, y truco de posiciones rotatorias.
2-10× más rápido con calidad igual o mejor. **Licencia Apache-2.0** (sí comercial).
**Cuándo**: el matcher aprendido a integrar primero (nuestro adaptador `lightglue`).

### 3.6 LoFTR (2021) — mención, detector-free
Matching DENSO coarse-to-fine con transformers: no detecta puntos, empareja rejillas de
características y refina a subpíxel. Brilla donde NO HAY esquinas (paredes lisas,
texturas débiles) — justo el modo de fallo de todo lo anterior. Costo: GPU y latencia.

## 4. Tabla de decisión

| Config | Descriptor | Costo CPU | GPU | Robustez blur/luz | Licencia | Estado en el repo |
|---|---|---|---|---|---|---|
| `orb` (default) | binario 32B | ★ | no | ★★ | BSD | ✔ implementado |
| `gftt-orb` | binario 32B | ★ | no | ★★ | BSD | ✔ implementado (didáctico) |
| `brisk` | binario 64B | ★★ | no | ★★★ | BSD | ✔ implementado |
| `akaze` | binario 61B | ★★★ | no | ★★★★ | BSD | ✔ implementado |
| `sift` | float 128D | ★★★★ | no | ★★★★ | libre (patente expiró) | ✔ implementado |
| `kaze` | float 64D | ★★★★★ | no | ★★★★ | BSD | ✔ implementado |
| `superpoint` | float 256D | — | sí | ★★★★★ | pesos no comerciales | ⚠ adaptador (`[deep]`) |
| `disk` | float 128D | — | sí | ★★★★★ | permisiva | ⚠ adaptador (`[deep]`) |
| matcher `ratio` | — | ★ | no | — | — | ✔ default |
| matcher `crosscheck` | — | ★ | no | — | — | ✔ implementado |
| matcher `flann` | — | ★ (sub-lineal) | no | — | BSD | ✔ implementado |
| matcher `lightglue` | — | — | sí | ★★★★★ | Apache-2.0 | ⚠ adaptador (`[deep]`) |

**Reglas rápidas de selección**
- CPU y tiempo real → `orb + ratio`. Si falla con blur/luz → `akaze + ratio` (paga 2-4×).
- GPU disponible y condiciones duras → `superpoint + lightglue` (vigilar licencia de pesos)
  o `disk + lightglue` (licencias limpias).
- Matching contra mapas grandes (relocalización, v0.3) → `flann`.
- Nada de lo anterior funciona (pared lisa) → LoFTR… o admitir que necesitas un método
  directo/denso (docs/01 §1.2 y §3).

**Nota de ingeniería**: cambiar de detector NO corrige la deriva de escala del ejemplo 01
— eso lo corrige el tracking 3D-2D (PnP, v0.2). El detector decide cuándo el tracking
*sobrevive*; el backend decide cuánto *deriva*. Por eso este catálogo llega junto con
`scripts/benchmark_frontends.py`: para medir, no para creer.
