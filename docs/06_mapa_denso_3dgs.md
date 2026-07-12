# 06 — El mapa denso: 3D Gaussian Splatting (v0.7)

> La tesis de la arquitectura (docs/01 §3.2), ejecutada: cambiar la
> REPRESENTACIÓN del mapa —de nube dispersa a gaussianas 3D foto-realistas—
> sin tocar frontend ni backend. Este documento es la visita guiada; la
> matemática completa vive EN el código (bloques `─── La matemática ───`) y
> cada número en las lecciones 39-42 de docs/05 §5.

## 1. La idea en una línea

El SLAM geométrico (v0.1-v0.6) ya da la ESTRUCTURA: poses métricas + nube
dispersa. El `GaussianSplattingMapper` la vuelve foto-realista por
**"renderiza y compara"**: el mapa es un conjunto de gaussianas 3D
{μ, Σ, α, c} y se ajusta por descenso de gradiente para re-sintetizar los
keyframes. Todo el render es diferenciable de punta a punta.

## 2. El rasterizador (vslam/mapping/gaussian_render.py)

Cadena: Σ = R·S·Sᵀ·Rᵀ (definida positiva por construcción) → proyección EWA
Σ' = J·W·Σ·Wᵀ·Jᵀ + dilatación → α-blending front-to-back por transmitancia
(producto acumulado exclusivo). Tres implementaciones con contrato IDÉNTICO
(la regla 3 del repo, como NumPy↔GTSAM en el BA):

| backend | qué es | para qué |
|---|---|---|
| `reference` | denso, O(N·H·W), legible | tests y docencia (OOM a ~8k gaussianas) |
| `tiled` | PyTorch por tiles, memoria acotada | equivalencia >40 dB; sigue lento (GIL/Python) |
| `gsplat` | kernels CUDA por tiles | el de trabajo: 15-20 ms/iter con 500k gaussianas |

**gsplat SOLO corre en Docker** (`docker/Dockerfile.gsplat`): en Windows
nativo el link es imposible — el mangling de nvcc (cudafe++) y MSVC diverge
en plantillas de ~28 argumentos (lección 40; probado con dos toolsets).

**La lección del medio píxel** (lección 40): el test de equivalencia
referencia↔gsplat daba 25 dB con el error en el NÚCLEO de las gaussianas.
Con una sola gaussiana, el pico de gsplat era exactamente exp(−0.5·0.5/σ²)
veces el nuestro: muestreábamos la rejilla en la esquina entera (i, j) y la
convención estándar es el CENTRO (i+0.5, j+0.5). Una línea de fix → 60 dB.
La gemela rápida auditó a la referencia legible: para eso existen los tests
de equivalencia (`test_pixel_center_convention` caza la regresión).

## 3. El mapper (vslam/mapping/gaussian.py)

- **Siembra** desde la profundidad de cada keyframe (retro-proyección de una
  rejilla): la nube dispersa del SLAM es demasiado rala (1968 pts → 15 dB,
  lección 39). Escala inicial POR PUNTO = huella de la celda, step·z/fx.
- **optimize()**: L1 contra keyframes aleatorios; Adam por grupos; decay del
  lr de medias (×0.01); delta SE(3) y exposición afín POR KEYFRAME opcionales;
  densificación (clone/split al 5% de mayor gradiente) y poda de opacidades.
- **update_poses()**: re-anclaje RÍGIDO por submapa tras un bucle
  (D = T'·T⁻¹; μ' = R_D·μ + t_D, la covarianza rota con R_D).

## 4. La cadena de ablaciones (lección 41) — cada hipótesis MEDIDA

| experimento (fr1/desk 640×480, gsplat) | PSNR |
|---|---|
| 300k gaussianas, siembra ingenua (3 cm fijos) | 15.5 dB |
| + escala por punto step·z/fx | 15.8 |
| + poses SE(3) + exposición por KF | 16.4 |
| + 30k iters con decay del lr de medias | **20.9** |
| + densificación/poda (→493k) | **21.0** |

Conclusiones: la capacidad NO era el cuello (dos veces medido); el factor
dominante fue el PRESUPUESTO+SCHEDULE; el refinamiento de poses es
obligatorio (1 cm de ATE ≈ 5 px a 1 m: la fusión fotométrica exige
sub-píxel); el RESIDUO es la consistencia fotométrica del dataset (motion
blur real: por-KF 17.1↔29.9 dB — en vistas bien condicionadas el mapa ya
toca 30). **Criterio recalibrado a paridad SOTA (≥21 dB; Photo-SLAM ~21,
SplaTAM ~22, MonoGS ~23-25; los >30 dB son de sintético) — CUMPLIDO: 21.0.**

## 5. En vivo, sin robar frames (lección 42)

El tercer hilo de ORB-SLAM en Python es un **PROCESO** (`DenseMappingProcess`):
el hilo paga +78% de latencia de tracking por el GIL aunque el mapa viva en
la GPU (cientos de llamadas Python→torch por iter); el proceso la deja en
+25% y el mapa recibe MÁS presupuesto. Las correcciones de pose viajan POR LA
COLA del worker (mutar el mapa desde fuera choca con un backward en vuelo —
carrera CUDA real, cazada y eliminada). Medido en examples/08: mismos
596/596 frames con el mapper ON, 80/80 KFs, 0 fallos.

## 6. Cómo correrlo

```bash
# offline (criterio): 21.0 dB
docker run --rm --gpus all -v "$PWD:/workspace" \
  -v gsplat-cache:/root/.cache/torch_extensions vslam-gsplat \
  python -u examples/07_gaussian_mapping.py \
    --root data/tum/rgbd_dataset_freiburg1_desk --backend gsplat --scale 1 \
    --max-points 300000 --iters 30000 --refine-poses --exposure --densify-every 500

# en vivo (tracker + mapper en proceso propio)
docker run ... python -u examples/08_live_dense_mapping.py \
    --root data/tum/rgbd_dataset_freiburg1_desk --dense --backend gsplat --scale 2
```

Tests: 15 (render 3, mapper 2, tiled 3, gsplat 4 —en Docker—, dense 3).
