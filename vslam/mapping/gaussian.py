"""GaussianSplattingMapper: el mapa denso de v0.7 detrás de `MapperBase`.

La tesis de la arquitectura (docs/01 §3.2): cambiar la REPRESENTACIÓN del mapa
—de nube dispersa a gaussianas 3D— sin tocar frontend ni backend. Este mapper
cumple el mismo contrato que `SparsePointMapper`:

- `integrate_keyframe(frame)`: guarda la imagen + pose del keyframe y lo encola
  como VISTA de supervisión. Barato (no optimiza aquí): el trabajo pesado corre
  fuera, en `optimize()`, que el hilo de mapeo (o el ejemplo) llama con el
  presupuesto que sobre — así el tracking nunca se bloquea (contrato de base.py).
- `add_points(pos, color, anchor_kf)`: SIEMBRA gaussianas desde la nube dispersa
  del tracker (media = punto 3D, color = muestra de la imagen). El SLAM
  geométrico ya nos da la estructura; 3DGS solo la vuelve foto-realista.
- `optimize(iters)`: descenso de gradiente sobre TODOS los parámetros de las
  gaussianas minimizando el error de re-render contra los keyframes guardados
  (gaussian_render.render, diferenciable).
- `update_poses(poses)`: tras un cierre de bucle el backend movió las poses;
  cada gaussiana está ANCLADA a un keyframe y se re-ancla RÍGIDAMENTE con el
  delta de su ancla (T_nuevo·T_viejo⁻¹) — la generalización densa del
  re-anclaje de la nube dispersa (SparsePointMapper.update_poses).

─── La matemática del re-anclaje rígido ──────────────────────────────────────
Una gaussiana anclada al keyframe k se mueve con él: si su pose pasa de T a T',
el delta en el mundo es D = T'·T⁻¹ (rígido, SE(3)). La media va con la parte
afín, μ' = R_D·μ + t_D, y la ORIENTACIÓN de la covarianza rota con R_D (la
covarianza Σ = R·S·Sᵀ·Rᵀ hereda R' = R_D·R). Las escalas y el color NO cambian:
el cierre de bucle re-coloca el submapa, no re-ilumina la escena.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import numpy as np
import torch

from vslam.core.frame import Frame
from vslam.mapping.base import MapperBase
# El backend de render (referencia o gsplat) se importa en __init__ según se elija.


def _se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """exp: se(3) → SE(3), diferenciable. `xi = [ρ, ω]` (traslación, rotación),
    la convención de tangente del proyecto (docs/02; aquí sin λ: el mapa métrico
    fija la escala). Rodrigues con eje unitario k = ω/θ:

        R = I + sin θ·K + (1−cos θ)·K²,      K = [k]ₓ
        V = I + ((1−cos θ)/θ)·K + ((θ−sin θ)/θ)·K²,   t = V·ρ

    θ se acota inferiormente (1e-8) para que el backward sea estable en el
    origen (sin(θ)/θ → 1 numéricamente, sin ramas que produzcan NaN)."""
    rho, omega = xi[:3], xi[3:]
    theta = omega.norm().clamp(min=1e-8)
    k = omega / theta
    K = torch.zeros(3, 3, device=xi.device, dtype=xi.dtype)
    K[0, 1], K[0, 2] = -k[2], k[1]
    K[1, 0], K[1, 2] = k[2], -k[0]
    K[2, 0], K[2, 1] = -k[1], k[0]
    I = torch.eye(3, device=xi.device, dtype=xi.dtype)
    K2 = K @ K
    R = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * K2
    V = I + ((1.0 - torch.cos(theta)) / theta) * K \
        + ((theta - torch.sin(theta)) / theta) * K2
    T = torch.eye(4, device=xi.device, dtype=xi.dtype)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Rotación (3, 3) → cuaternión [w, x, y, z]. Para rotar las gaussianas en
    update_poses (fórmula estándar de la traza, rama numéricamente estable)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = torch.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return torch.stack([w, x, y, z])


class GaussianSplattingMapper(MapperBase):
    """Mapper 3DGS. Gris o color (C canales del keyframe); la matemática del
    render vive en gaussian_render.py."""

    def __init__(self, camera, device: Optional[str] = None,
                 init_scale: float = 0.05, backend: str = "reference") -> None:
        self.camera = camera
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.K = torch.tensor(camera.K, dtype=torch.float32, device=self.device)
        self.init_scale = init_scale
        # Backend de render, contrato idéntico (regla 3, atado por tests de
        # equivalencia): "reference" (denso, legible, O(N·H·W) — para tests y
        # docencia), "tiled" (PyTorch por tiles, memoria acotada — el de trabajo
        # en real) o "gsplat" (tiles + CUDA, tiempo real — pendiente por el
        # toolchain de Windows, ver docs/05 §7).
        if backend == "gsplat":
            from vslam.mapping.gaussian_render_gsplat import render as _r
        elif backend == "tiled":
            from vslam.mapping.gaussian_render_tiled import render as _r
        else:
            from vslam.mapping.gaussian_render import render as _r
        self._render_fn = _r
        self.backend = backend
        # Parámetros de las gaussianas (crecen con add_points). Se optimizan
        # activados: scales = exp(log_scales), opac = sigmoid, color = sigmoid.
        z3 = torch.zeros(0, 3, device=self.device)
        self._means = z3.clone()
        self._quats = torch.zeros(0, 4, device=self.device)
        self._log_scales = z3.clone()
        self._opacity = torch.zeros(0, device=self.device)          # logit
        self._colors = None                    # (N, C) logit; C (gris=1/RGB=3)
        #                                        lo fija el primer add_points
        self._anchor = torch.zeros(0, dtype=torch.long, device=self.device)
        # Keyframes de supervisión: id → {"T": (4,4), "image": (H,W,C)}.
        self._kfs: Dict[int, Dict[str, torch.Tensor]] = {}
        # Compensación de exposición por keyframe [log-ganancia, sesgo],
        # ajustada por optimize(exposure=True); vacía = sin compensar.
        self._exposure: Dict[int, torch.Tensor] = {}
        self._lock = threading.Lock()

    # ── contrato MapperBase ───────────────────────────────────────────────────

    def integrate_keyframe(self, keyframe: Frame) -> None:
        """Guarda la vista de supervisión (imagen + pose). Barato: no optimiza."""
        if keyframe.image is None:
            return
        img = np.asarray(keyframe.image)
        if img.ndim == 2:
            img = img[..., None]                     # gris → (H, W, 1)
        t = torch.tensor(img, dtype=torch.float32, device=self.device) / 255.0
        T = torch.tensor(keyframe.T_w_c, dtype=torch.float32, device=self.device)
        with self._lock:
            self._kfs[keyframe.frame_id] = {"T": T, "image": t}

    def update_poses(self, optimized_poses: Dict[int, np.ndarray]) -> None:
        """Re-ancla rígidamente cada submapa por el delta de su keyframe ancla
        (teoría arriba). Solo mueve las gaussianas cuya ancla cambió."""
        with self._lock:
            for kf_id, T_new_np in optimized_poses.items():
                kf = self._kfs.get(kf_id)
                if kf is None:
                    continue
                T_new = torch.tensor(T_new_np, dtype=torch.float32, device=self.device)
                D = T_new @ torch.inverse(kf["T"])       # delta rígido en el mundo
                mask = self._anchor == kf_id
                if mask.any():
                    R_D, t_D = D[:3, :3], D[:3, 3]
                    self._means[mask] = (R_D @ self._means[mask].T).T + t_D
                    q_D = _rotmat_to_quat(R_D)
                    self._quats[mask] = _quat_mul(q_D, self._quats[mask])
                kf["T"] = T_new

    def get_map(self) -> Any:
        """Exporta las gaussianas activadas como arrays NumPy (viz/eval)."""
        with torch.no_grad():
            return {
                "means": self._means.cpu().numpy(),
                "scales": torch.exp(self._log_scales).cpu().numpy(),
                "quats": self._quats.cpu().numpy(),
                "opacity": torch.sigmoid(self._opacity).cpu().numpy(),
                "colors": torch.sigmoid(self._colors).cpu().numpy(),
            }

    # ── siembra desde la nube dispersa ────────────────────────────────────────

    def add_points(self, positions: np.ndarray, colors: np.ndarray,
                   anchor_kf_id, scales: Optional[np.ndarray] = None) -> None:
        """Añade gaussianas: media = punto 3D, color = muestra de imagen [0,1].
        Orientación identidad, opacidad media-alta. `anchor_kf_id` puede ser un
        entero (todas al mismo keyframe) o un array (N,) con el ancla de cada
        punto — el mapa real reparte por muchos KFs.

        `scales`: escala inicial isótropa POR PUNTO (N,), en metros. Si None se
        usa `self.init_scale` uniforme. La correcta es la huella de la celda de
        siembra en el mundo (~step·z/fx, como el vecino más cercano del 3DGS
        original): una escala fija sobredimensionada emborrona todo — medido en
        fr1/desk full-res: 3 cm fijos ≈ 16 px por gaussiana → 15.5 dB de techo."""
        p = torch.tensor(np.asarray(positions), dtype=torch.float32, device=self.device)
        n = len(p)
        if n == 0:
            return
        c = torch.tensor(np.asarray(colors), dtype=torch.float32, device=self.device)
        c = c.reshape(n, -1).clamp(1e-4, 1 - 1e-4)
        col_logit = torch.log(c / (1 - c))               # inversa de sigmoid
        quats = torch.zeros(n, 4, device=self.device); quats[:, 0] = 1.0
        anchors = np.broadcast_to(np.asarray(anchor_kf_id), (n,))
        anchor_t = torch.tensor(anchors, dtype=torch.long, device=self.device)
        if scales is None:
            log_s = torch.full((n, 3), float(np.log(self.init_scale)),
                               device=self.device)
        else:
            s = torch.tensor(np.asarray(scales, dtype=np.float32),
                             device=self.device).clamp(min=1e-4)
            log_s = torch.log(s)[:, None].expand(n, 3).contiguous()
        with self._lock:
            self._means = torch.cat([self._means, p])
            self._quats = torch.cat([self._quats, quats])
            self._log_scales = torch.cat([self._log_scales, log_s])
            self._opacity = torch.cat([
                self._opacity, torch.full((n,), 2.0, device=self.device)])   # sigmoid(2)≈0.88
            self._colors = col_logit if self._colors is None \
                else torch.cat([self._colors, col_logit])
            self._anchor = torch.cat([self._anchor, anchor_t])

    # ── optimización (renderiza y compara) ────────────────────────────────────

    def _densify_and_prune(self, grad_acc: torch.Tensor, cnt: int,
                           max_gaussians: int) -> None:
        """Densificación + poda (el motor de detalle del 3DGS original).

        ─── La matemática ───
        El gradiente de posición acumulado |∂L/∂μ| señala DÓNDE el mapa no puede
        explicar la imagen moviendo lo que ya hay: ahí hace falta MÁS geometría.
        Sobre el 5% superior: si la gaussiana es GRANDE (escala > 2·mediana) se
        DIVIDE (sub-reconstrucción: cubría de más → dos hijas a escala /1.6,
        muestreadas dentro de la madre); si es pequeña se CLONA (over-fit local:
        falta densidad → copia perturbada ~0.3σ). La PODA elimina opacidades
        <0.05: gaussianas que el blending ya apagó (no aportan, solo cuestan).
        Se llama bajo el lock; el optimizador se reconstruye fuera (los momentos
        de Adam se reinician — aceptable a esta escala, anotado)."""
        with torch.no_grad():
            avg = grad_acc / max(cnt, 1)
            n = len(self._means)
            scales = torch.exp(self._log_scales).max(dim=-1).values
            if n < max_gaussians:
                high = avg > torch.quantile(avg, 0.95)
                big = scales > 2.0 * scales.median()
                split = high & big
                clone = high & ~big
                room = max_gaussians - n
                idx = torch.cat([torch.nonzero(split).squeeze(-1),
                                 torch.nonzero(clone).squeeze(-1)])[:room]
                if len(idx):
                    is_split = split[idx]
                    noise = torch.randn(len(idx), 3, device=self.device)
                    child_scale = torch.where(is_split[:, None],
                                              self._log_scales[idx] - float(np.log(1.6)),
                                              self._log_scales[idx])
                    offset = noise * torch.exp(child_scale) * \
                        torch.where(is_split[:, None], 1.0, 0.3)
                    self._means = torch.cat([self._means, self._means[idx] + offset])
                    self._quats = torch.cat([self._quats, self._quats[idx]])
                    self._log_scales[idx[is_split]] -= float(np.log(1.6))  # madre
                    self._log_scales = torch.cat([self._log_scales, child_scale])
                    self._opacity = torch.cat([self._opacity, self._opacity[idx]])
                    self._colors = torch.cat([self._colors, self._colors[idx]])
                    self._anchor = torch.cat([self._anchor, self._anchor[idx]])
            keep = torch.sigmoid(self._opacity) > 0.05          # poda
            self._means = self._means[keep].contiguous()
            self._quats = self._quats[keep].contiguous()
            self._log_scales = self._log_scales[keep].contiguous()
            self._opacity = self._opacity[keep].contiguous()
            self._colors = self._colors[keep].contiguous()
            self._anchor = self._anchor[keep].contiguous()

    def optimize(self, iters: int = 100, lr_scale: float = 1.0,
                 log_every: int = 0, refine_poses: bool = False,
                 exposure: bool = False, densify_every: int = 0,
                 max_gaussians: int = 500000, lr_decay: float = 0.01) -> float:
        """Ajusta las gaussianas contra los keyframes guardados. Devuelve el
        PSNR medio final sobre las vistas. Barato de reanudar (Adam efímero).

        `refine_poses`: además del mapa, optimiza un delta SE(3) POR KEYFRAME
        (T' = T·exp(ξ), ξ ∈ se(3) en el frame de cámara). El porqué (lección 41):
        la fusión fotométrica exige precisión SUB-PÍXEL, y el ATE de ~cm del SLAM
        son varios píxeles a 1 m — con poses discrepantes, el óptimo del mapa es
        un promedio BORROSO de las vistas (medido: 15.8 dB de techo en fr1/desk).
        Es el lazo de MonoGS/SplaTAM: el mapa denso devuelve corrección a las
        poses. Al terminar, el delta se hornea en kf["T"] (update_poses y
        mean_psnr ven la pose refinada).

        `exposure`: gana/sesgo afín POR KEYFRAME sobre el render (g·I+b): las
        cámaras de TUM llevan auto-exposición y el mismo punto cambia de gris
        entre vistas; sin compensar, ese residuo también se paga como blur.
        El PSNR posterior se mide con la compensación aplicada (estándar en las
        evaluaciones con exposición variable, p.ej. NeRF-W)."""
        with self._lock:
            kf_items = list(self._kfs.items())
            n = len(self._means)
        if not kf_items or n == 0:
            return 0.0
        xi: Dict[int, torch.Tensor] = {}
        ab: Dict[int, torch.Tensor] = {}
        if refine_poses:
            xi = {k: torch.zeros(6, device=self.device, requires_grad=True)
                  for k, _ in kf_items}
        if exposure:
            ab = {k: torch.zeros(2, device=self.device, requires_grad=True)
                  for k, _ in kf_items}

        def _make_opt() -> torch.optim.Adam:
            """(Re)construye Adam sobre los tensores ACTUALES — tras densificar,
            los tensores son otros y los momentos se reinician (aceptable)."""
            for pm in (self._means, self._quats, self._log_scales,
                       self._opacity, self._colors):
                pm.requires_grad_(True)
            gs = [
                {"params": [self._means], "lr": 0.002 * lr_scale},
                {"params": [self._quats], "lr": 0.01 * lr_scale},
                {"params": [self._log_scales], "lr": 0.01 * lr_scale},
                {"params": [self._opacity], "lr": 0.05 * lr_scale},
                {"params": [self._colors], "lr": 0.02 * lr_scale},
            ]
            if xi:
                gs.append({"params": list(xi.values()), "lr": 1e-3 * lr_scale})
            if ab:
                gs.append({"params": list(ab.values()), "lr": 1e-2 * lr_scale})
            return torch.optim.Adam(gs)

        opt = _make_opt()
        grad_acc = torch.zeros(len(self._means), device=self.device)
        acc_cnt = 0
        H, W, _ = kf_items[0][1]["image"].shape
        g = torch.Generator(device="cpu").manual_seed(0)
        # Decay exponencial del lr de las MEDIAS (×lr_decay al final, como el
        # 3DGS original): al principio exploran posición, al final se ASIENTAN.
        # Sin decay, el paso fijo mantiene un jitter perpetuo que se paga como
        # blur (medido en fr1/desk: 16.4 → 20.9 dB al añadir decay+presupuesto).
        # En vivo (hilo, chunks cortos) se pasa lr_decay=1.0: el schedule por
        # chunk no tiene sentido — el asentamiento lo da el paso del tiempo.
        lr_means0 = 0.002 * lr_scale
        import time
        t0 = time.perf_counter()
        for it in range(iters):
            opt.param_groups[0]["lr"] = lr_means0 * (lr_decay ** (it / max(iters - 1, 1)))
            kf_id, kf = kf_items[int(torch.randint(len(kf_items), (1,), generator=g))]
            opt.zero_grad()
            T = kf["T"] @ _se3_exp(xi[kf_id]) if refine_poses else kf["T"]
            img, _ = self._render(T, H, W)
            if exposure:
                img = img * torch.exp(ab[kf_id][0]) + ab[kf_id][1]
            loss = torch.abs(img - kf["image"]).mean()
            loss.backward()
            opt.step()
            if densify_every:
                with torch.no_grad():
                    grad_acc += self._means.grad.norm(dim=-1)
                acc_cnt += 1
                # Densificar solo en la PRIMERA MITAD (como el 3DGS original):
                # la segunda mitad asienta el mapa con población estable.
                if (it + 1) % densify_every == 0 and it < iters // 2:
                    with self._lock:
                        self._densify_and_prune(grad_acc, acc_cnt, max_gaussians)
                    opt = _make_opt()
                    grad_acc = torch.zeros(len(self._means), device=self.device)
                    acc_cnt = 0
            if log_every and (it + 1) % log_every == 0:
                dt = time.perf_counter() - t0
                eta = dt / (it + 1) * (iters - it - 1)
                print(f"    iter {it+1}/{iters} | L1 {float(loss):.4f} | "
                      f"N {len(self._means)} | {dt/(it+1)*1000:.0f} ms/iter | "
                      f"ETA {eta:.0f}s", flush=True)
        for pm in (self._means, self._quats, self._log_scales,
                   self._opacity, self._colors):
            pm.requires_grad_(False)
        with torch.no_grad():
            if refine_poses:                     # hornear el delta en la pose
                with self._lock:
                    for kf_id, kf in kf_items:
                        kf["T"] = kf["T"] @ _se3_exp(xi[kf_id])
            if exposure:                         # recordar para mean_psnr
                self._exposure = {k: v.detach() for k, v in ab.items()}
        return self.mean_psnr()

    def _render(self, T_w_c: torch.Tensor, H: int, W: int):
        return self._render_fn(self._means, self._quats, torch.exp(self._log_scales),
                               torch.sigmoid(self._opacity), torch.sigmoid(self._colors),
                               T_w_c, self.K, H, W)

    def render_view(self, T_w_c: np.ndarray, height: int, width: int) -> np.ndarray:
        """Re-render desde una pose arbitraria (para evaluación/visualización)."""
        T = torch.tensor(T_w_c, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            img, _ = self._render(T, height, width)
        return img.clamp(0, 1).cpu().numpy()

    def mean_psnr(self) -> float:
        """PSNR medio de re-render sobre los keyframes de supervisión (criterio).
        Si optimize() ajustó exposición, se aplica antes de comparar."""
        vals = self.psnr_per_kf()
        return float(np.mean(list(vals.values()))) if vals else 0.0

    def psnr_per_kf(self) -> Dict[int, float]:
        """PSNR por keyframe — el diagnóstico de INCONSISTENCIA multi-vista:
        una dispersión grande (p.ej. 13↔19 dB) delata poses/exposición
        discrepantes, no falta de capacidad del mapa (lección 41)."""
        from vslam.mapping.gaussian_render import psnr
        with self._lock:
            kf_items = list(self._kfs.items())
        out: Dict[int, float] = {}
        with torch.no_grad():
            for kf_id, kf in kf_items:
                H, W, _ = kf["image"].shape
                img, _ = self._render(kf["T"], H, W)
                e = self._exposure.get(kf_id)
                if e is not None:
                    img = img * torch.exp(e[0]) + e[1]
                out[kf_id] = psnr(img.clamp(0, 1), kf["image"])
        return out


def _quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Producto de cuaterniones q ∘ r (q: (4,) aplicado a un lote r: (N, 4))."""
    w0, x0, y0, z0 = q[0], q[1], q[2], q[3]
    w1, x1, y1, z1 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    return torch.stack([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ], dim=-1)
