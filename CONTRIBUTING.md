# Contribuir a Visual-SLAM

Este repo es educativo con arquitectura seria. Las reglas que lo mantienen así
(los porqués, en docs/05):

1. **Nada se fusiona sin ejecutarse.** Tests (`python tests/test_X.py`, todos
   ejecutables sueltos) + el ejemplo afectado con ATE/PSNR contra ground truth.
   Los números de referencia están en docs/05 §3.2 — si tu cambio los mueve,
   la PR debe decir por qué y aportar la medición.
2. **Cada decisión con su medición.** Un umbral nuevo exige un barrido; un
   enfoque descartado se documenta con números (lecciones de docs/05 §5).
3. **La matemática va EN el código**: bloques `─── La matemática ───` en
   español; identificadores en inglés.
4. **Referencia legible + gemela rápida.** Lo acelerado (C++/GTSAM/gsplat) va
   atado a su referencia (NumPy/PyTorch) por un test de equivalencia.
5. **La cáscara no contamina.** El núcleo `vslam/` no importa ROS ni torch en
   el import raíz; las perillas se exponen por `vslam/config.py`, no
   hardcodeadas en los llamadores.
6. **Convenciones fijas**: `T_w_c` (cámara→mundo, ejes ópticos OpenCV),
   tangente `[ρ, ω, λ]`, formato TUM. Las interfaces de docs/02 no se rompen
   (API congelada en `vslam/__init__.py.__all__`).

Flujo: issue → rama → PR con la medición pegada en la descripción.
Licencia MIT: al contribuir aceptas que tu código se publique bajo ella.
