# ─────────────────────────────────────────────────────────────────────────────
# Contenedor de pruebas: ROS 2 + el núcleo Python de Visual-SLAM.
#
# El núcleo `vslam/` es Python puro (numpy + opencv) y NO importa ROS (ver
# ros2/README.md): este contenedor da un entorno donde conviven el paquete y
# un ROS 2 completo, para prototipar los wrappers `vslam_ros`/`vslam_msgs`
# (v0.6/v0.8 de la hoja de ruta) y correr los tests/ejemplos en Linux.
#
#   La estrategia de dependencias
#   Ubuntu 24.04 marca su Python como "externally managed" (PEP 668) y ROS se
#   compila contra la numpy de apt (1.26). Instalar numpy por pip la duplicaría
#   y rompería los bindings de ROS. Por eso TODAS las deps del núcleo entran por
#   apt (comparten ABI con ROS y respetan la restricción numpy<2 del proyecto),
#   y el paquete vslam se resuelve por PYTHONPATH (se monta como volumen), sin
#   pip install: para un paquete Python puro basta con que sea importable.
# ─────────────────────────────────────────────────────────────────────────────

# Distro por defecto: la imagen kilted-desktop-full que ya está en disco (build
# instantáneo). Para LTS (soporte hasta mayo 2029) construir con:
#   docker compose build --build-arg ROS_IMAGE=osrf/ros:jazzy-desktop-full
ARG ROS_IMAGE=osrf/ros:kilted-desktop-full
FROM ${ROS_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Dependencias del núcleo vslam + puentes ROS<->OpenCV, todo desde apt.
# ${ROS_DISTRO} lo define la imagen base (kilted / jazzy / ...).
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-numpy \
      python3-opencv \
      python3-matplotlib \
      python3-scipy \
      python3-pytest \
      python3-pip \
      ros-${ROS_DISTRO}-cv-bridge \
      ros-${ROS_DISTRO}-vision-opencv \
      ros-${ROS_DISTRO}-image-transport \
    && rm -rf /var/lib/apt/lists/*

# Backend headless para matplotlib: los ejemplos hacen savefig() a output/,
# no plt.show(). Evita errores de "no display" dentro del contenedor.
ENV MPLBACKEND=Agg

# El repo se monta en /workspace (ver docker-compose.yml). PYTHONPATH lo vuelve
# importable sin pip install (paquete Python puro).
ENV PYTHONPATH=/workspace
WORKDIR /workspace

# La imagen base ya trae ENTRYPOINT=/ros_entrypoint.sh, que hace
# `source /opt/ros/$ROS_DISTRO/setup.bash` antes de cada comando.
CMD ["bash"]
