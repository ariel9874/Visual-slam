# Núcleo C++ de vslam — ruta de rendimiento (ver docs/02_arquitectura.md §5).
#
# v0.1: solo los contratos de datos como biblioteca header-only (INTERFACE),
# para fijar convenciones. Los módulos compilados llegan en v0.4:
#   - vslam_frontend  (ORB/KLT con OpenCV)         -> espeja vslam/frontend
#   - vslam_backend   (grafo de poses con GTSAM)   -> espeja vslam/backend
#   - vslam_python    (bindings pybind11: `import vslam_cpp`)
#
# Compilar (cuando haya targets con fuentes):
#   cmake -S cpp -B cpp/build && cmake --build cpp/build

cmake_minimum_required(VERSION 3.16)
project(vslam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Contratos de datos compartidos (header-only por ahora).
add_library(vslam_core INTERFACE)
target_include_directories(vslam_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# TODO(v0.4): find_package(OpenCV REQUIRED), find_package(GTSAM), pybind11.
