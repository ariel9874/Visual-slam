# Núcleo C++ de vslam — ruta de rendimiento (ver docs/02_arquitectura.md §5).
#
# v0.5: primer módulo compilado, `vslam_cpp` (pybind11), con las rutas
# calientes que señaló el PERFILADOR (regla 3 del repo: solo se porta lo
# medido — docs/05 §7 y lección 30):
#   - guided_match  (matching guiado por reproyección: era el 37% del frame)
# El BA va por GTSAM (vslam/backend/gtsam_ba.py), no por aquí.
#
# Compilar (Windows, generador Visual Studio; en Linux basta Ninja/Make).
# PY = el python del entorno donde vayas a importar el módulo (p. ej. el de tu
# venv/conda: `python -c "import sys; print(sys.executable)"`):
#   PY=/ruta/a/tu/entorno/python
#   cmake -S cpp -B cpp/build -A x64 \
#         -DPython3_EXECUTABLE=$PY \
#         -Dpybind11_DIR=$($PY -c "import pybind11; print(pybind11.get_cmake_dir())")
#   cmake --build cpp/build --config Release
# El .pyd/.so resultante queda en la RAÍZ del repo (import vslam_cpp; está en
# .gitignore — cada máquina compila el suyo).

cmake_minimum_required(VERSION 3.16)
project(vslam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Contratos de datos compartidos (header-only).
add_library(vslam_core INTERFACE)
target_include_directories(vslam_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── módulo pybind11 (v0.5) ────────────────────────────────────────────────────
option(VSLAM_BUILD_PYTHON "Compilar el módulo pybind11 vslam_cpp" ON)
if(VSLAM_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG)   # del pip del env: -Dpybind11_DIR=...
  if(pybind11_FOUND)
    pybind11_add_module(vslam_cpp src/guided_match.cpp)
    target_link_libraries(vslam_cpp PRIVATE vslam_core)
    # Dejar el binario en la raíz del repo: los ejemplos/tests ya la tienen en
    # sys.path, así `import vslam_cpp` funciona sin instalar nada.
    set_target_properties(vslam_cpp PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
      LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/.."
      LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/..")
  else()
    message(STATUS "pybind11 no encontrado: se omite vslam_cpp (solo headers)")
  endif()
endif()
