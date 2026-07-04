// Contratos de datos en C++ — gemelos de vslam/core/frame.py.
// Regla del repo (docs/02_arquitectura.md): la clase Python y su gemela C++
// implementan el mismo contrato; los tests corren contra ambas.
//
// v0.1: solo tipos estándar para no imponer dependencias. En v0.4:
//   - pose      -> Sophus::SE3d (o gtsam::Pose3)
//   - image     -> cv::Mat
//   - keypoints -> std::vector<cv::KeyPoint>

#pragma once

#include <array>
#include <cstdint>

namespace vslam {

// Pose 4x4 en SE(3), row-major. Convención del repo: T_w_c transforma puntos
// del frame de la cámara al frame del mundo; ejes de cámara estilo OpenCV
// (+Z delante, +X derecha, +Y abajo).
struct Pose {
  std::array<double, 16> T_w_c{1, 0, 0, 0,
                               0, 1, 0, 0,
                               0, 0, 1, 0,
                               0, 0, 0, 1};
};

struct Frame {
  std::int64_t frame_id{0};
  double timestamp_s{0.0};
  Pose pose{};
  bool is_keyframe{false};
  // TODO(v0.4): imagen, keypoints y descriptores (tipos OpenCV).
};

}  // namespace vslam
