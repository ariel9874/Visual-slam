// vslam_cpp — núcleo C++ de las rutas calientes (v0.5).
//
// Primer módulo compilado: MATCHING GUIADO por reproyección — el perfilador lo
// señaló como el 37% del tiempo (frames TRACK a ~93 ms; el bucle Python por
// punto del mapa domina). Este archivo es el GEMELO EXACTO de
// PnPTracker._guided_match: misma matemática, mismos umbrales y la misma
// semántica de desempate que np.argmin (ante distancias iguales gana el índice
// MENOR) — verificado por tests/test_guided_match_cpp.py contra la referencia
// Python sobre las mismas entradas.
//
// ─── La matemática (idéntica a la referencia; racional en tracker.py) ───
//   T_c_w = T_pred⁻¹              (inversa de SE(3): [Rᵀ | −Rᵀ·t])
//   X_c   = R_c_w · X_w + t_c_w   (mapa al frame de cámara predicho)
//   (u,v) = (fx·x/z + cx, fy·y/z + cy)      con z > 1e-6 y dentro de imagen
//   candidato_i = argmin_{kp j : ‖kp_j − uv_i‖² ≤ r²} dist(desc_i, desc_j)
//   se acepta si dist ≤ max_dist; asignación GREEDY por distancia ascendente
//   (empates por (dist, i, j), como el sort de tuplas de Python) con cada
//   punto del mapa y cada keypoint usados a lo sumo una vez.
//
// Distancias: Hamming (descriptores binarios uint8, ORB — popcount del XOR)
// o L2 (float32, SuperPoint). El popcount usa una tabla por byte, la misma
// idea que _POPCOUNT8 del tracker.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

const std::array<std::uint8_t, 256>& popcount_table() {
  static const std::array<std::uint8_t, 256> table = [] {
    std::array<std::uint8_t, 256> t{};
    for (int i = 0; i < 256; ++i) {
      int c = 0, v = i;
      while (v) { c += v & 1; v >>= 1; }
      t[i] = static_cast<std::uint8_t>(c);
    }
    return t;
  }();
  return table;
}

inline double desc_distance(const std::uint8_t* a, const std::uint8_t* b, int d) {
  const auto& pop = popcount_table();
  int acc = 0;
  for (int k = 0; k < d; ++k) acc += pop[static_cast<std::uint8_t>(a[k] ^ b[k])];
  return static_cast<double>(acc);
}

inline double desc_distance(const float* a, const float* b, int d) {
  double acc = 0.0;
  for (int k = 0; k < d; ++k) {
    const double diff = static_cast<double>(a[k]) - static_cast<double>(b[k]);
    acc += diff * diff;
  }
  return std::sqrt(acc);
}

struct Cand {
  double dist;
  int i;   // índice en el mapa
  int j;   // índice del keypoint
  bool operator<(const Cand& o) const {
    if (dist != o.dist) return dist < o.dist;
    if (i != o.i) return i < o.i;
    return j < o.j;   // orden de tuplas (dist, i, j), como Python
  }
};

template <typename DescT>
std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>> guided_match(
    py::array_t<double, py::array::c_style | py::array::forcecast> kp_xy,    // (N,2)
    py::array_t<DescT, py::array::c_style | py::array::forcecast> desc,      // (N,D)
    py::array_t<double, py::array::c_style | py::array::forcecast> T_pred,   // (4,4)
    py::array_t<double, py::array::c_style | py::array::forcecast> map_pts,  // (M,3)
    py::array_t<DescT, py::array::c_style | py::array::forcecast> map_desc,  // (M,D)
    double fx, double fy, double cx, double cy,
    double width, double height, double radius_px, double max_dist) {
  const auto kp = kp_xy.template unchecked<2>();
  const auto de = desc.template unchecked<2>();
  const auto T = T_pred.template unchecked<2>();
  const auto mp = map_pts.template unchecked<2>();
  const auto md = map_desc.template unchecked<2>();
  const int N = static_cast<int>(kp.shape(0));
  const int M = static_cast<int>(mp.shape(0));
  const int D = static_cast<int>(de.shape(1));
  const double r2 = radius_px * radius_px;

  // T_c_w = T_pred⁻¹ (SE(3)): R_c_w = Rᵀ, t_c_w = −Rᵀ·t.
  double R[3][3], t[3];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) R[r][c] = T(c, r);   // transpuesta
  for (int r = 0; r < 3; ++r)
    t[r] = -(R[r][0] * T(0, 3) + R[r][1] * T(1, 3) + R[r][2] * T(2, 3));

  std::vector<Cand> cands;
  cands.reserve(static_cast<std::size_t>(M) / 2);
  for (int i = 0; i < M; ++i) {
    const double xw = mp(i, 0), yw = mp(i, 1), zw = mp(i, 2);
    const double xc = R[0][0] * xw + R[0][1] * yw + R[0][2] * zw + t[0];
    const double yc = R[1][0] * xw + R[1][1] * yw + R[1][2] * zw + t[1];
    const double zc = R[2][0] * xw + R[2][1] * yw + R[2][2] * zw + t[2];
    if (zc <= 1e-6) continue;
    const double u = fx * xc / zc + cx;
    const double v = fy * yc / zc + cy;
    if (u < 0.0 || u >= width || v < 0.0 || v >= height) continue;

    // argmin de distancia entre los keypoints dentro del radio; ante empate
    // gana el índice menor (semántica de np.argmin sobre el subconjunto).
    double best = -1.0;
    int best_j = -1;
    for (int j = 0; j < N; ++j) {
      const double dx = kp(j, 0) - u, dy = kp(j, 1) - v;
      if (dx * dx + dy * dy > r2) continue;
      const double dist = desc_distance(&md(i, 0), &de(j, 0), D);
      if (best_j < 0 || dist < best) { best = dist; best_j = j; }
    }
    if (best_j >= 0 && best <= max_dist) cands.push_back({best, i, best_j});
  }

  std::sort(cands.begin(), cands.end());
  std::vector<char> used_mp(M, 0), used_kp(N, 0);
  std::vector<int> out_i, out_j;
  std::vector<double> out_d;
  for (const Cand& c : cands) {
    if (used_mp[c.i] || used_kp[c.j]) continue;
    used_mp[c.i] = 1;
    used_kp[c.j] = 1;
    out_i.push_back(c.i);
    out_j.push_back(c.j);
    out_d.push_back(c.dist);
  }

  const py::ssize_t K = static_cast<py::ssize_t>(out_i.size());
  py::array_t<int> ai(K), aj(K);
  py::array_t<double> ad(K);
  std::copy(out_i.begin(), out_i.end(), ai.mutable_data());
  std::copy(out_j.begin(), out_j.end(), aj.mutable_data());
  std::copy(out_d.begin(), out_d.end(), ad.mutable_data());
  return {ai, aj, ad};
}

}  // namespace

PYBIND11_MODULE(vslam_cpp, m) {
  m.doc() = "vslam_cpp: rutas calientes en C++ (v0.5). import vslam_cpp as fast";
  m.def("guided_match_hamming", &guided_match<std::uint8_t>,
        "Matching guiado, descriptores binarios (Hamming). Gemelo de "
        "PnPTracker._guided_match; devuelve (idx_mapa, idx_kp, dist).",
        py::arg("kp_xy"), py::arg("desc"), py::arg("T_pred"), py::arg("map_pts"),
        py::arg("map_desc"), py::arg("fx"), py::arg("fy"), py::arg("cx"),
        py::arg("cy"), py::arg("width"), py::arg("height"),
        py::arg("radius_px"), py::arg("max_dist"));
  m.def("guided_match_l2", &guided_match<float>,
        "Matching guiado, descriptores float (L2). Mismo contrato.",
        py::arg("kp_xy"), py::arg("desc"), py::arg("T_pred"), py::arg("map_pts"),
        py::arg("map_desc"), py::arg("fx"), py::arg("fy"), py::arg("cx"),
        py::arg("cy"), py::arg("width"), py::arg("height"),
        py::arg("radius_px"), py::arg("max_dist"));
}
