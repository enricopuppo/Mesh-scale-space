#include "utilities.h"

using namespace Eigen;
void make_domain(std::vector<double> &coords_out, std::vector<uint> &tris_out,
                 const int N, const vector<vec2d> &domain,
                 const double min_angle = 20) {

  double max_area = 1 / (1.6 * pow(N, 2));
  auto x_min = domain[0].x();
  auto x_max = domain[0].y();
  auto y_min = domain[1].x();
  auto y_max = domain[1].y();
  std::string flags =
      "Qq" + std::to_string(min_angle) + "a" + std::to_string(max_area);
  triangle_wrap({x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max},
                {0, 1, 1, 2, 2, 3, 3, 0}, {}, 0, flags.c_str(), coords_out,
                tris_out);
}
void make_grid(std::vector<double> &coords_out, std::vector<uint> &tris_out,
               const vector<vec2d> &domain, int N) {
  double delta = 1.0 / N;
  double max_area = 1 / (1.6 * pow(N, 2));
  double x_min = domain[0].x();
  double x_max = domain[0].y();
  double y_min = domain[1].x();
  double y_max = domain[1].y();
  std::vector<double> points = {x_min, y_min, x_max, y_min,
                                x_max, y_max, x_min, y_max};
  std::vector<uint> segs = {};
  for (int i = 0; i <= N; ++i) {
    for (int j = 0; j <= N; ++j) {
      if ((i == 0 && j == 0) || (i == 0 && j == N) || (i == N && j == 0) ||
          (i == N && j == N)) {
        continue;
      }
      points.push_back(i * delta);
      points.push_back(j * delta);
    }
  }
  std::string flags = "a" + std::to_string(max_area) + "-Q";
  triangle_wrap(points, segs, {}, 0, flags.c_str(), coords_out, tris_out);
}
const vector<vec2d> sample_points(const vector<vec2d> &domain, int N) {

  double x_min = domain[0].x();
  double x_max = domain[0].y();
  double y_min = domain[1].x();
  double y_max = domain[1].y();
  double deltax = (x_max - x_min) / N;
  double deltay = (y_max - y_min) / N;
  auto result = vector<vec2d>(pow(N + 1, 2));
  for (int i = 0; i <= N; ++i) {
    for (int j = 0; j <= N; ++j) {
      result[(N + 1) * i + j] = vec2d{x_min + i * deltax, y_min + j * deltay};
    }
  }
  return result;
}
void make_grid_convex_hull(std::vector<double> &coords_out,
                           std::vector<uint> &tris_out,
                           const vector<pair<vec2d, int>> &ch, int N) {
  double delta = 1.0 / N;
  double max_area = 1.0 / (1.6 * pow(N, 2));
  auto k = ch.size();
  std::vector<double> points(2 * k);
  std::vector<uint> segs(2 * k);
  for (auto i = 0; i < k; ++i) {
    points[2 * i] = ch[i].first.x();
    points[2 * i + 1] = ch[i].first.y();
    segs[2 * i] = i;
    segs[2 * i + 1] = (i + 1) % k;
  }
  auto point_per_sector = 5.0;
  for (int i = 0; i < k; ++i) {
    auto v0 = ch[i].first;
    auto v1 = ch[(i + 1) % k].first;
    for (auto j = 1; j < point_per_sector; ++j) {
      for (auto h = 1; h < point_per_sector; ++h) {
        auto alpha = (double)j / point_per_sector;
        auto beta = (double)h / point_per_sector;
        auto p = alpha * v0 + beta * v1;
        points.push_back(p.x());
        points.push_back(p.y());
      }
    }
  }
  std::string flags = "-S" + std::to_string(100) + "-Q";

  triangle_wrap(points, segs, {}, 0, flags.c_str(), coords_out, tris_out);
}
const vector<vec2d> make_grid_convex_hull(const vector<vec2d> &ch, int N) {

  auto k = ch.size();
  auto result = vector<vec2d>{};
  auto point_per_sector = (double)N / k;
  for (int i = 0; i < k; ++i) {
    auto v0 = ch[i];
    auto v1 = ch[(i + 1) % k];
    for (auto j = 1; j < point_per_sector; ++j) {
      for (auto h = 1; h < point_per_sector; ++h) {
        auto alpha = (double)j / point_per_sector;
        auto beta = (double)h / point_per_sector;
        result.push_back(alpha * v0 + beta * v1);
      }
    }
  }
  return result;
}
void weights_heat_map(point_cloud &pc, const int vid) {
  // auto w_min = pc.min_max_weights.x();
  // auto w_max = pc.min_max_weights.y();
  auto w_min = __DBL_MAX__;
  auto w_max = __DBL_MIN__;
  auto w = pc.patches[vid].weights;
  for (auto i = 0; i < w.size(); ++i) {
    w_max = std::max(w_max, w[i]);
    w_min = std::min(w_min, w[i]);
  }
  auto nbr = pc.nbrs[vid];
  auto k = nbr.size();
  auto bad = pc.patches[vid].bad_ids;
  auto range = (float)(w_max - w_min);
  auto blue = vec3f{0.1, 0.1, 1};
  auto red = vec3f{1, 0.1, 0.1};
  for (auto i = 0; i < k; ++i) {
    auto it = find(bad.begin(), bad.end(), nbr[i]);
    if (it != bad.end()) {
      auto &curr = pc.points[nbr[i]];
      curr =
          DrawableSphere(curr.center, 2 * curr.radius, cinolib::Color::GRAY());
      continue;
    }

    auto &curr = pc.points[nbr[i]];
    auto curr_value = w[i];
    auto alpha = (float)(range - (curr_value - w_min)) / range;
    auto col = alpha * blue + (1 - alpha) * red;
    curr = DrawableSphere(curr.center, curr.radius,
                          cinolib::Color(col.x(), col.y(), col.z()));
  }
}
void total_weights_heat_map(point_cloud &pc) {
  auto blue = vec3f{0.1, 0.1, 1};
  auto red = vec3f{1, 0.1, 0.1};
  vector<double> avg(pc.positions.size());
  auto w_min = DBL_MAX;
  auto w_max = DBL_MIN;
  for (auto i = 0; i < pc.positions.size(); ++i) {
    auto w_avg = 0.0;
    auto w = pc.residuals[i];
    for (auto j = 0; j < w.size(); ++j) {
      w_avg += w[j];
    }
    w_avg /= w.size();
    avg[i] = w_avg;
    w_min = min(w_min, w_avg);
    w_max = max(w_max, w_avg);
  }
  auto range = (float)(w_max - w_min);
  for (auto i = 0; i < pc.positions.size(); ++i) {

    auto &curr = pc.points[i];
    auto curr_value = avg[i];
    auto alpha = (float)(range - (curr_value - w_min)) / range;
    auto col = alpha * blue + (1 - alpha) * red;
    curr = DrawableSphere(curr.center, curr.radius,
                          cinolib::Color(col.x(), col.y(), col.z()));
  }
}
void reset(point_cloud &pc, const double &point_size,
           const bool original_color) {

  for (auto i = 0; i < pc.positions.size(); ++i) {
    auto &curr = pc.points[i];
    auto col = (original_color) ? curr.color : cinolib::Color::GRAY();
    curr = DrawableSphere(pc.positions[i], point_size, col);
  }
}
void weights_heat_map(point_cloud &pc, const int vid, const Eigen::MatrixXd &W,
                      const vector<int> &nbr) {
  // auto w_min = pc.min_max_weights.x();
  // auto w_max = pc.min_max_weights.y();
  auto w_min = __DBL_MAX__;
  auto w_max = __DBL_MIN__;
  for (auto i = 0; i < W.rows(); ++i) {
    w_max = std::max(w_max, W(i, i));
    w_min = std::min(w_min, W(i, i));
  }
  auto k = nbr.size();

  auto range = (float)(w_max - w_min);
  auto blue = vec3f{0.1, 0.1, 1};
  auto red = vec3f{1, 0.1, 0.1};
  for (auto i = 0; i < k; ++i) {
    auto &curr = pc.points[nbr[i]];
    auto curr_value = W(i, i);
    auto alpha = (float)(range - (curr_value - w_min)) / range;
    auto col = alpha * blue + (1 - alpha) * red;
    curr = DrawableSphere(curr.center, curr.radius,
                          cinolib::Color(col.x(), col.y(), col.z()));
  }
}
void higlight_point(point_cloud &pc, const int vid, const int factor) {
  auto &curr = pc.points[vid];

  curr = DrawableSphere(curr.center, factor * curr.radius,
                        cinolib::Color::BLACK());
}

// DrawableCurve show_CH(const point_cloud &pc, const int vid) {
//   auto CH = pc.patches[vid].CH;
//   auto k = CH.size();
//   auto samples = vector<vec3d>(k);
//   auto e0 = pc.basis[vid][0];
//   auto e1 = pc.basis[vid][1];
//   for (auto i = 0; i < k; ++i)
//     samples[i] = e0 * CH[i].x() + e1 * CH[i].y();
//   DrawableCurve result;
//   result = DrawableCurve(samples);
//   return result;
// };
DrawableSegmentSoup show_CH(const point_cloud &pc, const int vid) {
  auto CH = pc.patches[vid].CH;
  auto k = CH.size();
  auto samples = vector<vec3d>(k);
  auto e0 = pc.basis[vid][0];
  auto e1 = pc.basis[vid][1];
  DrawableSegmentSoup s;

  auto pos = pc.positions[vid];
  for (auto i = 0; i < k; ++i) {
    samples[i] = e0 * CH[i].first.x() + e1 * CH[i].first.y();
    samples[i] += pos;

    if (i > 0)
      s.push_seg(samples[i - 1], samples[i], Color::BLACK());
  }
  s.push_seg(samples.back(), samples[0], Color::BLACK());
  return s;
};
DrawableSegmentSoup draw_line_in_tangent_space(const point_cloud &pc,
                                               const int vid,
                                               const double &theta,
                                               Color &col) {
  auto e0 = pc.basis[vid][0];
  auto e1 = pc.basis[vid][1];
  auto samples = vector<vec3d>(3);
  auto pos = pc.positions[vid];
  DrawableSegmentSoup s;
  for (auto i = 0; i < 3; ++i) {
    auto r = (double)i / 10;
    auto coords = vec2d{r * std::cos(theta), r * std::sin(theta)};
    samples[i] = e0 * coords.x() + e1 * coords.y();
    samples[i] += pos;
    if (i > 0)
      s.push_seg(samples[i - 1], samples[i], col);
  }

  return s;
}
DrawableTrimesh<> draw_plane(const vector<vec3d> &basis, const vec3d &p) {
  std::vector<double> coords_out;
  std::vector<uint> tris_out;
  auto domain_range = vector<vec2d>(2);
  domain_range[0] = domain_range[1] = vec2d{-0.5, 0.5};
  make_grid(coords_out, tris_out, domain_range, 100);
  if ((int)coords_out.size() % 3 != 0)
    std::cerr << "Error in computing DT" << std::endl;
  auto N = coords_out.size() / 3;
  std::vector<vec3d> pos_on_patch(N);
  auto e0 = basis[0];
  auto e1 = basis[1];
  for (auto i = 0; i < N; ++i) {
    auto pos_in_plane = vec2d{coords_out[3 * i], coords_out[3 * i + 1]};
    pos_on_patch[i] = p + pos_in_plane.x() * e0 + pos_in_plane.y() * e1;
  }
  return DrawableTrimesh(pos_on_patch, tris_out);
}
std::tuple<vector<DrawableSphere>, DrawableTrimesh<>>
show_tagent_plane_mapping(const point_cloud &pc, const int vid) {
  auto k = pc.nbrs[vid].size();
  auto points = vector<DrawableSphere>(k);
  auto e0 = pc.basis[vid][0];
  auto e1 = pc.basis[vid][1];
  auto pos = pc.positions[vid];
  auto &p = pc.patches[vid].parametric_nbr;
  for (auto i = 0; i < k; ++i) {
    auto curr_pos = pos + e0 * p[i].first.x() + e1 * p[i].first.y();
    // curr_pos += pos;
    points[i] = DrawableSphere(curr_pos, 0.002, cinolib::Color::BLACK());
  }

  return {points, draw_plane(pc.basis[vid], pc.positions[vid])};
}

void residual_heat_map(point_cloud &pc) {
  auto res_min = __DBL_MAX__;
  auto res_max = __DBL_MIN__;
  auto V = (int)pc.positions.size();
  for (auto i = 0; i < V; ++i) {
    auto curr_res = pc.patches[i].residual;
    res_max = std::max(res_max, curr_res);
    res_min = std::min(res_min, curr_res);
  }
  auto range = (float)(res_max - res_min);
  auto blue = vec3f{0.1, 0.1, 1};
  auto red = vec3f{1, 0.1, 0.1};
  for (auto i = 0; i < V; ++i) {

    auto &curr = pc.points[i];
    auto curr_value = pc.patches[i].residual;
    auto alpha = (float)(range - (curr_value - res_min)) / range;
    auto col = alpha * blue + (1 - alpha) * red;
    curr = DrawableSphere(curr.center, curr.radius,
                          cinolib::Color(col.x(), col.y(), col.z()));
  }
}
void reset_weights_heat_map(point_cloud &pc, const int vid) {
  auto nbr = pc.nbrs[vid];
  auto k = nbr.size();
  for (auto i = 0; i < k; ++i) {

    auto &curr = pc.points[nbr[i]];
    curr = DrawableSphere(curr.center, curr.radius, cinolib::Color::GRAY());
  }
}
vector<vec3d> mesh_wrapper(const vector<double> &pos) {
  auto N = pos.size();
  auto result = vector<vec3d>(N / 3);
  for (auto i = 0; i < N; ++i)
    result[i] = vec3d{pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};

  return result;
}
DrawableTrimesh<> draw_patch(const point_cloud &pc, const patch &p) {

  std::vector<double> coords_out;
  std::vector<uint> tris_out;

  if (p.parametric_nbr.size() < 3)
    return DrawableTrimesh(coords_out, tris_out);
  else if (p.parametric_nbr.size() < 7)
    make_grid(coords_out, tris_out, p.domain_range, 50);
  else {

    make_grid_convex_hull(coords_out, tris_out, p.CH, 50);
  }

  if ((int)coords_out.size() % 3 != 0)
    std::cerr << "Error in computing DT" << std::endl;
  auto N = coords_out.size() / 3;
  std::vector<vec3d> pos_on_patch(N);
  auto Q = p.quadric;
  for (auto i = 0; i < N; ++i) {

    pos_on_patch[i] =
        evaluate_quadric(Q, vec2d{coords_out[3 * i], coords_out[3 * i + 1]});
  }
  return DrawableTrimesh(pos_on_patch, tris_out);
}
DrawableTrimesh<> draw_parabolic_cylinder(const point_cloud &pc,
                                          const patch &p) {

  std::vector<double> coords_out;
  std::vector<uint> tris_out;

  make_grid_convex_hull(coords_out, tris_out, p.CH, 50);

  if ((int)coords_out.size() % 3 != 0)
    std::cerr << "Error in computing DT" << std::endl;
  auto N = coords_out.size() / 3;
  std::vector<vec3d> pos_on_patch(N);
  auto Q = p.Monge_quadric;
  for (auto i = 0; i < N; ++i) {

    // pos_on_patch[i] = pos_on_parabolic_cylinder(
    //     Q, vec2d{coords_out[3 * i], coords_out[3 * i + 1]}, p.u0, p.e);
    pos_on_patch[i] = pos_on_parabolic_cylinder(
        Q, vec2d{coords_out[3 * i], coords_out[3 * i + 1]}, p.e);
  }
  return DrawableTrimesh(pos_on_patch, tris_out);
}
DrawableTrimesh<> draw_patch(const point_cloud &pc, const int vid,
                             const bool Monge) {

  auto &p = pc.patches[vid];
  std::vector<double> coords_out;
  std::vector<uint> tris_out;

  if (p.parametric_nbr.size() < 3)
    return DrawableTrimesh(coords_out, tris_out);
  else if (p.parametric_nbr.size() < 7)
    make_grid(coords_out, tris_out, p.domain_range, 50);
  else {

    make_grid_convex_hull(coords_out, tris_out, p.CH, 50);
  }

  if ((int)coords_out.size() % 3 != 0)
    std::cerr << "Error in computing DT" << std::endl;
  auto N = coords_out.size() / 3;
  std::vector<vec3d> pos_on_patch(N);
  auto Q = p.quadric;
  for (auto i = 0; i < N; ++i) {
    if (!Monge)
      pos_on_patch[i] =
          evaluate_quadric(Q, vec2d{coords_out[3 * i], coords_out[3 * i + 1]});
    else {
      auto e = pc.basis[vid];

      pos_on_patch[i] = evaluate_quadric_MP(
          pc.positions[vid], vector<vec3d>{e[1], e[2], e[1].cross(e[2])},
          p.Monge_quadric, vec2d{coords_out[3 * i], coords_out[3 * i + 1]});
    }
  }
  return DrawableTrimesh(pos_on_patch, tris_out);
}
DrawableTrimesh<> draw_Monge_patch(const patch &p) {

  std::vector<double> coords_out;
  std::vector<uint> tris_out;

  make_grid_convex_hull(coords_out, tris_out, p.CH, 50);

  if ((int)coords_out.size() % 3 != 0)
    std::cerr << "Error in computing DT" << std::endl;
  auto N = coords_out.size() / 3;
  std::vector<vec3d> pos_on_patch(N);
  auto Q = p.quadric;
  for (auto i = 0; i < N; ++i) {

    auto e = p.e;

    pos_on_patch[i] = evaluate_quadric_MP(
        e[0], vector<vec3d>{e[1], e[2], e[1].cross(e[2])}, p.Monge_quadric,
        vec2d{coords_out[3 * i], coords_out[3 * i + 1]});
  }
  return DrawableTrimesh(pos_on_patch, tris_out);
}
vector<pair<vec2d, int>> domain_range(const vector<pair<vec2d, int>> &nbr) {
  auto u_max = DBL_MIN;
  auto u_min = DBL_MAX;
  auto v_max = DBL_MIN;
  auto v_min = DBL_MAX;
  auto result = vector<pair<vec2d, int>>(4, std::make_pair(vec2d{0, 0}, -1));

  for (auto i = 0; i < nbr.size(); ++i) {
    auto &uv = nbr[i].first;
    u_max = std::max(u_max, uv.x());
    u_min = std::min(u_min, uv.x());
    v_max = std::max(v_max, uv.y());
    v_min = std::min(v_min, uv.y());
  }
  result[0].first = vec2d{u_min, v_min};
  result[1].first = vec2d{u_max, v_min};
  result[2].first = vec2d{u_max, v_max};
  result[3].first = vec2d{u_min, v_max};

  return result;
}
DrawableTrimesh<> draw_primitive(const point_cloud &pc, const iVd &vor,
                                 const vector<int> &centers, const int center,
                                 const patch &p, const int type) {

  std::vector<double> coords_out;
  std::vector<uint> tris_out;
  auto entry =
      distance(centers.begin(), find(centers.begin(), centers.end(), center));
  auto p_nbr = vor.parametric_nbr[entry];
  make_grid_convex_hull(coords_out, tris_out, domain_range(p_nbr), 50);

  if ((int)coords_out.size() % 3 != 0)
    std::cerr << "Error in computing DT" << std::endl;
  auto N = coords_out.size() / 3;
  std::vector<vec3d> pos_on_patch(N);
  auto Q = p.quadric;
  for (auto i = 0; i < N; ++i) {
    pos_on_patch[i] = evaluate_primitive(
        Q, vec2d{coords_out[3 * i], coords_out[3 * i + 1]}, type);
  }
  return DrawableTrimesh(pos_on_patch, tris_out);
}
DrawableTrimesh<> draw_primitive(const point_cloud &pc, const int vid,
                                 const patch &p, const int type) {

  std::vector<double> coords_out;
  std::vector<uint> tris_out;
  make_grid_convex_hull(coords_out, tris_out, p.CH, 50);

  if ((int)coords_out.size() % 3 != 0)
    std::cerr << "Error in computing DT" << std::endl;
  auto N = coords_out.size() / 3;
  std::vector<vec3d> pos_on_patch(N);
  auto Q = p.quadric;
  for (auto i = 0; i < N; ++i) {
    pos_on_patch[i] = evaluate_primitive(
        Q, vec2d{coords_out[3 * i], coords_out[3 * i + 1]}, type);
  }
  return DrawableTrimesh(pos_on_patch, tris_out);
}

DrawableTrimesh<> draw_patches(const point_cloud &pc) {
  std::vector<vec3d> coords_out;
  std::vector<uint> tris_out;

  for (auto i = 0; i < pc.positions.size(); ++i) {
    if (pc.badones[i])
      continue;
    std::vector<double> curr_coords_out;
    std::vector<uint> curr_tris_out;
    make_grid_convex_hull(curr_coords_out, curr_tris_out, pc.patches[i].CH, 50);
    if ((int)curr_coords_out.size() % 3 != 0)
      std::cerr << "Error in computing DT" << std::endl;
    auto N = curr_coords_out.size() / 3;
    std::vector<vec3d> pos_on_patch(N);
    auto Q = pc.patches[i].quadric;
    for (auto j = 0; j < N; ++j) {
      pos_on_patch[j] = evaluate_quadric(
          Q, vec2d{curr_coords_out[3 * j], curr_coords_out[3 * j + 1]});
    }
    auto size = (uint)coords_out.size();
    for (auto &tid : curr_tris_out)
      tid += size;
    coords_out.insert(coords_out.end(), pos_on_patch.begin(),
                      pos_on_patch.end());
    tris_out.insert(tris_out.end(), curr_tris_out.begin(), curr_tris_out.end());
  }

  return DrawableTrimesh(coords_out, tris_out);
}

void detect_sharp_features(point_cloud &pc) {
  // auto tau = pc.tau;
  for (auto i = 0; i < pc.positions.size(); ++i) {
    auto curr = (pc.patch_tagging[i].size() > 0) ? pc.patch_tagging[i][0] : i;
    if (pc.patches[curr].residual > 1e-3) {
      auto &curr = pc.points[i];
      curr = DrawableSphere(curr.center, curr.radius, cinolib::Color::RED());
    }
  }
}
void color_points_according_to_quadric(point_cloud &pc, const int vid) {
  auto &p = pc.patches[vid];
  Color col;
  if (p.type == bilinear_patch)
    col = cinolib::Color::RED();
  else if (p.type == parabolic_cylinder)
    col = cinolib::Color::BLUE();
  else if (p.type == hyperbolic_paraboloid)
    col = cinolib::Color::GREEN();
  else if (p.type == elliptic_paraboloid)
    col = cinolib::Color::YELLOW();
  else if (p.type == ellipsoid)
    col = cinolib::Color::MAGENTA();
  else if (p.type == elliptic_cylinder)
    col = cinolib::Color::CYAN();
  else if (p.type == hyperbolic_cylinder)
    col = cinolib::Color::BLACK();
  else if (p.type == hyperboloid)
    col = cinolib::Color::PASTEL_VIOLET();

  for (auto &nei : pc.nbrs[vid]) {
    auto &curr = pc.points[nei];
    curr = DrawableSphere(curr.center, curr.radius, col);
  }
}
void color_points_according_to_quadric(point_cloud &pc, const vector<int> &nbr,
                                       const patch &p) {

  Color col;
  if (p.type == bilinear_patch)
    col = cinolib::Color::RED();
  else if (p.type == parabolic_cylinder)
    col = cinolib::Color::BLUE();
  else if (p.type == hyperbolic_paraboloid)
    col = cinolib::Color::GREEN();
  else if (p.type == elliptic_paraboloid)
    col = cinolib::Color::YELLOW();
  else if (p.type == ellipsoid)
    col = cinolib::Color::MAGENTA();
  else if (p.type == elliptic_cylinder)
    col = cinolib::Color::CYAN();
  else if (p.type == hyperbolic_cylinder)
    col = cinolib::Color::BLACK();
  else if (p.type == hyperboloid)
    col = cinolib::Color::PASTEL_GREEN();

  for (auto &nei : nbr) {
    auto &curr = pc.points[nei];
    curr = DrawableSphere(curr.center, curr.radius, col);
  }
}
void show_outliers(point_cloud &pc) {
  for (auto i = 0; i < pc.positions.size(); ++i) {
    auto vid = pc.patch_tagging[i][0];
    if (pc.patches[vid].residual > pc.tau) {
      auto &curr = pc.points[i];
      curr = DrawableSphere(curr.center, curr.radius, cinolib::Color::RED());
    }
  }
}
void show_secondary_patches(point_cloud &pc) {
  auto &ordered = pc.ordered_patches;
  for (auto i = 0; i < pc.positions.size(); ++i) {
    auto curr_center = ordered[i];
    auto &p = pc.patches[curr_center];
    if (!p.is_center)
      continue;

    for (auto &inv : p.invaded) {
      if (pc.patches[inv].invasors.size() == 1 && pc.nbrs[inv].size() > 7 &&
          pc.patches[inv].expandable) {
        auto &curr = pc.points[inv];
        curr =
            DrawableSphere(curr.center, curr.radius, cinolib::Color::GREEN());
      }
    }
  }
}
void show_centers(point_cloud &pc) {
  for (auto i = 0; i < pc.positions.size(); ++i) {
    if (pc.patches[i].is_center) {
      auto &curr = pc.points[i];
      curr = DrawableSphere(curr.center, curr.radius, cinolib::Color::RED());
    }
  }
}

void show_patch_tagging(point_cloud &pc) {
  for (auto i = 0; i < pc.positions.size(); ++i) {
    // if (!pc.patches[i].is_center)
    //   continue;
    auto &p = pc.patches[i];
    auto col = random_color();
    pc.patches[i].color = col;
    for (auto j = 0; j < pc.patches[i].tagged.size(); ++j) {
      auto &curr = pc.points[pc.patches[i].tagged[j]];
      auto curr_col = Color(col.x(), col.y(), col.z());
      curr = DrawableSphere(curr.center, curr.radius, curr_col);
    }
  }
}

void update_patch_tagging(point_cloud &pc, const int vid) {

  // if (!pc.patches[i].is_center)
  //   continue;
  auto col = pc.patches[vid].color;
  for (auto j = 0; j < pc.patches[vid].tagged.size(); ++j) {
    auto &curr = pc.points[pc.patches[vid].tagged[j]];
    curr = DrawableSphere(curr.center, curr.radius,
                          Color(col.x(), col.y(), col.z()));
  }
}

std::tuple<DrawableSegmentSoup, DrawableSegmentSoup>
draw_curvature_cross_field(const vector<vec3d> &positions,
                           const vector<vec3d> &k1, const vector<vec3d> &k2,
                           const double h) {
  DrawableSegmentSoup K1;
  DrawableSegmentSoup K2;
  for (auto i = 0; i < k1.size(); ++i) {
    auto p = positions[i];
    auto v = k1[i];
    v.normalize();
    auto p0 = p + h * v;
    auto p1 = p - h * v;
    K1.push_seg(p0, p1);
    v = k2[i];
    v.normalize();
    p0 = p + h * v;
    p1 = p - h * v;
    K2.push_seg(p0, p1);
  }

  return {K1, K2};
}

int max_curvature_point(const point_cloud &pc) {
  auto k = DBL_MIN;
  auto vid = -1;
  for (auto i = 0; i < pc.positions.size(); ++i) {
    auto curr_k = std::abs(gaussian_curvature(pc, i));
    if (curr_k > k) {
      k = curr_k;
      vid = i;
    }
  }

  return vid;
}
int min_curvature_point(const point_cloud &pc) {
  auto k = DBL_MAX;
  auto vid = -1;
  for (auto i = 0; i < pc.positions.size(); ++i) {
    auto curr_k = std::abs(gaussian_curvature(pc, i));
    if (curr_k < k) {
      k = curr_k;
      vid = i;
    }
  }

  return vid;
}
int is_local_maximum(const vector<double> &field, const point_cloud &pc,
                     const int vid) {
  auto nbr = knn(pc.tree, pc.positions, vid, 81);

  auto s = nbr.size();
  auto M = DBL_MIN;
  auto M_id = -1;
  for (auto i = 0; i < s; ++i) {

    if (field[nbr[i]] > M) {
      M = field[nbr[i]];
      M_id = nbr[i];
    }
  }

  if (M_id == vid)
    return -1;

  return M_id;
}
void highlight_points(point_cloud &pc, const vector<int> &points,
                      const double &size) {

  for (auto i = 0; i < points.size(); ++i) {
    auto &curr = pc.points[points[i]];
    auto rad = (size >= 0) ? size : 2 * curr.radius;
    curr = DrawableSphere(curr.center, rad, cinolib::Color::BLACK());
  }
}
void highlight_lndmarks(point_cloud &pc0, point_cloud &pc1,
                        const vector<int> &lndmarks, const double &size) {

  auto n = (int)lndmarks.size();
  srand(1);
  auto odd = (n % 2);
  for (auto i = 0; i < n; i += 2) {
    auto rnd = random_color();
    auto col = Color(rnd.x(), rnd.y(), rnd.z());
    auto &curr0 = pc0.points[lndmarks[i]];
    auto rad = (size < 0) ? 9 * curr0.radius : size;
    curr0 = DrawableSphere(curr0.center, rad, col);
    if (odd && i == (n - 1) / 2)
      continue;

    auto &curr1 = pc1.points[lndmarks[i + 1]];
    curr1 = DrawableSphere(curr1.center, rad, col);
  }
}
void highlight_paired_centers_original(point_cloud &pc0, point_cloud &pc1,
                                       const iVd &vor1,
                                       const vector<int> &centers0,
                                       const vector<int> &centers1,
                                       const vector<Color> &cols) {

  auto n = (int)centers0.size();

  for (auto i = 0; i < n; ++i) {
    auto &curr1 = pc1.points[centers1[i]];
    curr1 = DrawableSphere(curr1.center, 0.005, cols[i]);
  }
  for (auto i = 0; i < n; ++i) {
    auto &curr0 = pc0.points[centers0[i]];
    auto tag1 = vor1.voronoi_tags[centers0[i]];
    auto entry = distance(centers1.begin(),
                          find(centers1.begin(), centers1.end(), tag1));

    curr0 = DrawableSphere(curr0.center, 0.005, cols[entry]);
  }
}
void highlight_paired_centers_original(point_cloud &pc0, point_cloud &pc1,
                                       const vector<int> &centers0,
                                       const vector<int> &centers1,
                                       const unordered_map<int, int> &mapping,
                                       const vector<Color> &cols) {

  auto n = (int)centers0.size();

  for (auto i = 0; i < n; ++i) {
    auto &curr1 = pc1.points[centers1[i]];
    curr1 = DrawableSphere(curr1.center, 0.02, cols[i]);
  }
  for (auto i = 0; i < n; ++i) {
    auto &curr0 = pc0.points[centers0[i]];
    auto tag1 = mapping.at(centers0[i]);
    auto entry = distance(centers1.begin(),
                          find(centers1.begin(), centers1.end(), tag1));

    curr0 = DrawableSphere(curr0.center, 0.02, cols[entry]);
  }
}
void show_candidates(point_cloud &pc, const vector<double> &phi, const int V,
                     const int nf) {
  auto lndmarks = compute_candidates(phi, V, nf);
  printf("There are %d seeds", (int)lndmarks.size());
  highlight_points(pc, lndmarks);
}
vector<Color> create_colors(const int n, const int offset = 0) {
  auto result = vector<Color>(n + 1);

  for (auto i = 0; i < n + 1; ++i) {
    result[i] = Color::scatter(n + 1, i, 0.5, 0.85);
  }

  if (offset != 0) {
    auto rng = std::default_random_engine{};
    rng.seed(time(NULL));
    shuffle(result.begin(), result.end(), rng);
  }
  return result;
}
void show_voronoi_regions(DrawableTrimesh<> &m, const iVd &intrinsic_voronoi,
                          const vector<int> &voronoi_centers,
                          const int offset) {
  auto regions = (int)voronoi_centers.size();
  auto boundaries = m.num_verts();
  for (auto i = 0; i < m.num_polys(); ++i) {
    auto tagx = intrinsic_voronoi.voronoi_tags[m.poly_vert_id(i, 0)];
    auto tagy = intrinsic_voronoi.voronoi_tags[m.poly_vert_id(i, 1)];
    auto tagz = intrinsic_voronoi.voronoi_tags[m.poly_vert_id(i, 2)];
    if (tagx == tagy && tagx == tagz) {
      m.poly_data(i).label = tagx + offset;
    } else
      m.poly_data(i).label = boundaries;
  }
  m.poly_color_wrt_label();
  m.show_poly_color();
}
void show_voronoi_regions_and_centers(DrawableTrimesh<> &m, point_cloud &pc,
                                      const iVd &intrinsic_voronoi,
                                      const vector<int> &voronoi_centers,
                                      const double &rad, const int offset) {
  auto regions = (int)voronoi_centers.size();
  auto cols = create_colors(regions, offset);
  for (auto i = 0; i < m.num_polys(); ++i) {
    auto tagx = intrinsic_voronoi.voronoi_tags[m.poly_vert_id(i, 0)];
    auto tagy = intrinsic_voronoi.voronoi_tags[m.poly_vert_id(i, 1)];
    auto tagz = intrinsic_voronoi.voronoi_tags[m.poly_vert_id(i, 2)];
    // if (tagx == tagy && tagx == tagz) {
    auto entry =
        distance(voronoi_centers.begin(),
                 find(voronoi_centers.begin(), voronoi_centers.end(), tagx));
    m.poly_data(i).color = cols[entry];
    // } else
    //   m.poly_data(i).color = cols.back();
  }

  // for (auto i = 0; i < voronoi_centers.size(); ++i) {
  //   auto &curr = pc.points[voronoi_centers[i]];
  //   curr = DrawableSphere(curr.center, rad, cols[i]);
  // }
  m.show_poly_color();
}

vector<Color>
show_paired_voronoi_regions(DrawableTrimesh<> &m0, const iVd &ivd0,
                            const vector<int> &centers0, DrawableTrimesh<> &m1,
                            const iVd &ivd1, const vector<int> &centers1,
                            const std::unordered_map<int, int> &mapping) {
  auto regions = (int)centers0.size();
  auto cols = create_colors(regions);
  auto boundaries = regions + 1;

  for (auto i = 0; i < m1.num_polys(); ++i) {
    auto tagx = ivd1.voronoi_tags[m1.poly_vert_id(i, 1)];
    auto tagy = ivd1.voronoi_tags[m1.poly_vert_id(i, 1)];
    auto tagz = ivd1.voronoi_tags[m1.poly_vert_id(i, 2)];
    if (tagx == tagy && tagx == tagz) {
      auto entry = distance(centers1.begin(),
                            find(centers1.begin(), centers1.end(), tagx));
      m1.poly_data(i).color = cols[entry];
    } else
      m1.poly_data(i).color = cols.back();
  }
  for (auto i = 0; i < m0.num_polys(); ++i) {
    auto tagx = ivd0.voronoi_tags[m0.poly_vert_id(i, 0)];
    auto tagy = ivd0.voronoi_tags[m0.poly_vert_id(i, 1)];
    auto tagz = ivd0.voronoi_tags[m0.poly_vert_id(i, 2)];
    if (tagx == tagy && tagx == tagz) {
      auto entry =
          distance(centers1.begin(),
                   find(centers1.begin(), centers1.end(), mapping.at(tagx)));

      m0.poly_data(i).color = cols[entry];
    } else
      m0.poly_data(i).color = cols.back();
  }

  m0.show_poly_color();

  m1.show_poly_color();

  return cols;
}
vector<Color> show_paired_voronoi_regions_original(
    DrawableTrimesh<> &m0, const iVd &ivd0, const vector<int> &centers0,
    DrawableTrimesh<> &m1, const iVd &ivd1, const vector<int> &centers1,
    const int offset) {
  auto regions = (int)centers0.size();
  auto cols = create_colors(regions, offset);
  auto boundaries = regions + 1;

  for (auto i = 0; i < m1.num_polys(); ++i) {
    auto tagx = ivd1.voronoi_tags[m1.poly_vert_id(i, 1)];
    auto tagy = ivd1.voronoi_tags[m1.poly_vert_id(i, 1)];
    auto tagz = ivd1.voronoi_tags[m1.poly_vert_id(i, 2)];
    if (tagx == tagy && tagx == tagz) {
      auto entry = distance(centers1.begin(),
                            find(centers1.begin(), centers1.end(), tagx));
      m1.poly_data(i).color = cols[entry];
    } else
      m1.poly_data(i).color = cols.back();
  }
  for (auto i = 0; i < m0.num_polys(); ++i) {
    auto tagx = ivd0.voronoi_tags[m0.poly_vert_id(i, 0)];
    auto tagy = ivd0.voronoi_tags[m0.poly_vert_id(i, 1)];
    auto tagz = ivd0.voronoi_tags[m0.poly_vert_id(i, 2)];
    if (tagx == tagy && tagx == tagz) {
      auto tag1 = ivd1.voronoi_tags[tagx];
      auto entry = distance(centers1.begin(),
                            find(centers1.begin(), centers1.end(), tag1));

      m0.poly_data(i).color = cols[entry];
    } else
      m0.poly_data(i).color = cols.back();
  }

  m0.show_poly_color();

  m1.show_poly_color();

  return cols;
}
void show_paired_voronoi_regions(point_cloud &pc0, const iVd &ivd0,
                                 const vector<int> &centers0, point_cloud &pc1,
                                 const iVd &ivd1, const vector<int> &centers1,
                                 const std::unordered_map<int, int> &mapping) {
  auto regions = (int)centers0.size();
  auto cols = create_colors(regions);

  for (auto i = 0; i < pc1.positions.size(); ++i) {
    auto tagx = ivd1.voronoi_tags[i];

    auto entry = distance(centers1.begin(),
                          find(centers1.begin(), centers1.end(), tagx));
    auto &curr = pc1.points[i];
    curr = DrawableSphere(curr.center, curr.radius, cols[entry]);
  }
  for (auto i = 0; i < pc0.positions.size(); ++i) {
    auto tagx = ivd0.voronoi_tags[i];
    auto counterimage =
        mapping.at(tagx); // this could be not a center of centers1, the
                          // mappings are not one the inverse of the other
    auto entry = distance(centers1.begin(),
                          find(centers1.begin(), centers1.end(), counterimage));
    if (entry == centers1.size())
      std::cout << "Vertex" << i << "has no tags" << std::endl;
    auto &curr = pc0.points[i];
    curr = DrawableSphere(curr.center, curr.radius, cols[entry]);
  }
}
vector<Color>
show_paired_voronoi_regions(DrawableTrimesh<> &m0, const iVd &ivd0,
                            const vector<int> &centers0, DrawableTrimesh<> &m1,
                            const iVd &ivd1, const vector<int> &centers1,
                            const std::unordered_map<int, int> &m02,
                            const std::unordered_map<int, int> &m31) {
  auto regions = (int)centers0.size();
  auto cols = create_colors(regions);

  for (auto i = 0; i < m1.num_polys(); ++i) {
    auto tagx = ivd1.voronoi_tags[m1.poly_vert_id(i, 1)];
    auto tagy = ivd1.voronoi_tags[m1.poly_vert_id(i, 1)];
    auto tagz = ivd1.voronoi_tags[m1.poly_vert_id(i, 2)];
    if (tagx == tagy && tagx == tagz) {
      auto entry = distance(centers1.begin(),
                            find(centers1.begin(), centers1.end(), tagx));
      m1.poly_data(i).color = cols[entry];
    } else
      m1.poly_data(i).color = cols.back();
  }
  for (auto i = 0; i < m0.num_polys(); ++i) {
    auto tagx = ivd0.voronoi_tags[m0.poly_vert_id(i, 0)];
    auto tagy = ivd0.voronoi_tags[m0.poly_vert_id(i, 1)];
    auto tagz = ivd0.voronoi_tags[m0.poly_vert_id(i, 2)];
    if (tagx == tagy && tagx == tagz) {
      auto vid_or = m02.at(tagx);
      auto vid1 = m31.at(vid_or);
      auto tag1 = ivd1.voronoi_tags[vid1];
      auto entry = distance(centers1.begin(),
                            find(centers1.begin(), centers1.end(), tag1));

      m0.poly_data(i).color = cols[entry];
    } else
      m0.poly_data(i).color = cols.back();
  }

  m0.show_poly_color();

  m1.show_poly_color();

  return cols;
}
void find_local_maxima(point_cloud &pc, const vector<double> &f) {

  for (auto i = 0; i < f.size(); ++i) {
    auto is_max = is_local_maximum(f, pc, i);
    if (is_max == -1) {
      auto &curr = pc.points[i];
      curr =
          DrawableSphere(curr.center, 2 * curr.radius, cinolib::Color::BLACK());
    }
  }
}
DrawableSegmentSoup draw_gradient_field(const point_cloud &pc,
                                        const vector<vec3d> &gradient,
                                        const double h) {

  DrawableSegmentSoup G;
  G.default_color = Color::BLUE();
  for (auto i = 0; i < gradient.size(); ++i) {
    auto p = pc.positions[i];
    auto v = gradient[i];
    v.normalize();
    auto p0 = p + h * v;
    G.push_seg(p, p0);
  }

  return G;
}
DrawableSegmentSoup draw_local_gradient_field(const point_cloud &pc,
                                              const int vid,
                                              const vector<vec3d> &gradient,
                                              const double &r,
                                              const double &h) {

  DrawableSegmentSoup G;
  G.default_color = Color::GREEN();
  auto thetas = subdivide_angles(10);
  for (auto i = 0; i < gradient.size(); ++i) {
    auto curr_pos = vec2d{r * std::cos(thetas[i]), r * std::sin(thetas[i])};
    auto p = evaluate_quadric(pc.patches[vid].quadric, curr_pos);
    auto v = gradient[i];
    v.normalize();
    auto p0 = p + h * v;
    G.push_seg(p, p0);
  }

  return G;
}
DrawableSegmentSoup draw_local_princ_dir_field(const point_cloud &pc,
                                               const int vid,
                                               const vector<vec3d> &dir,
                                               const double &r,
                                               const double &h) {

  DrawableSegmentSoup G;
  G.default_color = Color::GREEN();
  auto thetas = subdivide_angles((int)dir.size());
  for (auto i = 0; i < dir.size(); ++i) {
    auto curr_pos = vec2d{r * std::cos(thetas[i]), r * std::sin(thetas[i])};
    auto p = evaluate_quadric(pc.patches[vid].quadric, curr_pos);
    auto v = dir[i];
    v.normalize();
    auto p0 = p + h * v;
    G.push_seg(p, p0);
  }

  return G;
}
pair<vec2d, int> closes_nei(const vector<pair<vec2d, int>> &p_nbr,
                            const vec2d &p) {
  auto closest = -1;
  auto d = DBL_MAX;
  for (auto i = 0; i < p_nbr.size(); ++i) {
    auto len = (p_nbr[i].first - p).norm();
    if (len < d) {
      d = len;
      closest = i;
    }
  }

  return p_nbr[closest];
}

void show_singularities(point_cloud &pc, const vector<double> &f) {
  auto sing = singularities(pc, f);
  auto col0 = Color::CYAN();
  auto col1 = Color::MAGENTA();
  for (auto i = 0; i < sing.size(); ++i) {
    auto &curr = pc.points[sing[i].x()];

    if (sing[i].y() < 0) {
      curr = DrawableSphere(curr.center, 3 * curr.radius, col0);
    } else {
      curr = DrawableSphere(curr.center, 3 * curr.radius, col1);
    }
  }
}
void show_singularities(point_cloud &pc, const DrawableTrimesh<> &m,
                        const vector<double> &f) {
  auto sing = singularities(m, f);
  auto col0 = Color::RED();
  auto col1 = Color::BLUE();
  auto col2 = Color::GREEN();
  for (auto i = 0; i < sing.size(); ++i) {
    auto &curr = pc.points[sing[i].x()];
    if (sing[i].y() == -1)
      curr = DrawableSphere(curr.center, 3 * curr.radius, col1);
    else if (sing[i].y() == 0)
      curr = DrawableSphere(curr.center, 3 * curr.radius, col2);
    else
      curr = DrawableSphere(curr.center, 3 * curr.radius, col0);
  }
}

void show_singularities(point_cloud &pc) {
  auto sing = singularities(pc);
  auto col0 = random_color();
  auto col1 = random_color();
  for (auto i = 0; i < sing.size(); ++i) {
    auto &curr = pc.points[sing[i].x()];

    if (sing[i].y() < 0) {
      auto curr_col = Color(col0.x(), col0.y(), col0.z());
      curr = DrawableSphere(curr.center, curr.radius, curr_col);
    } else {
      auto curr_col = Color(col1.x(), col1.y(), col1.z());
      curr = DrawableSphere(curr.center, curr.radius, curr_col);
    }
  }
}
void show_singularities(point_cloud &pc, const vector<vec2i> &sing,
                        const double &size) {
  auto col0 = Color::RED();
  auto col1 = Color::BLUE();
  auto col2 = Color::GREEN();

  for (auto i = 0; i < sing.size(); ++i) {
    auto &curr = pc.points[sing[i].x()];
    auto rad = (size >= 0) ? size : 2 * curr.radius;
    if (sing[i].y() == -1)
      curr = DrawableSphere(curr.center, rad, col1);
    else if (sing[i].y() == 0)
      curr = DrawableSphere(curr.center, rad, col2);
    else
      curr = DrawableSphere(curr.center, rad, col0);
  }
}
void write_file(const vector<double> &v, const string &filename) {

  std::ofstream outfile;
  outfile.open(filename);
  if (outfile.fail())
    std::cout << "didn't mangae to open the file" << std::endl;
  for (auto i = 0; i < v.size(); ++i) {
    outfile << v[i] << "\n";
  }

  outfile.close();
}
void save_landmarks(const vector<int> &lndmarks, const string &filename) {

  std::ofstream outfile;
  outfile.open(filename);
  if (outfile.fail())
    std::cout << "didn't mangae to open the file" << std::endl;
  for (auto i = 0; i < lndmarks.size(); ++i) {
    outfile << lndmarks[i] << "\n";
  }

  outfile.close();
}

vector<int> load_landmarks(const string &filename) {

  std::ifstream f;
  f.open(filename);
  if (!f.is_open()) {
    std::cerr << "File loading the landamrks has not been found";
  }
  auto result = vector<int>{};
  while (true) {
    int vid;
    f >> vid;
    if (f.eof())
      break;
    result.push_back(vid);
  }
  f.close();
  return result;
}

Eigen::SparseMatrix<double> import_laplacian(const string &filename,
                                             const int V) {

  std::ifstream f;
  f.open(filename);
  if (!f.is_open()) {
    std::cerr << "File loading the landamrks has not been found";
  }
  Eigen::SparseMatrix<double> result(V, V);
  vector<Eigen::Triplet<double>> entries;
  typedef Eigen::Triplet<double> T;

  while (true) {
    int row, col;
    double val;
    f >> row >> col >> val;
    if (f.eof())
      break;
    entries.push_back(T(row - 1, col - 1, val));
  }
  f.close();
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}
void export_matrix(const Eigen::MatrixXi &M, const string &filename) {
  std::ofstream outfile;
  outfile.open(filename);
  if (outfile.fail())
    std::cout << "didn't mangae to open the file" << std::endl;

  for (auto i = 0; i < M.rows(); ++i) {
    for (auto j = 0; j < M.cols(); ++j) {
      outfile << M(i, j);
      if (j != M.cols() - 1)
        outfile << ",";
    }

    outfile << "\n";
  }

  outfile.close();
}
template <typename M> M load_csv(const std::string &path) {
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<double> values;
  uint rows = 0;
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      values.push_back(std::stod(cell));
    }
    ++rows;
  }
  return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime,
                          M::ColsAtCompileTime, RowMajor>>(
      values.data(), rows, values.size() / rows);
}
void export_matrix(const Eigen::MatrixXd &M, const string &filename) {
  std::ofstream outfile;
  outfile.open(filename);
  if (outfile.fail())
    std::cout << "didn't mangae to open the file" << std::endl;

  for (auto i = 0; i < M.rows(); ++i) {
    for (auto j = 0; j < M.cols(); ++j) {
      outfile << M(i, j);
      if (j != M.cols() - 1)
        outfile << ",";
    }

    outfile << "\n";
  }

  outfile.close();
}
void write_binary_matrix(const char *filename, const Eigen::MatrixXd &matrix) {
  std::ofstream out(filename,
                    std::ios::out | std::ios::binary | std::ios::trunc);
  if (out.fail())
    std::cout << "did not manage to open the file" << std::endl;
  typename Eigen::MatrixXd::Index rows = matrix.rows(), cols = matrix.cols();
  out.write((char *)(&rows), sizeof(typename Eigen::MatrixXd::Index));
  out.write((char *)(&cols), sizeof(typename Eigen::MatrixXd::Index));
  out.write((char *)matrix.data(),
            rows * cols * sizeof(typename Eigen::MatrixXd::Scalar));
  out.close();
}

Eigen::MatrixXd read_binary_matrix(const char *filename) {
  Eigen::MatrixXd result;
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  if (in.fail())
    std::cout << "did not manage to open the file to load the matrix"
              << std::endl;
  Eigen::MatrixXd::Index rows = 0, cols = 0;
  in.read((char *)(&rows), sizeof(typename Eigen::MatrixXd::Index));
  in.read((char *)(&cols), sizeof(typename Eigen::MatrixXd::Index));
  result.resize(rows, cols);
  in.read((char *)result.data(),
          rows * cols * sizeof(typename Eigen::MatrixXd::Scalar));
  in.close();
  return result;
}
int find_key(const unordered_map<int, int> &map, const int value) {
  auto it = find_if(std::begin(map), std::end(map),
                    [&value](auto &p) { return p.second == value; });

  if (it == std::end(map))
    return -1;

  return it->first;
}
void export_lndmarks(const vector<int> &lndmarks, const string &name,
                     const string &foldername) {
  string filename = foldername + "/landmarks_" + name;

  std::ofstream outfile;

  outfile.open(filename);

  for (auto i = 0; i < lndmarks.size(); ++i) {
    outfile << lndmarks[i] + 1 << "\n";
  }

  outfile.close();
}
void export_centers(const vector<int> &centers, const string &name,
                    const string &foldername) {
  string filename = foldername + "/centers_" + name;

  std::ofstream outfile;

  outfile.open(filename);

  for (auto i = 0; i < centers.size(); ++i) {
    outfile << centers[i] + 1 << "\n";
  }

  outfile.close();
}
void export_field(const vector<double> &field, const string &name) {

  std::ofstream outfile;

  outfile.open(name);

  for (auto i = 0; i < field.size(); ++i) {
    outfile << field[i] << "\n";
  }

  outfile.close();
}
void export_lndmarks(const vector<int> &lndmarks, const string &name0,
                     const string &name1, const string &foldername,
                     const unordered_map<int, int> &m02,
                     const unordered_map<int, int> &m13) {
  string filename0 = foldername + "/landmarks_" + name0;
  string filename1 = foldername + "/landmarks_" + name1;
  std::ofstream outfile0;
  std::ofstream outfile1;

  outfile0.open(filename0);
  outfile1.open(filename1);

  for (auto i = 0; i < lndmarks.size(); i += 2) {
    outfile0 << m02.at(lndmarks[i]) + 1 << "\n";
    outfile1 << m13.at(lndmarks[i + 1]) + 1 << "\n";
  }

  outfile0.close();
  outfile1.close();
}

void export_centers(const vector<int> &centers0, const vector<int> &centers1,
                    const string &name0, const string &name1,
                    const string &foldername,
                    const unordered_map<int, int> &mapping,
                    const unordered_map<int, int> &m02,
                    const unordered_map<int, int> &m13) {
  string filename0 = foldername + "/centers_" + name0;
  string filename1 = foldername + "/centers_" + name1;
  std::ofstream outfile0;
  std::ofstream outfile1;

  outfile0.open(filename0);
  outfile1.open(filename1);

  auto re_ordered = vector<int>(centers0.size());
  for (auto i = 0; i < centers0.size(); ++i) {
    auto c1 = mapping.at(centers0[i]);
    auto entry =
        distance(centers1.begin(), find(centers1.begin(), centers1.end(), c1));
    re_ordered[entry] = centers0[i];
  }
  for (auto i = 0; i < centers1.size(); ++i) {
    outfile0 << m02.at(re_ordered[i]) << "\n";
    outfile1 << m13.at(centers1[i]) << "\n";
  }

  outfile0.close();
  outfile1.close();
}
void export_centersGT(const iVd &vor0, const iVd &vor1,
                      const vector<int> &centers0, const vector<int> &centers1,
                      const string &name0, const string &name1,
                      const string &foldername,
                      const unordered_map<int, int> &m02,
                      const unordered_map<int, int> &m13) {
  string filename0 = foldername + "/centers_" + name0;
  string filename1 = foldername + "/centers_" + name1;
  std::ofstream outfile0;
  std::ofstream outfile1;

  outfile0.open(filename0);
  outfile1.open(filename1);

  // auto re_ordered = vector<int>(centers0.size());
  // for (auto i = 0; i < centers0.size(); ++i) {
  //   auto entry =
  //       distance(centers1.begin(), find(centers1.begin(), centers1.end(),
  //                                       vor1.voronoi_tags[centers0[i]]));
  //   re_ordered[entry] = centers0[i];
  // }
  for (auto i = 0; i < centers0.size(); ++i)
    outfile0 << m02.at(centers0[i]) << "\n";

  for (auto i = 0; i < centers1.size(); ++i)
    outfile1 << m13.at(centers1[i]) << "\n";

  outfile0.close();
  outfile1.close();
}

void export_lndmarks(const vector<int> &lndmarks, const string &name0,
                     const string &name1, const string &foldername) {
  string filename0 = foldername + "/landmarks_" + name0;
  string filename1 = foldername + "/landmarks_" + name1;
  std::ofstream outfile0;
  std::ofstream outfile1;

  outfile0.open(filename0);
  outfile1.open(filename1);

  for (auto i = 0; i < lndmarks.size(); i += 2) {
    outfile0 << lndmarks[i] + 1 << "\n";
    outfile1 << lndmarks[i + 1] + 1 << "\n";
  }

  outfile0.close();
  outfile1.close();
}
std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> compute_indicator_functions(
    const iVd &vor0, const vector<int> &centers0, const iVd &vor1,
    const vector<int> &centers1, const unordered_map<int, int> &mapping,
    const unordered_map<int, int> &m20, const unordered_map<int, int> &m31,
    const int V2, const int V3) {
  auto n = centers0.size();
  Eigen::MatrixXi F(V2, n);
  Eigen::MatrixXi G(V3, n);

  F.setZero();
  G.setZero();

  for (auto i = 0; i < V3; ++i) {
    auto tag = vor1.voronoi_tags[m31.at(i)];
    auto it = find(centers1.begin(), centers1.end(), tag);
    if (it == centers1.end()) {
      std::cout << "We messed up" << std::endl;
      return {};
    }
    assert(it != centers1.end());
    auto entry = distance(centers1.begin(), it);

    G(i, entry) = 1;
  }

  for (auto i = 0; i < V2; ++i) {
    auto tag = vor0.voronoi_tags[m20.at(i)];
    auto it = find(centers1.begin(), centers1.end(), mapping.at(tag));
    if (it == centers1.end()) {
      std::cout << "We messed up" << std::endl;
      return {};
    }
    assert(it != centers1.end());
    auto entry = distance(centers1.begin(), it);
    F(i, entry) = 1;
  }
  return {F, G};
}

iVd voronoi_diagram_from_matrix(const Eigen::MatrixXd &F,
                                const vector<int> &centers) {
  iVd result;

  result.voronoi_tags.resize(F.rows());
  result.voronoi_regions.resize(F.cols());

  for (auto i = 0; i < F.rows(); ++i) {
    for (auto j = 0; j < F.cols(); ++j) {
      if (F(i, j) == 1) {
        result.voronoi_tags[i] = centers[j];
        result.voronoi_regions[j].push_back(i);
      }
    }
  }

  return result;
}
std::tuple<iVd, vector<int>> import_voronoi_diagrams(const string &name,
                                                     const string folder) {
  auto matrix_name = folder + +"matrix_" + name;
  auto centers_name = folder + +"centers_" + name;
  Eigen::MatrixXd F = load_csv<MatrixXd>(matrix_name.c_str());
  auto centers = load_landmarks(centers_name);
  auto result = voronoi_diagram_from_matrix(F, centers);

  return {result, centers};
}
bool export_regions(
    const iVd &vor0, const vector<int> &centers0, const iVd &vor1,
    const vector<int> &centers1, const vector<int> &landmarks,
    const unordered_map<int, int> &mapping, const unordered_map<int, int> &m20,
    const unordered_map<int, int> &m02, const unordered_map<int, int> &m13,
    const unordered_map<int, int> &m31, const int V2, const int V3,
    const string &foldername, const string &name0, const string &name1) {

  auto [F, G] = compute_indicator_functions(vor0, centers0, vor1, centers1,
                                            mapping, m20, m31, V2, V3);

  string filename0 = foldername + "/matrix_" + name0;
  string filename1 = foldername + "/matrix_" + name1;
  export_matrix(F, filename0);
  export_matrix(G, filename1);
  export_centers(centers0, centers1, name0, name1, foldername, mapping, m02,
                 m13);
  export_lndmarks(landmarks, name0, name1, foldername, m02, m13);

  return true;
}
bool export_regions_GT(
    const iVd &vor0, const vector<int> &centers0, const iVd &vor1,
    const vector<int> &centers1, const vector<int> &landmarks,
    const unordered_map<int, int> &mapping, const unordered_map<int, int> &m20,
    const unordered_map<int, int> &m02, const unordered_map<int, int> &m13,
    const unordered_map<int, int> &m31, const int V2, const int V3,
    const string &foldername, const string &name0, const string &name1) {
  auto n = centers0.size();
  Eigen::MatrixXi F(V2, n);
  Eigen::MatrixXi G(V3, n);

  F.setZero();
  G.setZero();

  for (auto i = 0; i < V3; ++i) {
    auto tag = vor1.voronoi_tags[m31.at(i)];
    auto it = find(centers1.begin(), centers1.end(), tag);
    if (it == centers1.end()) {
      std::cout << "We messed up" << std::endl;
      return false;
    }
    assert(it != centers1.end());
    auto entry = distance(centers1.begin(), it);

    G(i, entry) = 1;
  }

  for (auto i = 0; i < V2; ++i) {
    auto tag = vor0.voronoi_tags[m20.at(i)];
    auto it = find(centers1.begin(), centers1.end(), mapping.at(tag));
    if (it == centers1.end()) {
      std::cout << "We messed up" << std::endl;
      return false;
    }
    assert(it != centers1.end());
    auto entry = distance(centers1.begin(), it);
    F(i, entry) = 1;
  }

  string filename0 = foldername + "/matrix_" + name0;
  string filename1 = foldername + "/matrix_" + name1;
  export_matrix(F, filename0);
  export_matrix(G, filename1);
  export_lndmarks(landmarks, name0, name1, foldername);

  return true;
}
bool export_regions_GT(const iVd &vor0, const vector<int> &centers0,
                       const iVd &vor1, const vector<int> &centers1,
                       const unordered_map<int, int> &m20,
                       const unordered_map<int, int> &m02,
                       const unordered_map<int, int> &m13,
                       const unordered_map<int, int> &m31, const int V2,
                       const int V3, const string &foldername,
                       const string &name0, const string &name1) {
  auto n = centers0.size();
  Eigen::MatrixXi F(V2, n);
  Eigen::MatrixXi G(V3, n);

  F.setZero();
  G.setZero();

  for (auto i = 0; i < V3; ++i) {
    auto tag = vor1.voronoi_tags[m31.at(i)];
    auto it = find(centers1.begin(), centers1.end(), tag);
    if (it == centers1.end()) {
      std::cout << "We messed up" << std::endl;
      return false;
    }
    assert(it != centers1.end());
    auto entry = distance(centers1.begin(), it);

    G(i, entry) = 1;
  }

  for (auto i = 0; i < V2; ++i) {
    auto tag = vor0.voronoi_tags[m20.at(i)];
    auto tag1 = vor1.voronoi_tags[tag];
    auto it = find(centers1.begin(), centers1.end(), tag1);
    if (it == centers1.end()) {
      std::cout << "We messed up" << std::endl;
      return false;
    }
    assert(it != centers1.end());
    auto entry = distance(centers1.begin(), it);
    F(i, entry) = 1;
  }

  string filename0 = foldername + "/matrix_" + name0;
  string filename1 = foldername + "/matrix_" + name1;
  export_matrix(F, filename0);
  export_matrix(G, filename1);
  export_centersGT(vor0, vor1, centers0, centers1, name0, name1, foldername,
                   m02, m13);

  return true;
}
bool export_regions_general(const iVd &vor0, const vector<int> &centers0,
                            const iVd &vor1, const vector<int> &centers1,
                            const unordered_map<int, int> &m20,
                            const unordered_map<int, int> &m02,
                            const unordered_map<int, int> &m13,
                            const unordered_map<int, int> &m31, const int V2,
                            const int V3, const string &foldername,
                            const string &name0, const string &name1) {
  auto n = centers0.size();
  Eigen::MatrixXi F(V2, n);
  Eigen::MatrixXi G(V3, n);

  F.setZero();
  G.setZero();

  for (auto i = 0; i < V3; ++i) {
    auto tag = vor1.voronoi_tags[m31.at(i)];
    auto it = find(centers1.begin(), centers1.end(), tag);
    if (it == centers1.end()) {
      std::cout << "We messed up" << std::endl;
      return false;
    }
    assert(it != centers1.end());
    auto entry = distance(centers1.begin(), it);

    G(i, entry) = 1;
  }

  for (auto i = 0; i < V2; ++i) {
    auto tag = vor0.voronoi_tags[m20.at(i)];
    auto it = find(centers0.begin(), centers0.end(), tag);
    if (it == centers0.end()) {
      std::cout << "We messed up" << std::endl;
      return false;
    }
    assert(it != centers0.end());
    auto entry = distance(centers0.begin(), it);
    F(i, entry) = 1;
  }

  string filename0 = foldername + "/matrix_" + name0;
  string filename1 = foldername + "/matrix_" + name1;
  export_matrix(F, filename0);
  export_matrix(G, filename1);
  export_centersGT(vor0, vor1, centers0, centers1, name0, name1, foldername,
                   m02, m13);

  return true;
}
bool export_regions_GT_remeshed(
    const iVd &vor0, const vector<int> &centers0, const iVd &vor1,
    const vector<int> &centers1, const vector<int> &l0, const vector<int> &l1,
    const unordered_map<int, int> &mapping, const unordered_map<int, int> &m20,
    const unordered_map<int, int> &m02, const unordered_map<int, int> &m13,
    const unordered_map<int, int> &m31, const int V2, const int V3,
    const string &foldername, const string &name0, const string &name1) {
  auto n = centers0.size();
  Eigen::MatrixXi F(V2, n);
  Eigen::MatrixXi G(V3, n);

  F.setZero();
  G.setZero();

  for (auto i = 0; i < V3; ++i) {
    auto tag = vor1.voronoi_tags[m31.at(i)];
    auto it = find(centers1.begin(), centers1.end(), tag);
    if (it == centers1.end()) {
      std::cout << "We messed up" << std::endl;
      return false;
    }
    assert(it != centers1.end());
    auto entry = distance(centers1.begin(), it);

    G(i, entry) = 1;
  }

  for (auto i = 0; i < V2; ++i) {
    auto tag = vor0.voronoi_tags[m20.at(i)];
    auto it = find(centers1.begin(), centers1.end(), mapping.at(tag));
    if (it == centers1.end()) {
      std::cout << "We messed up" << std::endl;
      return false;
    }
    assert(it != centers1.end());
    auto entry = distance(centers1.begin(), it);
    F(i, entry) = 1;
  }

  string filename0 = foldername + "/matrix_" + name0;
  string filename1 = foldername + "/matrix_" + name1;
  export_matrix(F, filename0);
  export_matrix(G, filename1);
  export_lndmarks(l0, name0, foldername);
  export_lndmarks(l1, name1, foldername);

  return true;
}
bool check_mapping(const unordered_map<int, int> &m20,
                   const DrawableTrimesh<> &m) {
  for (auto i = 0; i < m.num_verts(); ++i) {
    if (m20.find(i) == m20.end())
      return false;
  }

  return true;
}

std::tuple<iVd, vector<int>> transfer_voronoi_diagram(
    const iVd &vor0, const vector<int> &centers0,
    const unordered_map<int, int> &m20, const unordered_map<int, int> &m02,
    const DrawableTrimesh<> &m0, const DrawableTrimesh<> &m2) {
  auto centers = vector<int>(centers0.size());
  if (!check_mapping(m20, m2))
    std::cout << "Error! mapping m20 is not correct" << std::endl;
  if (!check_mapping(m02, m0))
    std::cout << "Error! mapping m02 is not correct" << std::endl;

  for (auto i = 0; i < centers.size(); ++i)
    centers[i] = m02.at(centers0[i]);

  iVd result;

  result.voronoi_tags.resize(m2.num_verts());
  for (auto i = 0; i < m2.num_verts(); ++i) {
    auto point_on_m0 = m20.at(i);
    auto tag_on_m0 = vor0.voronoi_tags[point_on_m0];

    auto center_on_m2 = m02.at(tag_on_m0);
    if (find(centers.begin(), centers.end(), center_on_m2) == centers.end())
      std::cout << "This point is not correctly mapped" << std::endl;
    result.voronoi_tags[i] = center_on_m2;
  }

  return {result, centers};
}
double region_to_region_accuracy(const iVd &vor0, const iVd &vor1,
                                 const vector<int> &centers0,
                                 const unordered_map<int, int> &mapping,
                                 const unordered_map<int, int> &m02,
                                 const unordered_map<int, int> &m31,
                                 const int entry) {
  auto region0 = vor0.voronoi_regions[entry];
  auto k = region0.size();
  auto count = 0;
  for (auto i = 0; i < k; ++i) {
    auto curr = region0[i];
    auto phi0 = m02.at(curr);
    auto phi3 = m31.at(phi0);
    auto tag = vor1.voronoi_tags[phi3];
    if (tag == mapping.at(centers0[entry]))
      ++count;
  }

  return (double)count / k * 100;
}
double region_to_region_accuracy_GT(const iVd &vor0, const iVd &vor1,
                                    const iVd &vor4, const iVd &vor5,
                                    const unordered_map<int, int> &m02,
                                    const int entry) {
  auto region0 = vor0.voronoi_regions[entry];
  auto k = region0.size();
  auto count = 0;
  for (auto i = 0; i < k; ++i) {
    auto curr = region0[i];
    auto vid0 = m02.at(curr);
    auto tag0 = vor4.voronoi_tags[vid0];
    auto tag1 = vor5.voronoi_tags[vid0];
    if (tag0 == tag1)
      ++count;
  }
  if (count != k)
    std::cout << "Not 100%% accurate" << std::endl;

  return (double)count / k * 100;
}
double region_to_region_accuracy_GT(const iVd &vor0, const iVd &vor1,
                                    const int curr_center, const int entry) {
  auto region0 = vor0.voronoi_regions[entry];
  auto k = region0.size();
  if (k == 0)
    return 0.0;
  auto count = 0;
  for (auto i = 0; i < k; ++i) {
    auto curr = region0[i];
    auto tag = vor1.voronoi_tags[curr];
    if (tag == curr_center)
      ++count;
  }

  return (double)count / k * 100;
}
vec3d accuracy_of_region_mapping(
    const iVd &vor0, const vector<int> &centers0, const iVd &vor1,
    const vector<int> &centers1, const vector<vector<double>> &d0,
    const vector<vector<double>> &d1, const unordered_map<int, int> &mapping,
    const unordered_map<int, int> &m20, const unordered_map<int, int> &m02,
    const unordered_map<int, int> &m13, const unordered_map<int, int> &m31) {
  auto n = centers0.size();
  auto count_center = 0;
  auto region_overlap = 0.0;
  auto accuracy = vec3d{0, 0, 0};
  for (auto i = 0; i < n; ++i) {

    auto curr_center = mapping.at(centers0[i]);
    auto phi1 = m13.at(curr_center);
    auto vid_to_check = m20.at(phi1);
    auto tag = vor0.voronoi_tags[vid_to_check];
    region_overlap +=
        region_to_region_accuracy(vor0, vor1, centers0, mapping, m02, m31, i);
    if (tag == centers0[i])
      ++count_center;
  }

  accuracy.x() = (double)count_center / n * 100;
  accuracy.y() = region_overlap / n;
  auto err = 0.0;
  // for (auto i = 0; i < d0.size(); ++i) {
  //   for (auto j = 0; j < n; ++j) {
  //     auto fxi = mapping.at(centers0[j]);
  //     auto entryfxi = distance(centers1.begin(),
  //                              find(centers1.begin(), centers1.end(),
  //                              fxi));

  //     err += std::abs(d1[i][fxi] - d0[i][centers0[j]]);
  //   }
  // }

  err /= n;
  accuracy.z() = err;

  return accuracy;
}
double accuracy_of_region_mapping_GT(const iVd &vor0, const iVd &vor1,
                                     const unordered_map<int, int> &m20,
                                     const unordered_map<int, int> &m31,
                                     const unordered_map<int, int> &m13,
                                     const unordered_map<int, int> &m02,
                                     const int V5) {
  double region_overlap = 0;
  for (auto i = 0; i < V5; ++i) {
    auto vid0 = m20.at(i);
    auto vid1 = m31.at(i);
    auto tag1 = vor1.voronoi_tags[vid1];
    auto tag0 = vor0.voronoi_tags[vid0];
    auto tag_center = vor0.voronoi_tags[m20.at(m13.at(tag1))];
    if (tag_center == tag0)
      ++region_overlap;
  }

  auto acc = region_overlap / V5;

  return acc;
}
double accuracy_of_region_mapping_GT(const iVd &vor0, const iVd &vor1,
                                     const int V0) {

  double region_overlap = 0;
  for (auto i = 0; i < V0; ++i) {
    auto tag1 = vor1.voronoi_tags[i];
    auto tag0 = vor0.voronoi_tags[i];
    auto tag_center = vor0.voronoi_tags[tag1];
    if (tag0 == tag_center)
      ++region_overlap;
  }

  region_overlap /= V0;

  return region_overlap;
}
unordered_map<int, int> shape_correspondence(const vector<vec3d> &pos0,
                                             const vector<vec3d> &pos1) {
  unordered_map<int, int> result{};
  for (auto i = 0; i < pos0.size(); ++i) {
    auto curr_pos = pos0[i];
    auto nearest = -1;
    auto d = DBL_MAX;
    for (auto j = 0; j < pos1.size(); ++j) {
      auto curr = (curr_pos - pos1[j]).norm();
      if (curr < d) {
        d = curr;
        nearest = j;
      }
    }

    result[i] = nearest;
  }

  return result;
}
vector<double> obj_wrapper(const vector<vec3d> &pos) {
  auto result = vector<double>(3 * pos.size());
  for (auto i = 0; i < pos.size(); ++i) {
    result[3 * i] = pos[i].x();
    result[3 * i + 1] = pos[i].y();
    result[3 * i + 2] = pos[i].z();
  }

  return result;
}
void clean_filename(string &filename, const string &substring) {
  auto it = filename.find(substring);
  if (it != std::string::npos)
    filename.erase(it, substring.length());
}
std::tuple<vector<string>, vector<string>> set_pairing_for_non_iso_matching() {
  srand(0);
  auto result0 = vector<string>{};
  auto result1 = vector<string>{};
  result0.reserve(120);
  result1.reserve(120);
  auto matching_for_zero = vector<int>{1, 2, 3, 4, 5, 6, 8};
  auto matching_for_one = vector<int>{2, 3, 8};
  auto matching_for_two = vector<int>{4, 5, 6, 7, 8};
  auto matching_for_three = vector<int>{4, 5, 6, 7, 8, 9};
  auto matching_for_four = vector<int>{7, 9};
  auto matching_for_five = vector<int>{7, 9};
  auto matching_for_six = vector<int>{7, 9};
  auto matching_for_eight = vector<int>{9};
  auto pairings = vector<vector<int>>{};
  pairings = {matching_for_zero,  matching_for_one,  matching_for_two,
              matching_for_three, matching_for_four, matching_for_five,
              matching_for_six,   matching_for_eight};
  auto indices = vector<int>{0, 1, 2, 3, 4, 5, 6, 8};
  for (auto i = 0; i < pairings.size(); ++i) {
    for (auto p : pairings[i]) {
      for (auto j = 0; j < 4; ++j) {
        auto r0 = rand() % 9;
        result0.push_back("tr_reg_0" + to_string(indices[i]) + to_string(r0));
        result1.push_back("tr_reg_0" + to_string(p) + to_string(r0));
      }
    }
  }

  return {result0, result1};
}
vector<double> error_on_michaels(const Eigen::MatrixXd &distances,
                                 const vector<int> &T, const double &diam) {

  auto n = (int)distances.rows();
  auto result = vector<double>(n);
  for (auto i = 0; i < n; ++i) {
    result[i] = distances(i, T[i] - 1) / diam;
  }

  return result;
}
