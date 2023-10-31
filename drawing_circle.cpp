#include "drawing_circle.h"

std::tuple<int, double> find_seed_in_circle(const DrawableTrimesh<> &m,
                                            const vector<double> &distances,
                                            const double &radius,
                                            const vector<bool> &parsed) {

  for (auto i = 0; i < m.num_edges(); ++i) {
    if (parsed[i])
      continue;
    auto e = m.edge_vert_ids(i);
    auto d = vec2d{distances[e[0]] - radius, distances[e[1]] - radius};
    if (d.x() * d.y() <= 0)
      return {i, -d.x() / (d.y() - d.x())};
  }
  return {-1, 0};
}
double edge_is_crossed_by_isoline(const DrawableTrimesh<> &m,
                                  const vector<double> &distances,
                                  const double &radius, const int eid) {
  auto e = m.edge_vert_ids(eid);
  auto d = vec2d{distances[e[0]] - radius, distances[e[1]] - radius};
  if (d.x() * d.y() <= 0)
    return -d.x() / (d.y() - d.x());

  return -1;
}
Isoline create_isoline(const DrawableTrimesh<> &m,
                       const vector<double> &distances, const double &radius) {
  Isoline iso = {};
  auto parsed = vector<bool>(m.num_edges(), false);
  auto [seed, lerp] = find_seed_in_circle(m, distances, radius, parsed);
  parsed[seed] = true;
  while (seed != -1) {
    auto curve = closed_curve{};

    std::deque<pair<int, double>> Q;
    Q.push_back(make_pair(seed, lerp));
    while (!Q.empty()) {
      auto curr = Q.back();
      Q.pop_back();
      curve.edges.push_back(curr.first);
      curve.lerps.push_back(curr.second);
      auto tids = m.adj_e2p(curr.first);
      auto tid0 = tids[0];
      auto tid1 = tids[1];
      auto found = false;
      for (auto j = 0; j < 3; ++j) {
        auto curr_e = m.poly_edge_id(tid0, j);
        if (parsed[curr_e])
          continue;
        parsed[curr_e] = true;
        auto alpha = edge_is_crossed_by_isoline(m, distances, radius, curr_e);
        if (alpha >= 0) {
          Q.push_back(make_pair(curr_e, alpha));
          found = true;
        }
      }
      if (!found) {
        for (auto j = 0; j < 3; ++j) {
          auto curr_e = m.poly_edge_id(tid1, j);
          if (parsed[curr_e])
            continue;
          parsed[curr_e] = true;
          auto alpha = edge_is_crossed_by_isoline(m, distances, radius, curr_e);
          if (alpha >= 0) {
            Q.push_back(make_pair(curr_e, alpha));
            found = true;
          }
        }
      }
    }

    if (curve.edges.size() > 0) {
      iso.push_back(curve);
    }
    std::tie(seed, lerp) = find_seed_in_circle(m, distances, radius, parsed);
  }
  return iso;
}

bool set_radius(const DrawableTrimesh<> &m, Circle *circle,
                const double &radius) {

  if (radius == 0.f)
    return true;

  circle->isoline = create_isoline(m, circle->distances, radius);

  circle->radius = radius;

  return true;
}

vec3d lerp(const vec3d &v0, const vec3d &v1, const double x) {
  return (1 - x) * v0 + x * v1;
}
vector<vec3d> closed_curve_positions(const closed_curve &curve,
                                     const DrawableTrimesh<> &m) {

  auto pos = vector<vec3d>(curve.lerps.size() + 1);
  auto s = curve.edges.size();
  for (auto i = 0; i < s; i++) {

    auto e = m.edge_vert_ids(curve.edges[i]);
    auto x = curve.lerps[i];
    auto p = lerp(m.vert(e[0]), m.vert(e[1]), x);
    pos[i] = p;
  }
  pos.back() = pos.front();

  return pos;
}
std::vector<vector<vec3d>> circle_positions(const DrawableTrimesh<> &m,
                                            const Circle &c0) {
  auto result = vector<vector<vec3d>>(c0.isoline.size());
  for (auto i = 0; i < c0.isoline.size(); ++i) {
    result[i] = closed_curve_positions(c0.isoline[i], m);
  }

  return result;
}
Circle create_circle(const DrawableTrimesh<> &m, const int center,
                     const double &radius, const vector<double> &distances) {
  auto circle = Circle();
  circle.center = center;
  circle.distances = distances;

  if (!set_radius(m, &circle, radius)) {
    return Circle();
  }

  return circle;
}

double normalize_circle_length(const DrawableTrimesh<> &m, const Circle &c) {
  auto pos = circle_positions(m, c);
  auto len = 0.0;
  for (auto curve : pos) {
    for (auto i = 0; i < curve.size() - 1; ++i) {
      len += (curve[i + 1] - curve[i]).norm();
    }
  }
  len /= (2 * M_PI * c.radius);

  return len;
}