
#include "logging.h"

#include "drawing_polygons.h"
#include <algorithm>
#include <yocto/yocto_math.h>

using namespace logging;
using std::tie;

vector<mesh_point> make_n_gon(const shape_data &mesh,
                              const shape_geometry &geometry,
                              const Circle *circle, const int n) {
  if (circle == nullptr)
    return {};
  auto theta = circle->theta;
  auto radius = circle->radius;
  auto center = circle->center;

  auto step = 2 * pif / (float)n;
  auto result = vector<mesh_point>(n);
  auto flat_tid =
      init_flat_triangle(mesh.positions, mesh.triangles[center.face], 0);
  auto e = normalize(flat_tid[1]);
  for (auto i = 0; i < n; ++i) {
    result[i] =
        polthier_straightest_geodesic(mesh, geometry, center,
                                      rot_vect(e, -(theta + i * step)), radius)
            .back();
  }

  return result;
}
// put lambda=1 for a rectangle
vector<mesh_point> parallelogram_tangent_space(const shape_data &mesh,
                                               const shape_geometry &geometry,
                                               const Circle *circle,
                                               const float &sigma,
                                               const float &lambda,
                                               const float &theta) {
  auto r = circle->radius;
  auto center = circle->center;
  auto flat_tid =
      init_flat_triangle(mesh.positions, mesh.triangles[center.face], 0);
  auto e = normalize(flat_tid[1]);
  auto theta_sigma =
      std::acos((2 * pow(r, 2) - 2 * pow(sigma * r, 2)) / (2 * pow(r, 2)));
  auto p0 = polthier_straightest_geodesic(mesh, geometry, center,
                                          rot_vect(e, -theta), r * lambda)
                .back();
  auto p1 = polthier_straightest_geodesic(
                mesh, geometry, center, rot_vect(e, -(theta + theta_sigma)), r)
                .back();
  auto p2 = polthier_straightest_geodesic(
                mesh, geometry, center, rot_vect(e, -(theta + pif)), r * lambda)
                .back();
  auto p3 =
      polthier_straightest_geodesic(
          mesh, geometry, center, rot_vect(e, -(theta + pif + theta_sigma)), r)
          .back();

  return {p0, p1, p2, p3};
}
inline bool intersect_point(const ray3f &ray, const vec3f &p, float &dist) {
  // find parameter for line-point minimum distance
  auto w = p - ray.o;
  auto t = dot(w, ray.d) / dot(ray.d, ray.d);

  // exit if not within bounds
  if (t < ray.tmin - flt_eps || t > ray.tmax + flt_eps) {
    return false;
  }

  // test for line-point distance vs point radius
  auto rp = ray.o + ray.d * t;
  auto prp = p - rp;
  if (dot(prp, prp) > flt_eps * flt_eps)
    return false;

  // intersection occurred: set params and exit
  dist = t;
  return true;
}
inline bool intersect_line(const ray3f &ray, const vec3f &p0, const vec3f &p1,
                           float &dist) {
  // setup intersection params
  auto u = ray.d;
  auto v = p1 - p0;
  auto w = ray.o - p0;

  // compute values to solve a linear system
  auto a = dot(u, u);
  auto b = dot(u, v);
  auto c = dot(v, v);
  auto d = dot(u, w);
  auto e = dot(v, w);
  auto det = a * c - b * b;

  // check determinant and exit if lines are parallel
  // (could use EPSILONS if desired)
  if (det == 0)
    return false;

  // compute Parameters on both ray and segment
  auto t = (b * e - c * d) / det;
  auto s = (a * e - b * d) / det;

  // exit if not within bounds
  if (t < ray.tmin - 1e-4F || t > ray.tmax + 1e-4F || s < -1e-4F ||
      s > 1 + 1e-4F) {
    return false;
  }

  dist = t;
  return true;
}
mesh_point intersect_segments(const vector<vec3i> &triangles,
                              const vector<vec3f> &positions, const int &tid,
                              const vec3f &start1, const vec3f &end1,
                              const vec3f &start2, const vec3f &end2) {
  if (start1 - end1 == zero3f && start2 - end2 == zero3f) {
    return mesh_point();
  }

  float dist;
  auto c = positions[triangles[tid].x];
  auto n = tid_normal(triangles, positions, tid);

  auto s1 = project_point(start1, c, n);
  auto e1 = project_point(end1, c, n);
  auto s2 = project_point(start2, c, n);
  auto e2 = project_point(end2, c, n);

  if (start1 - end1 == zero3f) {
    auto ray = ray3f{s2, e2 - s2, 0.0F, 1.0F};

    if (intersect_point(ray, s1, dist)) {
      return eval_mesh_point(triangles, positions, tid, ray_point(ray, dist));
      ;
    }
  } else if (start2 - end2 == zero3f) {
    auto ray = ray3f{s1, e1 - s1, 0.0F, 1.0F};

    if (intersect_point(ray, s2, dist)) {
      return eval_mesh_point(triangles, positions, tid, ray_point(ray, dist));
      ;
    }
  } else {
    auto ray = ray3f{s1, e1 - s1, 0.0F, 1.0F};

    if (intersect_line(ray, s2, e2, dist)) {
      return eval_mesh_point(triangles, positions, tid, ray_point(ray, dist));
    }
  }

  return mesh_point();
}

mesh_point intersect_segments(const vector<vec3i> &triangles,
                              const vector<vec3f> &positions, const int &tid,
                              const mesh_point &start1, const mesh_point &end1,
                              const mesh_point &start2,
                              const mesh_point &end2) {
  auto b = intersect_segments(triangles, positions, tid,
                              eval_position(triangles, positions, start1),
                              eval_position(triangles, positions, end1),
                              eval_position(triangles, positions, start2),
                              eval_position(triangles, positions, end2));

  return b;
}
mesh_point intersect(const shape_data &mesh, const shape_geometry &geometry,
                     const vector<mesh_point> &first,
                     const vector<mesh_point> &second) {
  auto &triangles = mesh.triangles;
  auto &positions = mesh.positions;

  vector<mesh_point> intersection;

  auto f = polyline_pos(mesh.triangles, mesh.positions, first);
  auto s = polyline_pos(mesh.triangles, mesh.positions, second);

  for (auto i = 0; i < first.size(); ++i) {
    for (auto j = 0; j < second.size(); ++j) {
      if (first[i].face == second[j].face) {
        auto point = intersect_segments(triangles, positions, first[i].face,
                                        f[i], f[i + 1], s[j], s[j + 1]);

        if (point.face != -1) {
          return point;
        }
      }
    }
  }

  return mesh_point();
}
vec3f path_pos_from_entry(const vector<vec3i> &triangles,
                          const vector<vec3f> &positions,
                          const vector<vec3i> &adjacencies,
                          const geodesic_path &path, int entry) {
  auto u = path.lerps[entry];
  auto eid = get_edge(triangles, positions, adjacencies, path.strip[entry],
                      path.strip[entry + 1]);
  if (eid.x < 0) {
    assert(path.strip.size() == 1);
    return eval_position(triangles, positions, path.end);
  }
  auto p0 = positions[eid.x];
  auto p1 = positions[eid.y];
  return (1 - u) * p0 + u * p1;
}
inline mesh_point eval_point_from_lerp(const int tid, const int offset,
                                       const float &lerp) {
  auto bary = zero3f;
  bary[offset] = 1 - lerp;
  bary[(offset + 1) % 3] = lerp;
  return {tid, vec2f{bary.y, bary.z}};
}
mesh_point mesh_point_from_isoline_entry(const vector<vec3i> &triangles,
                                         const vector<vec3i> &adjacencies,
                                         const vector<int> &strip,
                                         const vector<float> &lerps,
                                         const int entry) {
  auto lerp = lerps[entry];
  auto tid0 = strip[entry];
  auto tid1 = strip[(entry + 1) % strip.size()];
  auto offset = find_in_vec(adjacencies[tid0], tid1);
  if (offset == -1)
    return mesh_point{};

  return eval_point_from_lerp(tid0, offset, lerp);
}
vec3f path_pos_from_entry_in_isoline(const vector<vec3i> &triangles,
                                     const vector<vec3f> &positions,
                                     const vector<vec3i> &adjacencies,
                                     const vector<int> &strip,
                                     const vector<float> &lerps, int entry) {

  auto u = lerps[entry];
  auto eid = get_edge(triangles, positions, adjacencies, strip[entry],
                      strip[(entry + 1) % strip.size()]);
  auto p0 = positions[eid.x];
  auto p1 = positions[eid.y];
  return (1 - u) * p0 + u * p1;
}

pair<mesh_point, mesh_point> intersect(const shape_data &mesh,
                                       const shape_geometry &geometry,
                                       const geodesic_path &path,
                                       const Isoline &isoline) {

  auto crossed_by_first = vector<bool>(mesh.triangles.size(), false);
  unordered_map<int, pair<mesh_point, mesh_point>> face_to_point;

  auto first_interesection_point = mesh_point{};
  auto second_interesection_point = mesh_point{};
  auto &strip = path.strip;
  for (auto i = 0; i < strip.size(); ++i) {
    crossed_by_first[strip[i]] = true;
    if (i == 0) {
      auto k = find_in_vec(geometry.adjacencies[path.start.face], strip[1]);
      auto next = eval_point_from_lerp(path.start.face, k, path.lerps[0]);
      face_to_point[strip[i]] = std::make_pair(path.start, next);
    } else if (i == strip.size() - 1) {
      auto k = find_in_vec(geometry.adjacencies[strip.back()], path.end.face);
      auto prev = eval_point_from_lerp(strip.back(), k, path.lerps.back());
      face_to_point[strip[i]] = std::make_pair(prev, path.end);
    } else {
      auto curr = strip[i];
      auto prev = strip[i - 1];
      if (face_to_point.count(curr))
        continue;
      auto h = find_in_vec(geometry.adjacencies[curr], prev);
      auto k = find_in_vec(geometry.adjacencies[prev], curr);
      face_to_point[curr] =
          std::make_pair(eval_point_from_lerp(curr, h, 1 - path.lerps[i - 1]),
                         eval_point_from_lerp(curr, k, path.lerps[i]));
    }
  }

  for (auto &curve : isoline) {
    auto s = curve.strip.size();
    for (auto i = 0; i < s; ++i)
      if (crossed_by_first[curve.strip[i]]) {
        auto first_segment = face_to_point.at(curve.strip[i]);
        auto prev = curve.strip[(s + i - 1) % s];
        auto curr = curve.strip[i];
        auto k = find_in_vec(geometry.adjacencies[curr], prev);
        auto start2 = eval_point_from_lerp(curve.strip[i], k,
                                           1 - curve.lerps[(s + i - 1) % s]);
        auto end2 = mesh_point_from_isoline_entry(
            mesh.triangles, geometry.adjacencies, curve.strip, curve.lerps, i);
        if (end2.face == -1)
          std::cerr << "This point should be well defined" << std::endl;
        auto second_segment = std::make_pair(start2, end2);

        auto lerp = intersect_line_segments(
            mesh.triangles, mesh.positions, curve.strip[i], first_segment.first,
            first_segment.second, second_segment.first, second_segment.second);
        if (lerp < -1e-3 || lerp > 1 + 1e-3)
          continue;
        auto bary0 = second_segment.first.uv;
        auto bary1 = second_segment.second.uv;
        auto bary =
            (1 - lerp) * vec3f{1 - bary0.x - bary0.y, bary0.x, bary0.y} +
            lerp * vec3f{1 - bary1.x - bary1.y, bary1.x, bary1.y};

        if (first_interesection_point.face == -1)
          first_interesection_point =
              mesh_point{curve.strip[i], {bary.y, bary.z}};
        else
          second_interesection_point =
              mesh_point{curve.strip[i], {bary.y, bary.z}};
      }
  }
  return {first_interesection_point, second_interesection_point};
}

std::tuple<mesh_point, mesh_point, vec2i, vec2i>
intersect_with_entries(const shape_data &mesh, const shape_geometry &geometry,
                       const vector<mesh_point> &path, const Isoline &isoline) {

  auto crossed_by_first = vector<bool>(mesh.triangles.size(), false);
  unordered_map<int, pair<mesh_point, mesh_point>> face_to_point;
  unordered_map<int, int> face_to_entries;
  auto range_first = zero2i;
  auto range_second = zero2i;
  auto first_interesection_point = mesh_point{};
  auto second_interesection_point = mesh_point{};

  for (auto i = 0; i < path.size(); ++i) {
    crossed_by_first[path[i].face] = true;
    if (i == 0)
      face_to_point[path[i].face] = std::make_pair(path[0], path[1]);
    else {
      auto curr = path[i];
      auto prev = path[i - 1];
      if (face_to_point.count(curr.face))
        continue;
      else if (curr.face != prev.face) {
        auto h = find_in_vec(geometry.adjacencies[curr.face], prev.face);
        auto k = find_in_vec(geometry.adjacencies[prev.face], curr.face);
        if (h == -1) {
          auto [curr_is_vert, kc] = point_is_vert(curr);
          auto [prev_is_vert, kp] = point_is_vert(prev);
          if (prev_is_vert) {
            auto vid = mesh.triangles[prev.face][kp];
            auto kv = find_in_vec(mesh.triangles[curr.face], vid);
            if (kv == -1)
              std::cerr << "This offset shouldn't be -1" << std::endl;
            face_to_point[curr.face] =
                std::make_pair(eval_point_from_lerp(curr.face, kv, 0.f), curr);
          } else
            std::cerr << "This shouldn't happens" << std::endl;
        } else {
          auto bary = vec3f{1 - prev.uv.x - prev.uv.y, prev.uv.x, prev.uv.y};
          face_to_point[curr.face] =
              std::make_pair(eval_point_from_lerp(curr.face, h, bary[k]), curr);
        }
      } else
        face_to_point[curr.face] = std::make_pair(prev, curr);
    }
    face_to_entries[path[i].face] = i;
  }

  for (auto &curve : isoline) {
    auto s = curve.strip.size();
    for (auto i = 0; i < s; ++i)
      if (crossed_by_first[curve.strip[i]]) {
        auto first_segment = face_to_point.at(curve.strip[i]);
        auto prev = curve.strip[(s + i - 1) % s];
        auto curr = curve.strip[i];
        auto k = find_in_vec(geometry.adjacencies[curr], prev);
        auto start2 = eval_point_from_lerp(curve.strip[i], k,
                                           1 - curve.lerps[(s + i - 1) % s]);
        auto end2 = mesh_point_from_isoline_entry(
            mesh.triangles, geometry.adjacencies, curve.strip, curve.lerps, i);
        if (end2.face == -1)
          std::cerr << "This point should be well defined" << std::endl;
        auto second_segment = std::make_pair(start2, end2);

        auto lerp = intersect_line_segments(
            mesh.triangles, mesh.positions, curve.strip[i], first_segment.first,
            first_segment.second, second_segment.first, second_segment.second);
        if (lerp < -1e-3 || lerp > 1 + 1e-3)
          continue;
        auto bary0 = second_segment.first.uv;
        auto bary1 = second_segment.second.uv;
        auto bary =
            (1 - lerp) * vec3f{1 - bary0.x - bary0.y, bary0.x, bary0.y} +
            lerp * vec3f{1 - bary1.x - bary1.y, bary1.x, bary1.y};

        if (first_interesection_point.face == -1) {
          first_interesection_point =
              mesh_point{curve.strip[i], {bary.y, bary.z}};
          range_second.x = i;
          range_first.x = face_to_entries.at(curr);
        } else {
          second_interesection_point =
              mesh_point{curve.strip[i], {bary.y, bary.z}};
          range_second.y = i;
          range_first.y = face_to_entries.at(curr);
        }
      }
  }
  return {first_interesection_point, second_interesection_point, range_first,
          range_second};
}
pair<mesh_point, mesh_point> intersect(const shape_data &mesh,
                                       const shape_geometry &geometry,
                                       const vector<mesh_point> &path,
                                       const Isoline &isoline) {

  auto crossed_by_first = vector<bool>(mesh.triangles.size(), false);
  unordered_map<int, pair<mesh_point, mesh_point>> face_to_point;

  auto first_interesection_point = mesh_point{};
  auto second_interesection_point = mesh_point{};

  for (auto i = 0; i < path.size(); ++i) {
    crossed_by_first[path[i].face] = true;
    if (i == 0)
      face_to_point[path[i].face] = std::make_pair(path[0], path[1]);
    else {
      auto curr = path[i];
      auto prev = path[i - 1];
      if (face_to_point.count(curr.face))
        continue;
      else if (curr.face != prev.face) {
        auto h = find_in_vec(geometry.adjacencies[curr.face], prev.face);
        auto k = find_in_vec(geometry.adjacencies[prev.face], curr.face);
        if (h == -1) {
          auto [curr_is_vert, kc] = point_is_vert(curr);
          auto [prev_is_vert, kp] = point_is_vert(prev);
          if (prev_is_vert) {
            auto vid = mesh.triangles[prev.face][kp];
            auto kv = find_in_vec(mesh.triangles[curr.face], vid);
            if (kv == -1)
              std::cerr << "This offset shouldn't be -1" << std::endl;
            face_to_point[curr.face] =
                std::make_pair(eval_point_from_lerp(curr.face, kv, 0.f), curr);
          } else
            std::cerr << "This shouldn't happens" << std::endl;
        } else {
          auto bary = vec3f{1 - prev.uv.x - prev.uv.y, prev.uv.x, prev.uv.y};
          face_to_point[curr.face] =
              std::make_pair(eval_point_from_lerp(curr.face, h, bary[k]), curr);
        }
      } else
        face_to_point[curr.face] = std::make_pair(prev, curr);
    }
  }

  for (auto &curve : isoline) {
    auto s = curve.strip.size();
    for (auto i = 0; i < s; ++i)
      if (crossed_by_first[curve.strip[i]]) {
        auto first_segment = face_to_point.at(curve.strip[i]);
        auto prev = curve.strip[(s + i - 1) % s];
        auto curr = curve.strip[i];
        auto k = find_in_vec(geometry.adjacencies[curr], prev);
        auto start2 = eval_point_from_lerp(curve.strip[i], k,
                                           1 - curve.lerps[(s + i - 1) % s]);
        auto end2 = mesh_point_from_isoline_entry(
            mesh.triangles, geometry.adjacencies, curve.strip, curve.lerps, i);
        if (end2.face == -1)
          std::cerr << "This point should be well defined" << std::endl;
        auto second_segment = std::make_pair(start2, end2);

        auto lerp = intersect_line_segments(
            mesh.triangles, mesh.positions, curve.strip[i], first_segment.first,
            first_segment.second, second_segment.first, second_segment.second);
        if (lerp < -1e-3 || lerp > 1 + 1e-3)
          continue;
        auto bary0 = second_segment.first.uv;
        auto bary1 = second_segment.second.uv;
        auto bary =
            (1 - lerp) * vec3f{1 - bary0.x - bary0.y, bary0.x, bary0.y} +
            lerp * vec3f{1 - bary1.x - bary1.y, bary1.x, bary1.y};

        if (first_interesection_point.face == -1)
          first_interesection_point =
              mesh_point{curve.strip[i], {bary.y, bary.z}};
        else
          second_interesection_point =
              mesh_point{curve.strip[i], {bary.y, bary.z}};
      }
  }
  return {first_interesection_point, second_interesection_point};
}
pair<mesh_point, mesh_point> intersect(const shape_data &mesh,
                                       const shape_geometry &geometry,
                                       const geodesic_path &path,
                                       const Circle &circle) {
  return intersect(mesh, geometry, path, circle.isoline);
}
pair<mesh_point, mesh_point> intersect(const shape_data &mesh,
                                       const shape_geometry &geometry,
                                       const vector<mesh_point> &path,
                                       const Circle &circle) {
  return intersect(mesh, geometry, path, circle.isoline);
}

pair<mesh_point, mesh_point> intersect(const shape_data &mesh,
                                       const shape_geometry &geometry,
                                       const geodesic_path &path,
                                       const closed_curve &curve) {
  return intersect(mesh, geometry, path, Isoline{curve});
}

pair<mesh_point, mesh_point> intersect(const shape_data &mesh,
                                       const shape_geometry &geometry,
                                       const Isoline &first,
                                       const Isoline &second) {
  auto crossed_by_first = vector<bool>(mesh.triangles.size(), false);
  unordered_map<int, pair<mesh_point, mesh_point>> face_to_point;
  auto first_interesection_point = mesh_point{};
  auto second_interesection_point = mesh_point{};
  for (auto &curve : first) {
    auto s0 = curve.strip.size();
    for (auto i = 0; i < s0; ++i) {
      crossed_by_first[curve.strip[i]] = true;
      auto prev = curve.strip[(s0 + i - 1) % s0];
      auto curr = curve.strip[i];
      auto k = find_in_vec(geometry.adjacencies[curr], prev);
      auto start1 =
          eval_point_from_lerp(curr, k, 1 - curve.lerps[(s0 + i - 1) % s0]);
      auto end1 = mesh_point_from_isoline_entry(
          mesh.triangles, geometry.adjacencies, curve.strip, curve.lerps, i);
      face_to_point[curve.strip[i]] = std::make_pair(start1, end1);
    }
  }
  for (auto &curve : second) {
    auto s1 = curve.strip.size();
    for (auto i = 0; i < s1; ++i)
      if (crossed_by_first[curve.strip[i]] == true) {
        auto first_segment = face_to_point.at(curve.strip[i]);
        auto prev = curve.strip[(s1 + i - 1) % s1];
        auto curr = curve.strip[i];
        auto k = find_in_vec(geometry.adjacencies[curr], prev);
        auto start2 =
            eval_point_from_lerp(curr, k, 1 - curve.lerps[(s1 + i - 1) % s1]);
        auto end2 = mesh_point_from_isoline_entry(
            mesh.triangles, geometry.adjacencies, curve.strip, curve.lerps, i);
        if (end2.face == -1)
          std::cerr << "This point should be well defined" << std::endl;
        auto second_segment = std::make_pair(start2, end2);

        auto lerp = intersect_line_segments(
            mesh.triangles, mesh.positions, curve.strip[i], first_segment.first,
            first_segment.second, second_segment.first, second_segment.second);
        if (lerp < -1e-3 || lerp > 1 + 1e-3)
          continue;
        auto bary0 = second_segment.first.uv;
        auto bary1 = second_segment.second.uv;
        auto bary =
            (1 - lerp) * vec3f{1 - bary0.x - bary0.y, bary0.x, bary0.y} +
            lerp * vec3f{1 - bary1.x - bary1.y, bary1.x, bary1.y};

        if (first_interesection_point.face == -1)
          first_interesection_point =
              mesh_point{curve.strip[i], {bary.y, bary.z}};
        else
          second_interesection_point =
              mesh_point{curve.strip[i], {bary.y, bary.z}};
      }
  }
  return {first_interesection_point, second_interesection_point};
}
mesh_point intersect(const shape_data &mesh, const shape_geometry &geometry,
                     const geodesic_path &first, const geodesic_path &second) {
  auto &triangles = mesh.triangles;
  auto &positions = mesh.positions;

  vector<mesh_point> intersection;

  auto f = path_positions(first, mesh.triangles, mesh.positions,
                          geometry.adjacencies);
  auto s = path_positions(second, mesh.triangles, mesh.positions,
                          geometry.adjacencies);

  for (auto i = 0; i < first.strip.size(); ++i) {
    for (auto j = 0; j < second.strip.size(); ++j) {
      if (first.strip[i] == second.strip[j]) {
        auto point = intersect_segments(triangles, positions, first.strip[i],
                                        f[i], f[i + 1], s[j], s[j + 1]);

        if (point.face != -1) {
          return point;
        }
      }
    }
  }

  return mesh_point();
}
mesh_point intersect(const shape_data &mesh, const shape_geometry &geometry,
                     const geodesic_path &first,
                     const vector<mesh_point> &second) {
  auto &triangles = mesh.triangles;
  auto &positions = mesh.positions;

  vector<mesh_point> intersection;

  auto f = path_positions(first, mesh.triangles, mesh.positions,
                          geometry.adjacencies);
  auto s = polyline_pos(mesh.triangles, mesh.positions, second);

  for (auto i = 0; i < first.strip.size(); ++i) {
    for (auto j = 0; j < second.size(); ++j) {
      if (first.strip[i] == second[j].face) {
        auto point = intersect_segments(triangles, positions, first.strip[i],
                                        f[i], f[i + 1], s[j], s[j + 1]);

        if (point.face != -1) {
          return point;
        }
      }
    }
  }

  return mesh_point();
}
pair<mesh_point, mesh_point> intersect(const shape_data &mesh,
                                       const shape_geometry &geometry,
                                       const Circle &first,
                                       const Circle &second) {
  return intersect(mesh, geometry, first.isoline, second.isoline);
}
Isoline find_segment_bisector_isolines(const shape_data &mesh,
                                       const shape_geometry &geometry,
                                       const geodesic_solver &solver,
                                       const geodesic_path &path) {
  auto dist0 =
      compute_geodesic_distances(solver, mesh.triangles, mesh.positions,
                                 geometry.adjacencies, {path.start});
  auto dist1 = compute_geodesic_distances(
      solver, mesh.triangles, mesh.positions, geometry.adjacencies, {path.end});

  return create_isoline(mesh.triangles, geometry.adjacencies, dist0, dist1);
}

vector<mesh_point> altitude_isoscele_triangle(
    const shape_data &mesh, const shape_geometry &geometry,
    const geodesic_solver &solver, const dual_geodesic_solver &dual_solver,
    const mesh_point &a, const mesh_point &b, const float &len,
    const bool flipped) {
  auto path = compute_geodesic_path(mesh, geometry, dual_solver, a, b);
  auto iso = find_segment_bisector_isolines(mesh, geometry, solver, path);
  auto midpoint = intersect(mesh, geometry, path, iso);

  auto c0 = create_circle(mesh, geometry, solver, midpoint.first, len);
  auto [i0, i1] = intersect(mesh, geometry, iso, c0.isoline);

  if (flipped)
    return {a, i0, b};
  return {a, i1, b};
}
vector<mesh_point> equilateral_triangle(
    const shape_data &mesh, const shape_geometry &geometry,
    const geodesic_solver &solver, const dual_geodesic_solver &dual_solver,
    const mesh_point &a, const mesh_point &b, const bool flipped) {
  auto len =
      path_length(compute_geodesic_path(mesh, geometry, dual_solver, a, b),
                  mesh.triangles, mesh.positions, geometry.adjacencies);
  auto c0 = create_circle(mesh, geometry, solver, a, len);
  auto c1 = create_circle(mesh, geometry, solver, b, len);

  auto [i0, i1] = intersect(mesh, geometry, c0, c1);
  if (flipped)
    return {a, b, i1};
  return {a, b, i0};
}
vector<mesh_point> same_lengths_isoscele_triangle(
    const shape_data &mesh, const shape_geometry &geometry,
    const geodesic_solver &solver, const mesh_point &a, const mesh_point &b,
    const float &len, const bool flipped) {

  auto c0 = create_circle(mesh, geometry, solver, a, len);
  auto c1 = create_circle(mesh, geometry, solver, b, len);
  auto [i0, i1] = intersect(mesh, geometry, c0, c1);
  if (flipped)
    return {a, i1, b};

  return {a, b, i0};
}
vector<mesh_point> euclidean_rectangle(const shape_data &mesh,
                                       const shape_geometry &geometry,
                                       const geodesic_solver &solver,
                                       const dual_geodesic_solver &dual_solver,
                                       const geodesic_path &base,
                                       const float &height) {
  auto width =
      path_length(base, mesh.triangles, mesh.positions, geometry.adjacencies);
  auto d = tangent_path_direction(mesh, geometry, base, true);
  d = rot_vect(d, pif / 2);
  auto h =
      polthier_straightest_geodesic(mesh, geometry, base.start, d, 3 * height);
  auto c0 = create_circle(mesh, geometry, solver, base.start, height);
  auto c = intersect(mesh, geometry, h, c0);
  auto c1 = create_circle(mesh, geometry, solver, c.first, width);
  auto c2 = create_circle(mesh, geometry, solver, base.end, height);
  auto [d0, d1] = intersect(mesh, geometry, c1, c2);
  auto check0 = compute_geodesic_path(mesh, geometry, dual_solver, c.first, d0);
  auto check1 =
      compute_geodesic_path(mesh, geometry, dual_solver, d0, base.end);
  if (intersect(mesh, geometry, base, check0).face == -1 &&
      intersect(mesh, geometry, check1, h).face == -1)
    return {base.start, c.first, d0, base.end};

  return {base.start, c.first, d1, base.end};
}
vector<mesh_point> same_lengths_rectangle(const shape_data &mesh,
                                          const shape_geometry &geometry,
                                          const geodesic_solver &solver,
                                          const geodesic_path &base,
                                          const float &height) {
  auto d0 = tangent_path_direction(mesh, geometry, base, true);
  d0 = rot_vect(d0, pif / 2);
  auto h0 = polthier_straightest_geodesic(mesh, geometry, base.start,
                                          normalize(d0), 2 * height);
  auto d1 = tangent_path_direction(mesh, geometry, base, false);
  d1 = rot_vect(d1, -pif / 2);
  auto h1 = polthier_straightest_geodesic(mesh, geometry, base.end,
                                          normalize(d1), 2 * height);
  auto c0 = create_circle(mesh, geometry, solver, base.start, height);
  auto c1 = create_circle(mesh, geometry, solver, base.end, height);
  auto c = intersect(mesh, geometry, h0, c0);
  auto d = intersect(mesh, geometry, h1, c1);
  return {base.start, c.first, d.first, base.end};
}
vector<mesh_point> diagonal_rectangle(const shape_data &mesh,
                                      const shape_geometry &geometry,
                                      const geodesic_path &base,
                                      const geodesic_path &diagonal) {
  auto width =
      path_length(base, mesh.triangles, mesh.positions, geometry.adjacencies);

  auto base_tg = tangent_path_direction(mesh, geometry, base, true);
  auto diag_tg = tangent_path_direction(mesh, geometry, diagonal, true);
  auto diag_end_tg = tangent_path_direction(mesh, geometry, diagonal, false);

  auto theta = angle(base_tg, diag_tg);
  auto ad_dir = rot_vect(base_tg, pif / 2);
  auto h0 = polthier_straightest_geodesic(mesh, geometry, base.start, ad_dir,
                                          1.5 * width);

  auto cd_dir = rot_vect(diag_end_tg, -theta);
  auto cb_dir = rot_vect(diag_end_tg, pif / 2 - theta);

  auto ell = polthier_straightest_geodesic(mesh, geometry, diagonal.end,
                                           normalize(cd_dir), 1.5 * width);
  auto h1 = polthier_straightest_geodesic(mesh, geometry, diagonal.end,
                                          normalize(cb_dir), 1.5 * width);
  auto d = intersect(mesh, geometry, h0, ell);
  auto b = intersect(mesh, geometry, base, h1);

  return {base.start, d, diagonal.end, b};
}
vector<mesh_point> circle_control_points(const shape_data &data,
                                         const shape_geometry &geometry,
                                         const mesh_point &center,
                                         const float &r, const int n) {
  auto result = vector<mesh_point>(2 * n + 1);
  auto alpha = 2 * pif / (2 * n);
  auto R = r / std::cos(alpha);

  for (auto i = 0; i < n; ++i) {
    auto odd_v = vec2f{R * std::sin((2 * i + 1) * alpha),
                       -R * std::cos((2 * i + 1) * alpha)};
    auto even_v =
        vec2f{r * std::sin(2 * i * alpha), -r * std::cos(2 * i * alpha)};

    auto end = polthier_straightest_geodesic(data, geometry, center,
                                             normalize(even_v), length(even_v))
                   .back();
    result[2 * i] = end;
    end = polthier_straightest_geodesic(data, geometry, center,
                                        normalize(odd_v), length(odd_v))
              .back();
    result[2 * i + 1] = end;
  }

  result[2 * n] = result[0];
  return result;
}
vector<mesh_point> g1_circle_control_points(const shape_data &data,
                                            const shape_geometry &geometry,
                                            const mesh_point &center,
                                            const float &r, const int n) {
  auto result = vector<mesh_point>(2 * n + 1);
  auto sides = vector<vector<mesh_point>>(2 * n);
  auto alpha = 2 * pif / (2 * n);

  for (auto i = 0; i < n; ++i) {
    auto even_v =
        vec2f{r * std::sin(2 * i * alpha), -r * std::cos(2 * i * alpha)};

    auto gamma = polthier_straightest_geodesic(
        data, geometry, center, normalize(even_v), length(even_v));

    result[2 * i] = gamma.back();
    auto p0 = gamma.rbegin()[1];
    auto p1 = gamma.back();
    auto tr = init_flat_triangle(data.positions,
                                 data.triangles[gamma.back().face], 0);
    auto p0_prime = interpolate_triangle(tr[0], tr[1], tr[2], p0.uv);
    auto p1_prime = interpolate_triangle(tr[0], tr[1], tr[2], p1.uv);
    auto t = rot_vect(normalize(p1_prime - p0_prime), pif / 2);
    sides[2 * i] = polthier_straightest_geodesic(data, geometry, p1,
                                                 normalize(-t), 1.5 * r);
    sides[2 * i + 1] = polthier_straightest_geodesic(data, geometry, p1,
                                                     normalize(t), 1.5 * r);
  }
  result[2 * n] = result[0];
  for (auto i = 0; i < n; ++i) {

    result[2 * i + 1] = intersect(data, geometry, sides[2 * i + 1],
                                  sides[(2 * (i + 1)) % (2 * n)]);
  }

  return result;
}
std::tuple<vector<mesh_point>, vector<vector<mesh_point>>>
debug_g1_circle_control_points(const shape_data &data,
                               const shape_geometry &geometry,
                               const mesh_point &center, const float &r,
                               const int n) {
  auto result = vector<mesh_point>(n, mesh_point{0, zero2f});
  auto sides = vector<vector<mesh_point>>(2 * n);
  auto alpha = 2 * pif / (2 * n);

  for (auto i = 0; i < n; ++i) {
    auto even_v =
        vec2f{r * std::sin(2 * i * alpha), -r * std::cos(2 * i * alpha)};

    auto gamma = polthier_straightest_geodesic(
        data, geometry, center, normalize(even_v), length(even_v));

    auto p0 = gamma.rbegin()[1];
    auto p1 = gamma.back();
    auto tr = init_flat_triangle(data.positions,
                                 data.triangles[gamma.back().face], 0);
    auto p0_prime = interpolate_triangle(tr[0], tr[1], tr[2], p0.uv);
    auto p1_prime = interpolate_triangle(tr[0], tr[1], tr[2], p1.uv);
    auto t = rot_vect(normalize(p1_prime - p0_prime), pif / 2);
    sides[2 * i] = polthier_straightest_geodesic(data, geometry, p1,
                                                 normalize(-t), 1.5 * r);
    sides[2 * i + 1] = polthier_straightest_geodesic(data, geometry, p1,
                                                     normalize(t), 1.5 * r);
  }

  for (auto j = 0; j < n; ++j) {
    auto prev = 2 * (n + j - 1) % (2 * n);
    auto next = 2 * (j + 1) % (2 * n);

    auto p = intersect(data, geometry, sides[2 * j], sides[prev]);
    if (p.face != -1)
      result[j] = p;

    p = intersect(data, geometry, sides[2 * j + 1], sides[next]);

    if (p.face != -1)
      result[j] = p;
  }
  return {result, sides};
}
vector<vector<vec3f>>
trace_circle(const shape_data &data, const shape_geometry &geometry,
             const shape_op &op, const geodesic_solver &geo_solver,
             const dual_geodesic_solver &solver, const mesh_point &center,
             const float &r, const int num_segments, const int k,
             const int type_of_solver, const bool g1_circle) {
  auto control_points =
      (g1_circle)
          ? g1_circle_control_points(data, geometry, center, r, num_segments)
          : circle_control_points(data, geometry, center, r, num_segments);
  if (!validate_points(control_points)) {
    std::cout << "Something went wrong when computing the control points."
              << std::endl;
    return {};
  }
  auto f = fields_from_lndmrks(data, geometry, geo_solver, control_points,
                               type_of_solver);

  auto grds = grads_from_lndmrks(op, f);
  auto n_p = control_points.size();
  auto result = vector<vector<vec3f>>(num_segments);
  auto weights = vector<float>(3, 1);
  weights[1] = std::cos(2 * pif / (2 * num_segments));
  auto curr_points = vector<mesh_point>(3);
  auto curr_fields = vector<vector<float>>(3);
  auto curr_grads = vector<vector<vec3f>>(3);
  for (auto i = 0; i < num_segments; ++i) {

    curr_points[0] = control_points[2 * i];
    curr_points[1] = control_points[2 * i + 1];
    curr_points[2] = control_points[2 * i + 2];

    curr_fields[0] = f[2 * i];
    curr_fields[1] = f[2 * i + 1];
    curr_fields[2] = f[2 * i + 2];

    curr_grads[0] = grds[2 * i];
    curr_grads[1] = grds[2 * i + 1];
    curr_grads[2] = grds[2 * i + 2];
    result[i] =
        rational_bézier_curve(data, geometry, op, solver, geo_solver,
                              curr_fields, curr_grads, curr_points, weights, k);
  }

  return result;
}