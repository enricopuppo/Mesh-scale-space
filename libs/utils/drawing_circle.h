#pragma once

#include "struct.h"
#include <diff_geo/diff_geo.h>
#include <realtime/gpu.h>
#include <utils/utilities.h>
#include <vector>
#include <yocto/yocto_mesh.h>
using gpu::Shape, gpu::Shader;
using std::vector;
using yocto::mesh_point, yocto::vec3i, yocto::vec3f, yocto::mat4f, yocto::vec2i;

struct closed_curve {
  vector<int> strip = {};
  vector<float> lerps = {};
};
struct circle_tids {
  float lerp = -1;
  int offset = -1;
};
typedef vector<closed_curve> Isoline;

struct Circle {
  mesh_point center;
  float radius;
  Isoline isoline;
  vector<int> tids = {};
  vector<vec3f> pos = {};
  vector<float> distances = {};
  float theta = 0.f;
  int primitive = -1;
  int construction = -1;
  vector<mesh_point> vertices = {};
  bool has_been_edited = false;
  int levels = -1;
};
struct Ellipse {
  mesh_point midpoint;
  float scale_factor;
  Isoline isoline;
  vector<float> distances_from_midpoint = {};
};

bool set_radius(const vector<vec3i> &triangles,
                const vector<vec3i> &adjacencies, Circle *circle,
                const float &radius);

Circle create_circle(const shape_data &mesh, const shape_geometry &geometry,
                     const geodesic_solver &solver, const mesh_point &center,
                     const float &radius);

Circle create_circle(const vector<vec3i> &triangles,
                     const vector<vec3f> &positions,
                     const vector<vec3i> &adjacencies, const mesh_point &center,
                     const float &radius, const vector<float> &distances);

Circle create_circle(const shape_data &mesh, const shape_geometry &geometry,
                     const geodesic_solver &solver, const mesh_point &center,
                     const mesh_point &point);

vector<vec3f> circle_positions(const vector<vec3i> &triangles,
                               const vector<vec3f> &positions,
                               const vector<vec3i> &adjacencies,
                               const Circle &c0);

mesh_point get_closed_curve_point(const vector<vec3i> &triangles,
                                  const vector<vec3f> &positions,
                                  const vector<vec3i> &adjacencies,
                                  const closed_curve &curve, int ix);

vector<vec3f> closed_curve_positions(const closed_curve &curve,
                                     const vector<vec3i> &triangles,
                                     const vector<vec3f> &positions,
                                     const vector<vec3i> &adjacencies,
                                     const vec2i &range = zero2i);
Isoline create_isoline(const vector<vec3i> &triangles,
                       const vector<vec3i> &adjacencies,
                       const vector<float> &distances, const float &radius);

Isoline create_isoline(const vector<vec3i> &triangles,
                       const vector<vec3i> &adjacencies,
                       const vector<float> &d0, const vector<float> &d1);

inline float get_distance(const vec3i &triangle, const vec3f bary,
                          const vector<float> &distances) {
  return dot(bary, vec3f{distances[triangle.x], distances[triangle.y],
                         distances[triangle.z]});
}
template <typename T> inline int find_in_vec(const T &vec, int x) {
  for (int i = 0; i < size(vec); i++)
    if (vec[i] == x)
      return i;
  return -1;
}