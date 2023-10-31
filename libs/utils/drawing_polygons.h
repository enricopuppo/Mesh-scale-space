#pragma once

#include "drawing_circle.h"
#include <realtime/gpu.h>
#include <utils/utilities.h>
#include <vector>
#include <yocto/yocto_mesh.h>
#include <yocto_gui/yocto_shade.h>

using namespace std;
using namespace yocto;

enum rectangle_type { Diagonal, Euclidean, SameLengths };
enum triangle_type { Equilateral, IsoSameL, IsoBisector };
enum circle_type { Classical, G1 };

vector<mesh_point> make_n_gon(const shape_data &mesh,
                              const shape_geometry &geometry,
                              const Circle *circle, const int n);

vector<mesh_point> equilateral_triangle(
    const shape_data &mesh, const shape_geometry &geometry,
    const geodesic_solver &solver, const dual_geodesic_solver &dual_solver,
    const mesh_point &a, const mesh_point &b, const bool flipped);

vector<mesh_point> same_lengths_isoscele_triangle(
    const shape_data &mesh, const shape_geometry &geometry,
    const geodesic_solver &solver, const mesh_point &a, const mesh_point &b,
    const float &len, const bool flipped);

vector<mesh_point> altitude_isoscele_triangle(
    const shape_data &mesh, const shape_geometry &geometry,
    const geodesic_solver &solver, const dual_geodesic_solver &dual_solver,
    const mesh_point &a, const mesh_point &b, const float &len,
    const bool flipped);

vector<mesh_point> parallelogram_tangent_space(const shape_data &mesh,
                                               const shape_geometry &geometry,
                                               const Circle *circle,
                                               const float &sigma,
                                               const float &lambda,
                                               const float &theta);
vector<mesh_point> diagonal_rectangle(const shape_data &mesh,
                                      const shape_geometry &geometry,
                                      const geodesic_path &base,
                                      const geodesic_path &diagonal);

vector<mesh_point> same_lengths_rectangle(const shape_data &mesh,
                                          const shape_geometry &geometry,
                                          const geodesic_solver &solver,
                                          const geodesic_path &base,
                                          const float &height);

vector<mesh_point> euclidean_rectangle(const shape_data &mesh,
                                       const shape_geometry &geometry,
                                       const geodesic_solver &solver,
                                       const dual_geodesic_solver &dual_solver,
                                       const geodesic_path &base,
                                       const float &height);

std::tuple<vector<mesh_point>, vector<vector<mesh_point>>>
debug_g1_circle_control_points(const shape_data &data,
                               const shape_geometry &geometry,
                               const mesh_point &center, const float &r,
                               const int n);

vector<vector<vec3f>>
trace_circle(const shape_data &data, const shape_geometry &geometry,
             const shape_op &op, const geodesic_solver &geo_solver,
             const dual_geodesic_solver &solver, const mesh_point &center,
             const float &r, const int num_segments, const int k,
             const int type_of_solver, const bool g1_circle);
inline vec3f project_point(const vec3f &p, const vec3f &c, const vec3f &n) {
  auto v = p - c;
  auto proj = n * dot(v, n);

  return c + v - proj;
}