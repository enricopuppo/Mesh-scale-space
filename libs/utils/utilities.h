#ifndef UTILITIES_H
#define UTILITIES_H

#include <diff_geo/diff_geo.h>
#include <fstream>
#include <utils/mesh_io.h>
#include <yocto/yocto_sceneio.h>

mesh_point eval_mesh_point(const vector<vec3i> &triangles,
                           const vector<vec3f> &positions, const int &face,
                           const vec3f &point);

vector<vec3f> PCE_grad(const Eigen::SparseMatrix<double> &G,
                       const Eigen::VectorXd &f, const int F);
vector<vec3f> AGS_grad(const Eigen::SparseMatrix<double> &G,
                       const Eigen::VectorXd &f);

vector<vector<float>> fields_from_lndmrks(const shape_data &data,
                                          const shape_geometry &geometry,
                                          const geodesic_solver &geo_solver,
                                          const vector<mesh_point> &lndmrks,
                                          const int solver);
void update_fields_from_lndmrks(const shape_data &data,
                                const shape_geometry &geometry,
                                const geodesic_solver &geo_solver,
                                const vector<mesh_point> &lndmrks,
                                const int solver, const vector<bool> &moved,
                                vector<vector<float>> &f);

vector<vector<vec3f>> grads_from_lndmrks(const shape_op &op,
                                         const vector<vector<float>> &fields);

void update_grads_from_lndmrks(const shape_op &op,
                               const vector<vector<float>> &f,
                               const vector<bool> &moved,
                               vector<vector<vec3f>> &grds);

vector<vec3f> bézier_curve(const shape_data &data,
                           const shape_geometry &geometry, const shape_op &op,
                           const geodesic_solver &geo_solver,
                           const dual_geodesic_solver &solver,
                           const vector<mesh_point> &control_points,
                           const int k, const int type_of_solver);

vector<vector<mesh_point>> paired_samples(const shape_data &data,
                                          const shape_geometry &geometry,
                                          const vector<mesh_point> &gamma0,
                                          const vector<mesh_point> &gamma1,
                                          const int N);
vector<mesh_point> geodesic_average(const shape_data &data,
                                    const shape_geometry &geometry,
                                    const dual_geodesic_solver &solver,
                                    const vector<mesh_point> &gamma0,
                                    const vector<mesh_point> &gamma1,
                                    const int N);
vector<mesh_point>
geodesic_average(const shape_data &data, const shape_geometry &geometry,
                 const dual_geodesic_solver &solver, const mesh_point &p0,
                 const mesh_point &p3, const vec2f &d0, const vec2f &d1,
                 const float &theta0, const float &theta1, const float &lambda1,
                 const float &lambda2, const float &lambda3, const float &len);

std::tuple<vector<mesh_point>, vector<mesh_point>>
geodesic_average_w_control_points(const shape_data &data,
                                  const shape_geometry &geometry,
                                  const dual_geodesic_solver &solver,
                                  const mesh_point &p0, const mesh_point &p3,
                                  const vec2f &d0, const vec2f &d1,
                                  const float &theta0, const float &theta1,
                                  const float &lambda1, const float &lambda2,
                                  const float &lambda3, const float &len);

vector<mesh_point>
equiangular_polygon(const shape_data &data, const shape_geometry &geometry,
                    const dual_geodesic_solver &solver,
                    const vector<mesh_point> &vertices, const float &lambda1,
                    const float &lambda2, const float &lambda3,
                    const bool as_euclidean_as_possibile = true);

vector<vec3f> rational_bézier_curve(
    const shape_data &data, const shape_geometry &geometry, const shape_op &op,
    const geodesic_solver &geo_solver, const dual_geodesic_solver &solver,
    const vector<mesh_point> &control_points, const vector<float> &weights,
    const int k = 8, const int method = graph);
vector<vec3f> rational_bézier_curve(
    const shape_data &data, const shape_geometry &geometry, const shape_op &op,
    const dual_geodesic_solver &solver, const geodesic_solver &geo_solver,
    const vector<vector<float>> &f, const vector<vector<vec3f>> &grds,
    const vector<mesh_point> &control_points, const vector<float> &weights,
    const int k);
vector<vec3f> béz_interp_rational_curve(
    const shape_data &data, const shape_geometry &geometry, const shape_op &op,
    const geodesic_solver &geo_solver, const dual_geodesic_solver &solver,
    const vector<vector<float>> &f, const vector<vector<vec3f>> &grds,
    const vector<mesh_point> &control_points, const vector<float> &betas,
    const int k);

std::tuple<vector<vector<vec3f>>, vector<vector<mesh_point>>>
trace_bspline(const shape_data &data, const shape_geometry &geometry,
              const shape_op &op, const dual_geodesic_solver &solver,
              const vector<vector<float>> &fields,
              const vector<vector<vec3f>> &grds,
              const vector<mesh_point> &control_points, const int k);

std::tuple<vector<vector<vec3f>>, vector<vector<mesh_point>>>
trace_rational_bspline(const shape_data &data, const shape_geometry &geometry,
                       const shape_op &op, const dual_geodesic_solver &solver,
                       const vector<vector<float>> &fields,
                       const vector<vector<vec3f>> &grds,
                       const vector<mesh_point> &control_points,
                       const vector<float> &weights, const int k);

void update_rational_spline(
    const shape_data &data, const shape_geometry &geometry, const shape_op &op,
    const dual_geodesic_solver &solver, const vector<vector<float>> &fields,
    const vector<vector<vec3f>> &grds, const vector<mesh_point> &control_points,
    const vector<float> &weights, const int k, const int moved_point,
    vector<vector<vec3f>> &pos, vector<vector<mesh_point>> &points);

vector<mesh_point> get_polyline_from_path(const vector<vec3i> &triangles,
                                          const vector<vec3i> &adjacencies,
                                          const geodesic_path &path);

vector<vector<mesh_point>>
geodesic_sheaf(const shape_data &data, const shape_geometry &geometry,
               const dual_geodesic_solver &solver, const mesh_point &p0,
               const mesh_point &p1, const int total_number_of_curve = 5);

vector<vec3f> jacobi_vector_field(const shape_data &data,
                                  const shape_geometry &geometry,
                                  const vector<mesh_point> &gamma,
                                  const float &L);

inline vector<vec3f> jacobi_vector_field(const shape_data &data,
                                         const shape_geometry &geometry,
                                         const geodesic_path &gamma,
                                         const float &L = 1) {
  return jacobi_vector_field(
      data, geometry,
      get_polyline_from_path(data.triangles, geometry.adjacencies, gamma), L);
}

inline bool validate_points(const vector<mesh_point> &list) {
  for (auto point : list) {
    if (point.face == -1)
      return false;
  }

  return true;
}

#endif