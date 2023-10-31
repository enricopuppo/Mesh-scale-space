#ifndef DIFF_GEO_H
#define DIFF_GEO_H

#include "drawing_circle.h"
#include <ANN/ANN.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <VTP/geodesic_algorithm_exact.h>
#include <VTP/geodesic_mesh.h>
#include <cinolib/drawable_sphere.h>
#include <cinolib/geometry/aabb.h>
#include <cinolib/geometry/triangle_utils.h>
#include <cinolib/geometry/vec_mat.h>
#include <cinolib/laplacian.h>
#include <cinolib/matrix_eigenfunctions.h>
#include <cinolib/octree.h>
#include <cinolib/profiler.h>
#include <cinolib/tetgen_wrap.h>
using namespace std;
using namespace cinolib;

enum type_of_distance { geodesic, isophotic, Euclidean, Spectral, Biharmonic };

enum primitves {
  bilinear_patch,
  parabolic_cylinder,
  ellipsoid,
  hyperboloid,
  elliptic_paraboloid,
  elliptic_cylinder,
  hyperbolic_paraboloid,
  hyperbolic_cylinder,
  no_good
};
struct geodesic_solver {
  struct graph_edge {
    int node = -1;
    float length = DBL_MAX;
    float isophotic_length = DBL_MAX;
  };
  vector<vector<graph_edge>> graph = {};
};
struct iVd {
  // map of voronoi verts (triangles)
  vector<int> voronoi_verts = {};
  // vector having the same size of the mesh,
  // vornoi_tags[i]=closest voronoi center of i
  vector<int> voronoi_tags = {};
  // distance field from the voronoi
  // centers
  vector<double> distances = {};

  // regions
  vector<vector<int>> voronoi_regions = {};
  // maximum normal deviation of the vertices
  // in every region w.r.t to the normal at
  // the center
  vector<pair<double, int>> region_normal_deviation = {};

  // PCA of the region
  vector<vector<vec3d>> basis = {};

  // coordinates of the points in the plane
  // defined by PCA
  vector<vector<pair<vec2d, int>>> parametric_nbr = {};

  // maximum in distances
  float R = 0.f;
};
struct patch {
  Eigen::MatrixXd quadric;
  Eigen::VectorXd Monge_quadric;
  Eigen::MatrixXd C;
  vec3d xu;
  vec3d xv;
  vec3d xuu;
  vec3d xuv;
  vec3d xvv;
  vec3d Monge_normal;
  vector<double> weights;
  Eigen::MatrixXd res;
  vector<pair<vec2d, int>> parametric_nbr;
  vector<int> bad_ids;
  vector<vec2d> domain_range;
  vector<pair<vec2d, int>> CH;
  vector<int> tagged;
  vec3f color = vec3f{0, 0, 0};
  vec2d weights_range;
  double residual;
  bool is_center = false;
  bool expandable = true;
  int type = -1;
  vector<int> invaded;
  vector<vector<int>> stolen_from;
  vector<int> invasors;
  vector<vector<int>> stolen_by;
  vector<vec3d> e;
  Eigen::Vector2d u0;
};
inline float ccw(const pair<vec2d, int> &a, const pair<vec2d, int> &b,
                 const pair<vec2d, int> &c) {
  return (b.first.x() - a.first.x()) * (c.first.y() - a.first.y()) -
         (b.first.y() - a.first.y()) * (c.first.x() - a.first.x());
}
struct ccwSorter {
  const pair<vec2d, int> &pivot;

  ccwSorter(const pair<vec2d, int> &inPivot) : pivot(inPivot) {}

  bool operator()(const pair<vec2d, int> &a, const pair<vec2d, int> &b) {
    return ccw(pivot, a, b) < 0;
  }
};
struct point_cloud {
  vector<vec3d> positions = {};
  vector<vector<int>> patch_tagging = {};
  vector<vector<int>> nbrs = {};
  vector<vector<vec3d>> basis = {};
  vector<DrawableSphere> points = {};
  vector<int> ordered_patches = {};
  vector<patch> patches = {};
  vector<bool> badones = {};
  vector<vector<double>> residuals = {};
  geodesic_solver solver;
  Eigen::SparseMatrix<double> G;
  Eigen::SparseMatrix<double> L;
  ANNkd_tree *tree;
  double tau = 0.0;
  int num_patches = 0;
};
// init point cloud

std::tuple<vector<bool>, int> duplicated_vertices(const DrawableTrimesh<> &m);

double diameter(const Eigen::MatrixXd &G);

vector<int> knn(ANNkd_tree *tree, const vector<vec3d> &positions, const int vid,
                const int k);
point_cloud init_pc(vector<vec3d> &positions);

vector<vec3d> get_normals(const point_cloud &pc);

vector<vec3d> compute_grad(const point_cloud &pc, const vector<double> &field);

void clean_singularities(const DrawableTrimesh<> &m, const vector<vec3d> &grad,
                         vector<int> &singularities);
vector<vec3d> compute_grad_cino(const Eigen::SparseMatrix<double> &G,
                                const vector<double> &field);

vector<double> compute_laplacian(const Eigen::SparseMatrix<double> &L,
                                 const vector<double> &field);

vector<vec3d> compute_grad_MP(const point_cloud &pc,
                              const vector<double> &field);

vector<vec3d> compute_grad_slow(const point_cloud &pc,
                                const vector<double> &field);

geodesic_solver make_geodesic_solver(const DrawableTrimesh<> &m);

vector<double> compute_geodesic_distances(const geodesic_solver &solver,
                                          const vector<int> &sources,
                                          const int type = geodesic);

vector<double> exact_geodesic_distance(const DrawableTrimesh<> &m,
                                       const int &source);

vector<int> constrained_farthest_point_sampling(const geodesic_solver &solver,
                                                const DrawableTrimesh<> &mesh,
                                                const vector<int> &gt,
                                                const int seed, const int k);

Eigen::MatrixXd geodesic_matrix(const geodesic_solver &solver);

vector<int> farthest_point_sampling(const geodesic_solver &solver,
                                    const DrawableTrimesh<> &mesh,
                                    const int seed, const int k);

pair<iVd, vector<int>> farthest_point_sampling(
    const geodesic_solver &solver, const DrawableTrimesh<> &mesh,
    const vector<double> &phi, const Eigen::MatrixXd &D,
    const vector<int> seeds, const int type, const int k = -1);

void refine_voronoi_diagram(iVd &vor, vector<int> &vor_centers,
                            const geodesic_solver &solver,
                            const DrawableTrimesh<> &mesh,
                            const vector<double> &phi, const Eigen::MatrixXd &D,
                            const point_cloud &pc, const int type);

void move_voronoi_centers(iVd &vor, vector<int> &vor_centers,
                          const geodesic_solver &solver,
                          const DrawableTrimesh<> &mesh,
                          const vector<double> &phi, const Eigen::MatrixXd &D,
                          const point_cloud &pc, const int type);

pair<int, double> best_matching_center(const vector<double> &curr,
                                       const vector<vector<double>> &others);

void normalize_field(vector<double> &f, const double &M);

vector<vector<pair<double, int>>>
compute_descriptors(const DrawableTrimesh<> &m, const vector<int> &centers,
                    const vector<vector<double>> &dist);
double p_descr(const vector<pair<double, int>> &Dx,
               const vector<pair<double, int>> &Dy, const double sigma);
std::unordered_map<int, int>
voronoi_mapping(vector<int> &centers0, vector<int> &centers1,
                const vector<int> &lndmarks, const DrawableTrimesh<> &m0,
                const DrawableTrimesh<> &m1, const double &A0, const double &A1,
                const vector<vector<double>> &d0,
                const vector<vector<double>> &d1, const bool GH);

std::unordered_map<int, int> voronoi_mapping(vector<int> &centers0,
                                             vector<int> &centers1,
                                             const DrawableTrimesh<> &m0,
                                             const DrawableTrimesh<> &m1,
                                             const vector<vector<double>> &c0,
                                             const vector<vector<double>> &c1);
std::unordered_map<int, int> voronoi_mapping(
    vector<int> &centers0, vector<int> &centers1, const vector<int> &lndmarks,
    const DrawableTrimesh<> &m0, const DrawableTrimesh<> &m1,
    const vector<vector<double>> &l0, const vector<vector<double>> &l1,
    const vector<vector<double>> &c0, const vector<vector<double>> &c1);

ANNpoint ann_wrapper(const vec3d &p);
void rehorient_pc(point_cloud &pc, const vector<vec3d> &normals);
void rehorient_pc(point_cloud &pc);
// get 3D basis of tangent spaces
std::tuple<vector<vec3d>, vector<vec3d>, vector<vec3d>>
get_basis(const point_cloud &pc);

void fit_into_cube(vector<vec3d> &positions);

void patch_fitting(point_cloud &pc, const DrawableTrimesh<> &m,
                   const double &max_err, const bool isophotic = false);

void patch_fitting(point_cloud &pc, const DrawableTrimesh<> &m,
                   const vector<bool> &bad, const int goodones,
                   const double &max_err = 1e-3);
vector<int> compute_candidates(const vector<double> &phi, const int V,
                               const int nf);

vector<int> compute_candidates(const vector<double> &phi, const int V,
                               const int nf, const point_cloud &pc);

std::tuple<patch, int> best_fitting_primitive(point_cloud &pc, const int vid);

void Monge_patch_fitting(point_cloud &pc);

std::tuple<vector<vec3d>, vec2d, vec2d>
isophotic_geodesic(const patch &p, const int vid, const vec2d &uv_start,
                   const vec2d &dir);

vector<vec3d> isophotic_geodesic(const point_cloud &pc, const int start,
                                 const int target, const double &alpha);
std::tuple<vector<vec3d>, vec2d, vec3d>
isophotic_geodesic_along_principal_direction(
    const point_cloud &pc, const int vid, const bool flipped,
    const vec2d &uv_start = vec2d{0, 0},
    const vec3d &prev_dir = vec3d{0, 0, 0});

std::tuple<patch, int>
fitting_primitive_to_voronoi_region(const iVd &vor, point_cloud &pc,
                                    const vector<int> &voronoi_centers,
                                    const int center);

std::tuple<vector<vec3d>, vector<vec3d>>
patch_fitting_w_pricipal_dir(point_cloud &pc);

vector<pair<vec2d, int>> GrahamScan(const vector<pair<vec2d, int>> &p_nbr);

double gaussian_curvature(const point_cloud &pc, const int &vid);

vector<double> curvature_field(const point_cloud &pc);

Eigen::Matrix2d shape_operator(const patch &p, const vec2d &uv);

vector<vec3d> PCA(const vector<int> &nbr, const vector<vec3d> &positions);

bool my_matrix_eigenfunctions(const Eigen::SparseMatrix<double> &m,
                              const int nf, std::vector<double> &f,
                              std::vector<double> &f_min,
                              std::vector<double> &f_max);

iVd intrinsic_voronoi_diagram(const geodesic_solver &solver,
                              const DrawableTrimesh<> &mesh,
                              const vector<int> seeds);

Eigen::MatrixXd compute_biharmonic_distance(const DrawableTrimesh<> &m);

pair<iVd, vector<int>> intrinsic_voronoi_diagram_blended(
    const geodesic_solver &solver, const DrawableTrimesh<> &mesh,
    const point_cloud &pc, const float &alpha, const vector<int> &seeds);

vec3d pos_on_parabolic_cylinder(const Eigen::VectorXd &Q, const vec2d &uv,
                                const Eigen::Vector2d &u0,
                                const vector<vec3d> &e);

vec3d pos_on_parabolic_cylinder(const Eigen::VectorXd &Q, const vec2d &uv,
                                const vector<vec3d> &e);
patch direct_parabolic_cylinder_fitting(point_cloud &pc,
                                        const vector<int> &nbr);

void refine_voronoi_diagram(iVd &vor, vector<int> &vor_centers,
                            const geodesic_solver &solver,
                            const DrawableTrimesh<> &mesh,
                            const vector<double> &phi, const Eigen::MatrixXd &D,
                            const point_cloud &pc, const int type, const int k);

void refine_voronoi_diagram_blended(iVd &vor, vector<int> &vor_centers,
                                    const geodesic_solver &solver,
                                    const DrawableTrimesh<> &mesh,
                                    const vector<double> &phi,
                                    const Eigen::MatrixXd &D,
                                    const point_cloud &pc, const double &alpha,
                                    const int k);
// sort patches in ascending order w.r.t residual
std::tuple<vector<int>, double, double> ordered_patches(const point_cloud &pc);

// compute mean residual
double compute_tau(const point_cloud &pc);

// update the nbr of points by removing outliers
void remove_outliers(point_cloud &pc, const double &max_err);

bool remove_outliers(point_cloud &pc, const int vid, const double &max_err);

std::tuple<Eigen::Matrix2d, vector<vec3d>> metric_tensor(const patch &p,
                                                         const vec2d &uv);

std::tuple<vector<int>, vector<int>, double> tag_points(point_cloud &pc);

void update_tagging(point_cloud &pc, const int vid, const vector<int> &centers);

vec3d evaluate_quadric_MP(const vec3d &p, const vector<vec3d> &e,
                          const Eigen::VectorXd &Q, const vec2d &uv);

vec3d evaluate_primitive(const Eigen::MatrixXd &Q, const vec2d &uv,
                         const int type);

vec3d evaluate_quadric(const Eigen::MatrixXd &Q, const vec2d &uv);

vec3d evaluate_quadric_du(const Eigen::MatrixXd &Q, const vec2d &uv);

vec3d evaluate_quadric_dv(const Eigen::MatrixXd &Q, const vec2d &uv);

vec3d evaluate_quadric_duu(const Eigen::MatrixXd &Q, const vec2d &uv);

vec3d evaluate_quadric_dvv(const Eigen::MatrixXd &Q, const vec2d &uv);

vec3d evaluate_quadric_duv(const Eigen::MatrixXd &Q, const vec2d &uv);

patch quadric_fitting(point_cloud &pc, const vector<int> &nbr);

std::tuple<vector<vec3d>, vector<vec3d>>
principal_curvature_field(const point_cloud &pc);

void enlarge_patch(point_cloud &pc, const int vid, const double p = 1);

void expand_centers(point_cloud &pc, const double &max_err);

void expand_patch(point_cloud &pc, const int vid, const double &threshold);

bool check_secondary_patch(const point_cloud &pc, const int vid);

void trim_secondary_patches(point_cloud &pc, const double &max_err);

void kill_small_patches(point_cloud &pc, const double &max_err);

vector<int> patch_to_patch_adjacency(const point_cloud &pc, const int vid);

vector<double> subdivide_angles(const int number_of_subdivision);

vector<vec3d> local_grad(const point_cloud &pc, const int vid,
                         const vector<double> &f, const double &r);

vector<vec3d> local_principal_dir(const point_cloud &pc, const int vid,
                                  const double &r);

vector<double> cotangent_laplacian(const Eigen::SparseMatrix<double> &L,
                                   const Eigen::SparseMatrix<double> &M,
                                   const vector<double> &field);

vector<vec2i> singularities(const point_cloud &pc, const vector<double> &f);

vector<vec2i> singularities(const DrawableTrimesh<> &m,
                            const vector<double> &f);

vector<int> singular_vertices(const point_cloud &pc, const vector<double> &f);

vector<int> singular_vertices(const DrawableTrimesh<> &m,
                              const vector<double> &f);

vector<int> singular_vertices(const point_cloud &pc, const vector<double> &f,
                              const vector<bool> &bad);

vector<vec2i> singularities(const point_cloud &pc);

inline pair<Eigen::MatrixXd, Eigen::MatrixXi>
libigl_wrapper(const vector<vec3d> &positions,
               const vector<vector<uint>> &triangles) {
  Eigen::MatrixXd V(positions.size(), 3);
  Eigen::MatrixXi F(triangles.size(), 3);

  for (int i = 0; i < positions.size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      V(i, j) = positions[i][j];
    }
  }
  for (int i = 0; i < triangles.size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      F(i, j) = triangles[i][j];
    }
  }

  return {V, F};
}

#endif
