#include "diff_geo.h"
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <algorithm>
#include <cinolib/cino_inline.h>
#include <cinolib/geometry/triangle_utils.h>
#include <cinolib/vertex_mass.h>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>
using namespace cinolib;
using namespace Eigen;
using namespace std;
using namespace std::complex_literals;

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//                                  INLINE FUNCTIONS

inline double cross(const vec2d v, const vec2d w) {
  return v.x() * w.y() - w.x() * v.y();
}
vec3d project_vec(const vec3d &v, const vec3d &n) {
  auto proj = n * v.dot(n);

  return v - proj;
}
vec3d rot_vec(const vec3d &v, const vec3d &axis, const float &angle) {
  mat3d R = mat3d::ROT_3D(axis, angle);
  return R * v;
}
VectorXd wrapper(const vector<double> &f) {
  VectorXd F(f.size());
  for (int i = 0; i < f.size(); ++i) {
    F(i) = f[i];
  }
  return F;
}
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//                                 Cleaning
std::tuple<vector<bool>, int> duplicated_vertices(const DrawableTrimesh<> &m) {
  auto result = vector<bool>(m.num_verts(), false);
  auto goodones = m.num_verts();
  for (auto i = 0; i < m.num_verts(); ++i) {
    auto star = m.adj_v2p(i);
    for (auto tid : star) {
      if (m.poly_area(tid) == 0) {
        result[i] = true;
        --goodones;
        break;
      }
    }
  }

  return {result, goodones};
}

vector<int> cleaned_k_ring(const DrawableTrimesh<> &m, const int vid,
                           const int n, const vector<bool> &bad) {
  std::vector<int> active_set;
  std::vector<int> ring = {vid};

  active_set.push_back(vid);
  while (ring.size() < n) {
    std::vector<int> next_active_set;

    for (int curr : active_set)
      for (int tid : m.adj_v2p(curr)) {
        if (m.poly_area(tid) == 0)
          continue;
        auto eid = m.edge_opposite_to(tid, curr);
        auto nei = m.edge_vert_ids(eid);
        for (auto nbr : nei) {
          if (std::find(ring.begin(), ring.end(), nbr) == ring.end()) {
            next_active_set.push_back(nbr);
            ring.push_back(nbr);
          }
        }
      }

    active_set = next_active_set;
  }
  return ring;
}
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//                                 ANN
ANNpoint ann_wrapper(const vec3d &p) {
  ANNpoint point = annAllocPt(3);
  for (auto j = 0; j < 3; ++j) {
    ANNcoord c = p[j];
    point[j] = c;
  }
  return point;
}
ANNpointArray ann_wrapper(const vector<vec3d> &positions) {
  auto pts = annAllocPts(positions.size(), 3);
  for (auto i = 0; i < positions.size(); ++i) {
    auto p = ann_wrapper(positions[i]);
    pts[i] = p;
  }

  return pts;
}

vector<int> knn(const vector<vec3d> &positions, const vec3d &pos, const int k) {

  auto result = vector<int>(k + 1);
  auto d = vector<pair<double, int>>(positions.size());
  for (auto i = 0; i < positions.size(); ++i) {
    d[i] = std::make_pair((pos - positions[i]).norm(), i);
  }
  sort(d.begin(), d.end());
  for (auto i = 0; i <= k; ++i)
    result[i] = d[i].second;

  return result;
}
vector<int> knn(ANNkd_tree *tree, const vector<vec3d> &positions, const int vid,
                const int k) {
  ANNidxArray ids = new ANNidx[k];
  ANNdistArray d = new ANNdist[k];
  double eps = 0;
  auto point = ann_wrapper(positions[vid]);

  // point = tree->thePoints()[vid];
  tree->annkSearch(point, k, ids, d, eps);

  vector<int> nbr(k);
  for (auto i = 0; i < k; ++i)
    nbr[i] = (int)ids[i];

  return nbr;
}
vector<int> n_ring(const DrawableTrimesh<> &m, const int vid, const uint n) {
  std::vector<int> active_set;
  std::vector<int> ring = {vid};

  active_set.push_back(vid);
  for (int i = 0; i < n; ++i) {
    std::vector<int> next_active_set;

    for (int curr : active_set)
      for (int nbr : m.adj_v2v(curr)) {
        if (find(ring.begin(), ring.end(), nbr) == ring.end()) {
          next_active_set.push_back(nbr);
          ring.push_back(nbr);
        }
      }

    active_set = next_active_set;
  }
  return ring;
}
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//                                  PC I/O

// vector<int> fps(const point_cloud &pc, const int k) {
//   auto argmax = [](const vector<double> &d) {
//     auto max_dist = __DBL_MIN__;
//     auto result = -1;
//     for (auto i = 0; i < d.size(); ++i)
//       if (d[i] > max_dist) {
//         max_dist = d[i];
//         result = i;
//       }

//     return result;
//   };
//   auto N = pc.positions.size();
//   auto result = vector<int>(k);
//   vector<double> dist(N, __DBL_MAX__);

//   auto curr_id = 0;
//   for (auto i = 0; i < k; ++i) {
//     auto pos = pc.positions[curr_id];
//     for (auto j = 0; j < N; ++j) {
//       auto d = (pc.positions[j] - pos).norm();
//       if (dist[j] < d)
//         dist[j] = d;
//     }
//     result.push_back(curr_id);
//     curr_id = argmax(dist);
//   }
// }
vec3d centroid(const vector<vec3d> &p) {
  auto k = p.size();
  auto c = vec3d{0, 0, 0};
  for (auto i = 0; i < k; ++i)
    c += p[i];

  c /= k;
  return c;
}
vec2d centroid(const vector<vec2d> &p) {
  auto k = p.size();
  auto c = vec2d{0, 0};
  for (auto i = 0; i < k; ++i)
    c += p[i];

  c /= k;
  return c;
}
vec3d centroid(const vector<int> &nbr, const vector<vec3d> &positions) {
  auto pos = vector<vec3d>(nbr.size());
  for (auto i = 0; i < nbr.size(); ++i)
    pos[i] = positions[nbr[i]];
  return centroid(pos);
}

vector<vec3d> PCA(const vector<int> &nbr, const vector<vec3d> &positions) {

  auto k = (int)nbr.size();
  vector<float> lens(k);
  float sigma = 0.f;

  // for (auto i = 0; i < nbr.size(); ++i) {
  //   auto len = (positions[nbr[i]] - pos).norm_sqrd();
  //   lens[i] = len;
  //   sigma += len;
  // }
  auto c = centroid(nbr, positions);
  MatrixXd P(k, 3);
  for (auto i = 0; i < k; ++i) {
    Vector3d curr_pos;

    curr_pos << positions[nbr[i]].x() - c.x(), positions[nbr[i]].y() - c.y(),
        positions[nbr[i]].z() - c.z();
    P.row(i) = curr_pos;
  }
  // sigma /= k;
  // auto w = 0.j;
  // MatrixXd W(k, k);
  // W.setZero();
  // auto sum = 0.f;
  // for (auto j = 0; j < k; ++j) {
  //   auto wgt = exp(-lens[j] / pow(sigma, 2));
  //   W.diagonal()[j] = wgt;
  //   sum += wgt;
  // }

  Matrix3d M = 1.f / k * P.transpose() * P;
  SelfAdjointEigenSolver<Matrix3d> dec(M);
  Matrix3d E = dec.eigenvectors();
  Vector3d e0 = E.col(2);
  Vector3d e1 = E.col(1);
  Vector3d n = E.col(0);

  auto result = vector<vec3d>(4);
  result[0] = c;
  result[1] = vec3d{e0(0), e0(1), e0(2)};
  result[2] = vec3d{e1(0), e1(1), e1(2)};
  result[1].normalize();
  result[2].normalize();
  result[3] = result[1].cross(result[2]);

  return result;
}
std::tuple<vector<vec3d>, vector<int>> PCA(ANNkd_tree *tree,
                                           const vector<vec3d> &positions,
                                           const int vid, const int k = 15) {

  auto nbr = knn(tree, positions, vid, k + 1);

  auto e = PCA(nbr, positions);

  return {e, nbr};
}
void fit_into_cube(vector<vec3d> &positions) {
  AABB box = AABB(positions);
  auto center = box.center();
  auto scale_factor = 1.0 / (box.max - box.min).max_entry();
  for (auto &p : positions) {
    p = (p - center) * scale_factor;
  }
}
Matrix2d first_fund(const MatrixXd &Q, const vec2d &uv) {
  Matrix2d I;
  auto xu = evaluate_quadric_du(Q, uv);
  auto xv = evaluate_quadric_dv(Q, uv);
  I << xu.dot(xu), xu.dot(xv), xu.dot(xv), xv.dot(xv);

  return I;
}
Matrix2d second_fund(const MatrixXd &Q, const vec2d &uv) {
  Matrix2d II;
  auto xu = evaluate_quadric_du(Q, uv);
  auto xv = evaluate_quadric_dv(Q, uv);
  auto xuu = evaluate_quadric_duu(Q, uv);
  auto xuv = evaluate_quadric_duv(Q, uv);
  auto xvv = evaluate_quadric_dvv(Q, uv);
  auto N = xu.cross(xv);
  N.normalize();
  II << N.dot(xuu), N.dot(xuv), N.dot(xuv), N.dot(xvv);

  return II;
}
Matrix2d shape_operator(const patch &p, const vec2d &uv) {
  auto Q = p.quadric;
  auto I = first_fund(Q, uv);
  auto II = second_fund(Q, uv);
  Matrix2d A = -I.inverse() * II;

  return A;
}
Matrix2d third_fund(const patch &p, const vec2d &uv) {
  auto Q = p.quadric;
  auto I = first_fund(Q, uv);
  auto II = second_fund(Q, uv);
  Matrix2d S = -II * I.inverse();
  SelfAdjointEigenSolver<Matrix2d> dec(S);
  auto k1 = dec.eigenvalues()(0);
  auto k2 = dec.eigenvalues()(1);
  auto H = (k1 + k2) / 2;
  auto K = k1 * k2;
  Matrix2d III = 2 * H * II - K * I;

  return III;
}
// T is the canonical-to-frame matrix
Matrix4d transformation_matrix(const vec3d &x1, const vec3d &y1,
                               const vec3d &z1, const vec3d &O1) {
  Matrix4d T;

  T << x1.x(), x1.y(), x1.z(), -x1.dot(O1), y1.x(), y1.y(), y1.z(), -y1.dot(O1),
      z1.x(), z1.y(), z1.z(), -z1.dot(O1), 0, 0, 0, 1;

  return T;
}

vec3d evaluate_primitive(const MatrixXd &Q, const vec2d &uv, const int type) {

  auto pos = vec3d{0, 0, 0};
  switch (type) {
  case bilinear_patch: {
    for (auto j = 0; j < 3; ++j) {
      pos[j] = Q(0, j) + uv.x() * Q(1, j) + uv.y() * Q(2, j) +
               Q(3, j) * uv.x() * uv.y();
    }
  } break;
  case parabolic_cylinder: {
    pos = vec3d{Q(0, 0) + Q(1, 0) * uv.x(), Q(2, 0) + Q(3, 0) * uv.y(),
                Q(4, 0) + Q(5, 0) * pow(uv.x(), 2)};

  } break;

  case elliptic_paraboloid: {

    pos = vec3d{Q(0, 0) + Q(1, 0) * uv.x() * cos(uv.y()),
                Q(2, 0) + Q(3, 0) * uv.x() * sin(uv.y()),
                Q(4, 0) + Q(5, 0) * pow(uv.x(), 2)};
  } break;

  case hyperbolic_paraboloid: {
    pos = vec3d{Q(0, 0) + Q(1, 0) * uv.x() + Q(2, 0) * uv.y(),
                Q(3, 0) + Q(4, 0) * uv.y(),
                Q(5, 0) + Q(6, 0) * uv.x() + uv.y() + Q(7, 0) * pow(uv.x(), 2)};
  } break;
  }
  return pos;
}
vec3d evaluate_quadric_MP(const vec3d &p, const vector<vec3d> &e,
                          const VectorXd &Q, const vec2d &uv) {

  auto T = transformation_matrix(e[0], e[1], e[2], p);
  Vector4d pos;
  pos << uv.x(), uv.y(),
      Q(0) + Q(1) * uv.x() + Q(2) * uv.y() + 0.5 * Q(3) * pow(uv.x(), 2) +
          Q(4) * uv.x() * uv.y() + 0.5 * pow(uv.y(), 2) * Q(5),
      1;
  Vector4d v = T.inverse() * pos;

  return vec3d{v(0), v(1), v(2)};
}
double evaluate_parabolic_cylinder(const Eigen::VectorXd &Q, const vec2d &uv,
                                   const Vector2d &u0) {

  return Q(0) + Q(1) * uv.x() + Q(2) * uv.y() +
         Q(3) *
             (pow(u0(0), 2) * pow(uv.x(), 2) + pow(u0(1), 2) * pow(uv.y(), 2) +
              2 * u0(0) * u0(1) * uv.x() * uv.y());
}
double evaluate_parabolic_cylinder(const Eigen::VectorXd &Q, const vec2d &uv) {

  return Q(0) + Q(1) * uv.y() + Q(2) * pow(uv.y(), 2);
}
vec3d pos_on_parabolic_cylinder(const Eigen::VectorXd &Q, const vec2d &uv,
                                const Vector2d &u0, const vector<vec3d> &e) {
  auto T = transformation_matrix(e[1], e[2], e[3], e[0]);
  Vector4d pos;
  pos << uv.x(), uv.y(), evaluate_parabolic_cylinder(Q, uv, u0), 1;
  Vector4d v = T.inverse() * pos;

  return vec3d{v(0), v(1), v(2)};
}
vec3d pos_on_parabolic_cylinder(const Eigen::VectorXd &Q, const vec2d &uv,
                                const vector<vec3d> &e) {
  auto T = transformation_matrix(e[1], e[2], e[3], e[0]);
  Vector4d pos;
  pos << uv.x(), uv.y(), evaluate_parabolic_cylinder(Q, uv), 1;
  Vector4d v = T.inverse() * pos;

  return vec3d{v(0), v(1), v(2)};
}
vec3d pos_on_parabolic_cylinder(const Eigen::VectorXd &Q, const vec2d &uv,
                                const vector<vec3d> &e,
                                const Eigen::Vector2d &u0) {
  auto T = transformation_matrix(e[1], e[2], e[3], e[0]);
  Vector4d pos;
  pos << uv.x(), uv.y(), evaluate_parabolic_cylinder(Q, uv, u0), 1;
  Vector4d v = T.inverse() * pos;

  return vec3d{v(0), v(1), v(2)};
}
vec3d evaluate_quadric(const MatrixXd &Q, const vec2d &uv) {

  auto pos = vec3d{0, 0, 0};
  for (auto j = 0; j < 3; ++j) {
    pos[j] = Q(0, j) + uv.x() * Q(1, j) + uv.y() * Q(2, j) +
             1.f / 2 * Q(3, j) * std::pow(uv.x(), 2) +
             Q(4, j) * uv.x() * uv.y() +
             1.f / 2 * Q(5, j) * std::pow(uv.y(), 2);
  }

  return pos;
}

vec3d evaluate_quadric_du(const MatrixXd &Q, const vec2d &uv) {

  auto pos = vec3d{0, 0, 0};
  for (auto j = 0; j < 3; ++j) {
    pos[j] = Q(1, j) + Q(3, j) * uv.x() + Q(4, j) * uv.y();
  }

  return pos;
}
vec3d evaluate_quadric_dv(const MatrixXd &Q, const vec2d &uv) {

  auto pos = vec3d{0, 0, 0};
  for (auto j = 0; j < 3; ++j) {
    pos[j] = Q(2, j) + Q(5, j) * uv.y() + Q(4, j) * uv.x();
  }

  return pos;
}
vec3d evaluate_quadric_duu(const MatrixXd &Q, const vec2d &uv) {

  auto pos = vec3d{0, 0, 0};
  for (auto j = 0; j < 3; ++j) {
    pos[j] = Q(3, j);
  }

  return pos;
}
vec3d evaluate_quadric_dvv(const MatrixXd &Q, const vec2d &uv) {

  auto pos = vec3d{0, 0, 0};
  for (auto j = 0; j < 3; ++j) {
    pos[j] = Q(5, j);
  }

  return pos;
}
vec3d evaluate_quadric_duv(const MatrixXd &Q, const vec2d &uv) {

  auto pos = vec3d{0, 0, 0};
  for (auto j = 0; j < 3; ++j) {
    pos[j] = Q(4, j);
  }

  return pos;
}
double evaluate_quadric(const VectorXd &Q, const vec2d &uv) {

  return Q(0) + uv.x() * Q(1) + uv.y() * Q(2) +
         1.f / 2 * Q(3) * std::pow(uv.x(), 2) + Q(4) * uv.x() * uv.y() +
         1.f / 2 * Q(5) * std::pow(uv.y(), 2);
}
double evaluate_quadric_du(const VectorXd &Q, const vec2d &uv) {

  return Q(1) + Q(3) * uv.x() + Q(4) * uv.y();
}
double evaluate_quadric_dv(const VectorXd &Q, const vec2d &uv) {

  return Q(2) + Q(5) * uv.y() + Q(4) * uv.x();
}
double evaluate_quadric_duu(const VectorXd &Q, const vec2d &uv) { return Q(3); }
double evaluate_quadric_dvv(const VectorXd &Q, const vec2d &uv) { return Q(5); }
double evaluate_quadric_duv(const VectorXd &Q, const vec2d &uv) { return Q(4); }
std::tuple<Matrix2d, vector<vec3d>>
isophotic_metric(const vec3d &xu, const vec3d &xv, const vec3d &xuu,
                 const vec3d &xuv, const vec3d &xvv, const double &w = 0,
                 const double &w_star = 1) {
  Matrix2d I;
  Matrix2d II;
  auto n = xu.cross(xv);
  n.normalize();
  auto E = xu.dot(xu);
  auto F = xu.dot(xv);
  auto G = xv.dot(xv);
  auto L = n.dot(xuu);
  auto M = n.dot(xuv);
  auto N = n.dot(xvv);
  I << E, F, F, G;
  II << L, M, M, N;

  auto K = (L * N - pow(M, 2)) / (E * G - pow(F, 2));
  auto H = (E * N - 2 * F * M + G * L) / (E * G - pow(F, 2)); // multplied by 2

  Matrix2d III = H * II - K * I;

  return {w * I + w_star * III, {xu, xv, xuu, xvv, xuv}};
}
std::tuple<Matrix2d, vector<vec3d>> isophotic_metric(const patch &p,
                                                     const vec2d &uv,
                                                     const double &w = 1,
                                                     const double &w_star = 0) {
  auto Q = p.quadric;
  Matrix2d I;
  Matrix2d II;

  auto xu = evaluate_quadric_du(Q, uv);
  auto xv = evaluate_quadric_dv(Q, uv);
  auto xuu = evaluate_quadric_duu(Q, uv);
  auto xuv = evaluate_quadric_duv(Q, uv);
  auto xvv = evaluate_quadric_dvv(Q, uv);
  return isophotic_metric(xu, xv, xuu, xuv, xvv, w, w_star);
}
std::tuple<Matrix2d, vector<vec3d>> metric_tensor(const patch &p,
                                                  const vec2d &uv) {
  auto Q = p.quadric;

  auto xu = evaluate_quadric_du(Q, uv);
  auto xv = evaluate_quadric_dv(Q, uv);
  auto xuu = evaluate_quadric_duu(Q, uv);
  auto xuv = evaluate_quadric_duv(Q, uv);
  auto xvv = evaluate_quadric_dvv(Q, uv);
  Matrix2d I;
  I << xu.dot(xu), xu.dot(xv), xu.dot(xv), xv.dot(xv);
  return {I, {xu, xv, xuu, xuv, xvv}};
}
geodesic_solver compute_geodesic_solver(point_cloud &pc) {
  geodesic_solver solver;
  auto V = pc.positions.size();
  solver.graph.resize(V);
  for (auto i = 0; i < V; ++i) {
    auto &p = pc.patches[i];
    auto pos = pc.positions[i];
    auto nbr = p.parametric_nbr;
    auto k = nbr.size();
    solver.graph[i].resize(k);
    auto vert = pc.positions[i];
    auto I = first_fund(p.quadric, vec2d{0, 0});
    auto [S, X] = isophotic_metric(p, vec2d{0, 0}, 1, 2);
    for (auto j = 0; j < k; ++j) {
      solver.graph[i][j].node = nbr[j].second;
      // solver.graph[i][j].length = (pc.positions[nbr[j]] - vert).norm();
      // auto it = find_if(p.parametric_nbr.begin(), p.parametric_nbr.end(),
      //                   [](const auto point) { return point.second; });
      // assert(it != p.parametric_nbr.end());
      auto v = nbr[j].first;
      Vector2d vd;
      vd << v.x(), v.y();
      double geo_len = (double)(vd.transpose() * I * vd);
      double iso_len = (double)(vd.transpose() * S * vd);
      solver.graph[i][j].length = vd.transpose() * I * vd;
      solver.graph[i][j].isophotic_length = vd.transpose() * S * vd;
    }
  }

  return solver;
}
void connect_nodes(geodesic_solver &solver, int a, int b, float length) {
  solver.graph[a].push_back({b, length});
  solver.graph[b].push_back({a, length});
}
double opposite_nodes_arc_length(const vector<vec3d> &positions, int a, int c,
                                 const vec2i &edge) {
  // Triangles (a, b, d) and (b, d, c) are connected by (b, d) edge
  // Nodes a and c must be connected.

  auto b = edge.x(), d = edge.y();
  auto ba = positions[a] - positions[b];
  auto bc = positions[c] - positions[b];
  auto bd = positions[d] - positions[b];
  ba.normalize();
  bd.normalize();
  auto cos_alpha = ba.dot(bd);
  auto cos_beta = bc.dot(bd);
  auto sin_alpha = sqrt(max(0.0, 1 - cos_alpha * cos_alpha));
  auto sin_beta = sqrt(max(0.0, 1 - cos_beta * cos_beta));

  // cos(alpha + beta)
  auto cos_alpha_beta = cos_alpha * cos_beta - sin_alpha * sin_beta;
  if (cos_alpha_beta <= -1)
    return DBL_MAX;

  // law of cosines (generalized Pythagorean theorem)
  ba = positions[a] - positions[b];
  bc = positions[c] - positions[b];
  bd = positions[d] - positions[b];
  auto len =
      ba.dot(ba) + bc.dot(bc) - ba.norm() * bc.norm() * 2 * cos_alpha_beta;

  if (len <= 0)
    return DBL_MAX;
  else
    return sqrt(len);
}

static void connect_opposite_nodes(geodesic_solver &solver,
                                   const vector<vec3d> &positions,
                                   const vector<uint> &tr0,
                                   const vector<uint> &tr1, const vec2i &edge) {
  auto opposite_vertex = [](const vector<uint> &tr, const vec2i &edge) -> int {
    for (auto i = 0; i < 3; ++i) {
      if (tr[i] != edge.x() && tr[i] != edge.y())
        return tr[i];
    }
    return -1;
  };

  auto v0 = opposite_vertex(tr0, edge);
  auto v1 = opposite_vertex(tr1, edge);
  if (v0 == -1 || v1 == -1)
    return;
  auto length = opposite_nodes_arc_length(positions, v0, v1, edge);
  connect_nodes(solver, v0, v1, length);
}

geodesic_solver make_geodesic_solver(const DrawableTrimesh<> &m) {
  auto solver = geodesic_solver{};
  solver.graph.resize(m.num_verts());
  for (auto face = 0; face < m.num_polys(); face++) {
    for (auto k = 0; k < 3; k++) {
      auto a = m.poly_vert_id(face, k);
      auto b = m.poly_vert_id(face, (k + 1) % 3);

      // connect mesh edges
      auto len = (m.vert(a) - m.vert(b)).norm();
      if (a < b)
        connect_nodes(solver, a, b, len);

      // connect opposite nodes
      auto neighbor = m.adj_p2p(face)[k];
      if (face < neighbor) {
        connect_opposite_nodes(solver, m.vector_verts(), m.adj_p2v(face),
                               m.adj_p2v(neighbor), vec2i{(int)a, (int)b});
      }
    }
  }
  return solver;
}

geodesic_solver compute_geodesic_solver(point_cloud &pc,
                                        const vector<bool> &badones) {
  geodesic_solver solver;
  auto V = pc.positions.size();
  solver.graph.resize(V);
  for (auto i = 0; i < V; ++i) {
    if (badones[i])
      continue;
    auto &p = pc.patches[i];
    auto pos = pc.positions[i];
    auto nbr = p.parametric_nbr;
    auto k = nbr.size();
    solver.graph[i].resize(k);
    auto I = first_fund(p.quadric, vec2d{0, 0});
    auto [S, X] = isophotic_metric(p, vec2d{0, 0}, 1, 2);
    for (auto j = 0; j < k; ++j) {
      solver.graph[i][j].node = nbr[j].second;
      auto v = nbr[j].first;
      Vector2d vd;
      vd << v.x(), v.y();
      double geo_len = (double)(vd.transpose() * I * vd);
      double iso_len = (double)(vd.transpose() * S * vd);
      solver.graph[i][j].length = vd.transpose() * I * vd;
      solver.graph[i][j].isophotic_length = vd.transpose() * S * vd;
      //(pc.positions[nbr[j].second] - pos).norm();
    }
  }

  return solver;
}

point_cloud init_pc(vector<vec3d> &positions) {
  // Profiler prof;
  // prof.push("INIT Point Cloud");
  point_cloud result;
  auto N = (int)positions.size();
  auto spheres = vector<DrawableSphere>(N);
  auto basis = vector<vector<vec3d>>(N);
  auto nbrs = vector<vector<int>>(N);
  auto pts = ann_wrapper(positions);
  result.tree = new ANNkd_tree(pts, (int)positions.size(), 3);

  for (auto i = 0; i < N; ++i) {
    spheres[i] = DrawableSphere(positions[i], 0.002, cinolib::Color::GRAY());
  }
  result.positions = positions;
  result.points = spheres;
  result.basis = basis;
  result.nbrs = nbrs;
  // prof.pop();
  //  result.seeds = fps(result, 2000);

  return result;
}
void rehorient_pc(point_cloud &pc, const vector<vec3d> &normals) {
  for (auto i = 0; i < pc.basis.size(); ++i) {
    auto &e = pc.basis[i];
    if (e[3].dot(normals[i]) < 0)
      e[3] *= -1;

    if (e[1].cross(e[2]).dot(e[3]) < 0)
      e[2] *= -1;
  }
}
void rehorient_pc(point_cloud &pc) {
  std::deque<int> q;
  q.push_front(0);
  auto parsed = vector<bool>(pc.positions.size(), false);
  parsed[0] = true;
  auto &adjust = pc.basis[0];
  // adjust[3] *= -1;
  if (adjust[1].cross(adjust[2]).dot(adjust[3]) < 0)
    adjust[2] *= -1;

  while (!q.empty()) {
    auto curr = q.front();
    q.pop_front();
    auto e = pc.basis[curr];
    auto &nbr = pc.nbrs[curr];
    for (auto j = 0; j < 10; ++j) {
      if (parsed[nbr[j]])
        continue;

      auto &e_curr = pc.basis[nbr[j]];
      if (e_curr[3].dot(e[3]) < 0)
        e_curr[3] *= -1;

      if (e_curr[1].cross(e_curr[2]).dot(e_curr[3]) < 0)
        e_curr[2] *= -1;

      parsed[nbr[j]] = true;
      q.push_back(nbr[j]);
    }
  }
}

std::tuple<vector<vec3d>, vector<vec3d>, vector<vec3d>>
get_basis(const point_cloud &pc) {
  auto N = pc.positions.size();
  auto e0 = vector<vec3d>(N);
  auto e1 = vector<vec3d>(N);
  auto n = vector<vec3d>(N);
  for (auto i = 0; i < N; ++i) {
    e0[i] = pc.basis[i][1];
    e1[i] = pc.basis[i][2];
    n[i] = pc.basis[i][3];
  }

  return {e0, e1, n};
}
vector<vec3d> get_normals(const point_cloud &pc) {
  auto N = pc.positions.size();
  auto n = vector<vec3d>(N);
  for (auto i = 0; i < N; ++i) {
    n[i] = pc.basis[i][3];
  }

  return n;
}

vec3d normal_estimation(const vector<vec3d> &positions, const int vid,
                        const int k = 6) {
  vector<int> nbr = knn(positions, positions[vid], k);
  vector<float> lens(k);
  float sigma = 0.f;
  vec3d pos = positions[vid];

  for (auto i = 0; i < nbr.size(); ++i) {
    auto len = (positions[nbr[i]] - pos).norm_sqrd();
    lens[i] = len;
    sigma += len;
  }
  auto c = centroid(nbr, positions);
  MatrixXd P(k, 3);
  for (auto i = 0; i < k; ++i) {
    Vector3d curr_pos;

    curr_pos << positions[nbr[i]].x() - c.x(), positions[nbr[i]].y() - c.y(),
        positions[nbr[i]].z() - c.z();
    P.row(i) = curr_pos;
  }
  sigma /= k;
  auto w = 0.j;
  MatrixXd W(k, k);
  W.setZero();
  auto sum = 0.f;
  for (auto j = 0; j < k; ++j) {
    auto wgt = exp(-lens[j] / pow(sigma, 2));
    W.diagonal()[j] = wgt;
    sum += wgt;
  }

  Matrix3d M = 1.f / k * P.transpose() * P;
  SelfAdjointEigenSolver<Matrix3d> dec(M);
  Vector3d n = dec.eigenvectors().col(0);
  auto result = vec3d{n(0), n(1), n(2)};
  result.normalize();
  return result;
}
std::tuple<vector<vec2d>, vector<int>>
project_points(const int vid, const point_cloud &pc, const vector<int> &nbr) {
  auto basis = pc.basis[vid];
  auto u = basis[1];
  auto n = basis[3];
  auto parametric_coords = vector<vec2d>(nbr.size());
  auto added_points = vector<int>(nbr.size());
  auto pos = pc.positions[vid];
  for (auto i = 0; i < nbr.size(); ++i) {
    if (n.angle_rad(pc.basis[nbr[i]][3]) > M_PI_2)
      continue;
    auto v = pc.positions[nbr[i]] - pos;
    auto len = v.norm();
    if (len != 0) {
      auto p = project_vec(v, n);
      auto theta = u.angle_rad(p);

      if ((u.cross(p).dot(n)) < 0)
        theta = 2 * M_PI - theta;
      parametric_coords[i] =
          vec2d{len * std::cos(theta), len * std::sin(theta)};
      added_points[i] = nbr[i];
    } else {
      parametric_coords[i] = vec2d{0, 0};
      added_points[i] = nbr[i];
    }
  }

  return {parametric_coords, added_points};
}
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//                                 Patch fitting

double max_residual(const MatrixXd &r) {
  auto max_err = __DBL_MIN__;
  for (auto i = 0; i < r.rows(); ++i) {
    max_err = std::max(r.row(i).norm(), max_err);
  }
  return max_err;
}

double max_residual(const VectorXd &r) {
  auto max_err = __DBL_MIN__;
  for (auto i = 0; i < r.rows(); ++i) {
    max_err = std::max(r(i), max_err);
  }
  return max_err;
}
std::tuple<VectorXd, VectorXd, VectorXd, VectorXd>
cgstep_1d(const VectorXd &g, const VectorXd &G, const VectorXd &r,
          const VectorXd &x0, const VectorXd &s, const VectorXd &S,
          const int it) {

  if (it == 0) {
    auto GdG = G.dot(G);
    if (GdG < 1e-10)
      return {x0, r, s, S};
    auto GdR = G.dot(r);
    auto alpha = GdR / GdG;
    VectorXd new_s = alpha * g;
    VectorXd new_S = alpha * G;
    VectorXd new_sol = x0 + new_s;
    VectorXd new_r = r - new_S;
    return {new_sol, new_r, new_s, new_S};
  }
  auto GdG = G.dot(G);
  auto GdS = S.dot(G);
  auto SdS = S.dot(S);
  auto GdR = G.dot(r);
  auto SdR = S.dot(r);
  auto det = GdG * SdS - pow(GdS, 2);
  if (det < 1e-12)
    return {x0, r, s, S};
  auto alpha = (SdS * GdR - GdS * SdR) / det;
  auto beta = (-GdS * GdR + GdG * SdR) / det;
  VectorXd new_s = alpha * g + beta * s;
  VectorXd new_S = alpha * G + beta * S;
  VectorXd new_sol = x0 + new_s;
  VectorXd new_r = r - new_S;
  return {new_sol, new_r, new_s, new_S};
}
std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd>
cgstep(const MatrixXd &g, const MatrixXd &G, const MatrixXd &r,
       const MatrixXd &x0, const MatrixXd &s, const MatrixXd &S, const int it) {
  auto k = r.rows();
  MatrixXd new_sol(6, 3);
  MatrixXd new_res(k, 3);
  MatrixXd new_s(6, 3);
  MatrixXd new_S(k, 3);
  auto GRt = (G * r.transpose()).trace();
  auto GSt = (G * S.transpose()).trace();
  auto RSt = (r * S.transpose()).trace();
  auto GGt = (G * G.transpose()).trace();
  auto SSt = (S * S.transpose()).trace();
  if (it == 0) {
    if (GGt < 1e-12)
      return {x0, r, s, S};
    auto alpha = GRt / GGt;
    new_s = alpha * g;
    new_S = alpha * G;
    new_sol = x0 + new_s;
    new_res = r - new_S;
    return {new_sol, new_res, new_s, new_S};
  }
  if (GGt < 1e-8)
    return {x0, r, s, S};
  auto beta = 0.0;
  if ((SSt * GGt - pow(GSt, 2)) > 1e-8)
    beta = (RSt * GGt - GRt * GSt) / (SSt * GGt - pow(GSt, 2));

  auto alpha = (GRt - beta * GSt) / (GGt);
  new_s = alpha * g + beta * s;
  new_S = alpha * G + beta * S;
  new_sol = x0 + new_s;
  new_res = r - new_S;

  return {new_sol, new_res, new_s, new_S};
}
std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd>
cgdescent(const MatrixXd &g, const MatrixXd &G, const MatrixXd &r,
          const MatrixXd &x0, const MatrixXd &s, const MatrixXd &S,
          const int it) {
  auto k = r.rows();
  MatrixXd new_sol = x0;
  MatrixXd new_res = r;
  MatrixXd new_s = s;
  MatrixXd new_S = S;

  for (auto i = 0; i < it; ++i)
    std::tie(new_sol, new_res, new_s, new_S) =
        cgstep(g, G, new_res, new_sol, new_s, new_S, i);

  return {new_sol, new_res, new_s, new_S};
}
double residual_median(const MatrixXd &res) {
  auto k = (int)res.rows();
  vector<double> r(k);
  for (auto i = 0; i < k; ++i)
    r[i] = res.row(i).norm();

  sort(r.begin(), r.end());

  return (k % 2) ? r[(k + 1) / 2] : r[k / 2];
}
double compute_epsilon(const MatrixXd &d) {
  auto max_val = __DBL_MIN__;
  for (auto i = 0; i < d.rows(); ++i)
    max_val = std::max(max_val, d.row(i).norm());

  return max_val / 100;
}
double compute_epsilon(const VectorXd &d) {
  auto max_val = __DBL_MIN__;
  for (auto i = 0; i < d.rows(); ++i)
    max_val = std::max(max_val, std::abs(d(i)));

  return max_val / 100;
}
vec2d update_weights(MatrixXd &W, const double &epsilon, const MatrixXd &res,
                     const double p, const vector<int> &constrained = {}) {
  int k = res.rows();
  auto sum = 0.0;
  auto sigma = 1.4826 * residual_median(res);
  auto w_max = __DBL_MIN__;
  auto w_min = __DBL_MAX__;
  for (auto i = 0; i < k; ++i) {
    auto curr_r = res.row(i).norm();
    auto w = 0.0;

    // w = (curr_r > 1e-4) ? pow(curr_r, (p - 2) / 2) : pow(1e-4, (p - 2) / 2);
    w = pow(1e-6 + curr_r, (p - 2) / 2);
    if (isnan(w)) {
      std::cout << "we have a problem" << std::endl;
      std::cout << res(i, 0) << std::endl;
      std::cout << res(i, 1) << std::endl;
      std::cout << res(i, 2) << std::endl;
    }

    // W(i, i) = w;
    // sum += w;
    // if (curr_r > 2 * sigma && i >= 7)
    //   w = 0;
    // else
    //   w = 2 / (sigma * pow(1 + pow(curr_r / sigma, 2), 2));

    W(i, i) = w;
    sum += w;
  }
  for (auto i = 0; i < k; ++i) {
    W(i, i) /= sum;
    w_max = std::max(W(i, i), w_max);
    w_min = std::min(W(i, i), w_min);
  }

  return vec2d{w_min, w_max};
}

vec2d update_weights_kalogerakis(MatrixXd &W, const double &epsilon,
                                 const MatrixXd &res, const double p) {
  int k = res.rows();
  auto sum = 0.0;

  auto w_max = __DBL_MIN__;
  auto w_min = __DBL_MAX__;
  for (auto i = 0; i < k; ++i) {
    auto curr_r = res.row(i).norm();
    auto w = 0.0;

    w = (curr_r < 1e-8) ? pow(1e-8, (p - 2) / 2) : pow(curr_r, (p - 2) / 2);

    W(i, i) = w;
    sum += w;
  }
  for (auto i = 0; i < k; ++i) {
    W(i, i) /= sum;
    w_max = std::max(W(i, i), w_max);
    w_min = std::min(W(i, i), w_min);
  }

  return vec2d{w_min, w_max};
}
void update_weights_1d(DiagonalMatrix<double, Dynamic> &W, const VectorXd &d,
                       const VectorXd &res, const double p) {
  int k = res.rows();
  auto sum = 0.0;
  auto epsilon = compute_epsilon(d);
  for (auto i = 0; i < k; ++i) {
    auto curr_r = std::abs(res(i));
    auto w = pow(1e-6 + curr_r, (p - 2) / 2);
    W.diagonal()[i] = w;
    sum += w;
  }
  for (auto i = 0; i < k; ++i) {
    W.diagonal()[i] /= sum;
  }
}
std::tuple<MatrixXd, MatrixXd, MatrixXd, vec2d>
IRLS_Claerbout(const MatrixXd &L, const MatrixXd &d,
               const double epsilon = 1e-6) {
  auto k = (int)d.rows();
  MatrixXd Lt = L.transpose();
  auto cols = (int)L.cols();
  MatrixXd W(k, k);
  W.setIdentity();
  MatrixXd x0(cols, 3);
  x0.setZero();
  MatrixXd res = L * x0 - d;
  auto it = 0;
  vec2d weights_range = vec2d{0, 1};
  auto epsilon_w = compute_epsilon(d);

  MatrixXd s(cols, 3);
  MatrixXd S(k, 3);
  MatrixXd g(cols, 3);
  MatrixXd G(k, 3);
  s.setZero();
  S.setZero();
  auto max_err = max_residual(res);
  while (max_err > epsilon && it < 50 &&
         weights_range.x() != weights_range.y()) {

    weights_range = update_weights(W, epsilon_w, res, 1.5);
    g = 2 * Lt * W * res;
    G = L * g;

    std::tie(x0, res, s, S) = cgstep(g, G, res, x0, s, S, it);
    // ColPivHouseholderQR<MatrixXd> A(W * L);

    // for (auto j = 0; j < 3; ++j)
    //   x0.col(j) = A.solve(W * d.col(j));

    // res = L * x0 - d;
    auto curr_err = max_residual(res);

    if (std::abs(curr_err - max_err) < 1e-10)
      break;
    max_err = curr_err;
    ++it;
  }
  ColPivHouseholderQR<MatrixXd> dec(W * L);

  for (auto i = 0; i < 3; ++i)
    x0.col(i) = dec.solve(W * d.col(i));

  res = L * x0 - d;
  return {x0, W, res, weights_range};
}

std::tuple<VectorXd, DiagonalMatrix<double, Dynamic>, double>
IRLS_Claerbout_1d(const MatrixXd &L, const VectorXd &d,
                  const double epsilon = 1e-6) {
  auto k = (int)d.rows();
  MatrixXd Lt = L.transpose();
  DiagonalMatrix<double, Dynamic> W(k);

  VectorXd x0(5, 1);
  x0.setZero();
  VectorXd r = L * x0 - d;
  VectorXd s(5);
  VectorXd S(k);

  s.setZero();
  S.setZero();

  auto max_err = max_residual(r);
  auto it = 0;
  while (max_err > epsilon && it < 50) {
    update_weights_1d(W, x0, r, 1.5);
    VectorXd g = Lt * r;
    VectorXd G = L * g;

    // std::tie(x0.col(0), res.col(0), sx, Sx) =

    std::tie(x0, r, s, S) = cgstep_1d(g, G, r, x0, s, S, it);

    auto curr_err = max_residual(r);

    if (std::abs(curr_err - max_err) < 1e-10)
      break;
    max_err = curr_err;
    ++it;
  }
  ColPivHouseholderQR<MatrixXd> dec(W * L);
  x0 = dec.solve(W * d);
  // MatrixXd sol(6, 3);
  // for (auto i = 0; i < 6; ++i) {
  //   sol(i, 0) = x0(i);
  //   sol(i, 1) = x0(i + 6);
  //   sol(i, 2) = x0(i + 12);
  // }

  return {x0, W, max_err};
}
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//                                 Geodesic Tracing

vector<vector<double>> Christ_Symbol(const Matrix2d &g,
                                     const vector<vec3d> &X) {
  auto result = vector<vector<double>>(2, vector<double>(3));
  Matrix2d g_inv = g.inverse();

  // Gamma_00^0
  result[0][0] = g_inv(0, 0) * X[0].dot(X[2]) + g_inv(0, 1) * X[1].dot(X[2]);
  // Gamma_01^0
  result[0][1] = g_inv(0, 0) * X[0].dot(X[4]) + g_inv(0, 1) * X[1].dot(X[4]);
  // Gamma_11^0
  result[0][2] = g_inv(0, 0) * X[0].dot(X[3]) + g_inv(0, 1) * X[1].dot(X[3]);

  result[1][0] = g_inv(1, 0) * X[0].dot(X[2]) + g_inv(1, 1) * X[1].dot(X[2]);
  result[1][1] = g_inv(1, 0) * X[0].dot(X[4]) + g_inv(1, 1) * X[1].dot(X[4]);
  result[1][2] = g_inv(1, 0) * X[0].dot(X[3]) + g_inv(1, 1) * X[1].dot(X[3]);

  return result;
}
inline double f3(const vector<vector<double>> &Gamma, const double &u_prime,
                 const double &v_prime) {
  return -Gamma[0][0] * pow(u_prime, 2) - 2 * Gamma[0][1] * u_prime * v_prime -
         Gamma[0][2] * pow(v_prime, 2);
}
inline double f4(const vector<vector<double>> &Gamma, const double &u_prime,
                 const double &v_prime) {
  return -Gamma[1][0] * pow(u_prime, 2) - 2 * Gamma[1][1] * u_prime * v_prime -
         Gamma[1][2] * pow(v_prime, 2);
}

vector<vec4d> RungeKuttaCoeff(const double &pn, const double &qn,
                              const double &step,
                              const vector<vector<double>> &Gamma) {
  auto result = vector<vec4d>(4, vec4d{0, 0, 0, 0});
  result[0].x() = step * pn;
  result[1].x() = step * qn;
  result[2].x() = step * f3(Gamma, pn, qn);
  result[3].x() = step * f4(Gamma, pn, qn);

  result[0].y() = step * (pn + result[2].x() / 2);
  result[1].y() = step * (qn + result[3].x() / 2);
  result[2].y() =
      step * f3(Gamma, pn + result[2].x() / 2, qn + result[3].x() / 2);
  result[3].y() =
      step * f4(Gamma, pn + result[2].x() / 2, qn + result[3].x() / 2);

  result[0].z() = step * (pn + result[2].y() / 2);
  result[1].z() = step * (qn + result[3].y() / 2);
  result[2].z() =
      step * f3(Gamma, pn + result[2].y() / 2, qn + result[3].y() / 2);
  result[3].z() =
      step * f4(Gamma, pn + result[2].y() / 2, qn + result[3].y() / 2);

  result[0].w() = step * (pn + result[2].z());
  result[1].w() = step * (qn + result[3].z());
  result[2].w() = step * f3(Gamma, pn + result[2].z(), qn + result[3].z());
  result[3].w() = step * f4(Gamma, pn + result[2].z(), qn + result[3].z());

  return result;
}
// https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
std::pair<float, float> intersect(const vec2d &direction, const vec2d &left,
                                  const vec2d &right) {
  auto cross = [](const vec2d &u, const vec2d v) {
    return u.x() * v.y() - u.y() * v.x();
  };

  auto v1 = -left;
  auto v2 = right - left;
  auto v3 = vec2d{-direction.y(), direction.x()};
  auto t0 = cross(v2, v1) / v2.dot(v3);
  auto t1 = left.dot(v3) / v2.dot(v3);
  return std::make_pair(t0, t1);
};
std::tuple<vec2d, vec2d> RungeKutta(const double &un, const double &vn,
                                    const double &pn, const double &qn,
                                    const vector<vector<double>> &Gamma,
                                    const double &step) {
  auto coeff = RungeKuttaCoeff(pn, qn, step, Gamma);
  auto coordinates = vec2d{0, 0};
  auto direction = vec2d{0, 0};
  coordinates.x() = un + 1.f / 6 *
                             (coeff[0].x() + 2 * coeff[0].y() +
                              2 * coeff[0].z() + coeff[0].w());
  coordinates.y() = vn + 1.f / 6 *
                             (coeff[1].x() + 2 * coeff[1].y() +
                              2 * coeff[1].z() + coeff[1].w());

  direction.x() = pn + 1.f / 6 *
                           (coeff[2].x() + 2 * coeff[2].y() + 2 * coeff[2].z() +
                            coeff[2].w());
  direction.y() = qn + 1.f / 6 *
                           (coeff[3].x() + 2 * coeff[3].y() + 2 * coeff[3].z() +
                            coeff[3].w());

  return {coordinates, direction};
}
bool isLeftOf(const pair<vec2d, int> &a, const pair<vec2d, int> &b) {
  return (a.first.x() < b.first.x() ||
          (a.first.x() == b.first.x() && a.first.y() < b.first.y()));
}
vector<pair<vec2d, int>> GrahamScan(const vector<pair<vec2d, int>> &p_nbr) {
  if (p_nbr.size() < 3)
    return p_nbr;
  auto v = p_nbr;
  // Put our leftmost point at index 0
  swap(v[0], *min_element(v.begin(), v.end(), isLeftOf));

  // Sort the rest of the points in counter-clockwise order
  // from our leftmost point.
  sort(v.begin() + 1, v.end(), ccwSorter(v[0]));

  // Add our first three points to the hull.
  vector<pair<vec2d, int>> hull;
  auto it = v.begin();
  hull.push_back(*it++);
  hull.push_back(*it++);
  hull.push_back(*it++);

  while (it != v.end()) {
    // Pop off any points that make a convex angle with *it
    while (ccw(*(hull.rbegin() + 1), *(hull.rbegin()), *it) >= 0) {
      hull.pop_back();
    }
    hull.push_back(*it++);
  }

  // clean_convex_hull(hull);

  return hull;
}
vector<pair<double, int>> sorted_sectors(const vector<pair<vec2d, int>> &ch) {
  auto k = ch.size();

  auto angles = vector<pair<double, int>>(k);
  for (auto i = 0; i < k; ++i) {
    auto phi = atan2(ch[i].first.y(), ch[i].first.x());
    if (phi < 0)
      phi += 2 * M_PI;
    angles[i] = make_pair(phi, i);
  }

  sort(angles.begin(), angles.end());

  return angles;
}
bool is_out_of_convex_hull(const vector<pair<vec2d, int>> &ch,
                           const vector<pair<double, int>> &sectors,
                           const vec2d &p) {
  auto theta = atan2(p.y(), p.x());
  if (theta < 0)
    theta += 2 * M_PI;
  vec2d v0, v1;
  int p0, p1;
  auto k = sectors.size();
  double phi;

  for (auto i = 0; i < k; ++i) {
    if (sectors[i].first > theta) {
      if (i == 0) {
        std::tie(v0, p0) = ch[sectors.back().second];
        std::tie(v1, p1) = ch[sectors[i].second];
      } else {
        std::tie(v1, p1) = ch[sectors[i].second];
        std::tie(v0, p0) = ch[sectors[(k - 1 + i) % k].second];
      }
      break;
    } else if (i == k - 1) {
      std::tie(v0, p0) = ch[sectors[i].second];
      std::tie(v1, p1) = ch[sectors[0].second];
      break;
    }
  }
  auto inside = point_in_triangle_2d(p, vec2d{0, 0}, v0, v1);

  return (inside == STRICTLY_OUTSIDE);
}
std::tuple<vector<vec3d>, vec2d, vec2d>
isophotic_geodesic(const patch &p, const int vid, const vec2d &uv_start,
                   const vec2d &dir) {
  auto result = vector<vec3d>{};
  auto &Q = p.quadric;
  auto uv = uv_start;
  auto d = dir;
  auto [g, X] = isophotic_metric(p, uv);
  auto sectors = sorted_sectors(p.CH);

  auto Gamma = Christ_Symbol(g, X);

  while (!is_out_of_convex_hull(p.CH, sectors, uv)) {
    result.push_back(evaluate_quadric(Q, uv));
    std::tie(uv, d) = RungeKutta(uv.x(), uv.y(), d.x(), d.y(), Gamma, 0.001);
    std::tie(g, X) = isophotic_metric(p, uv);
    Gamma = Christ_Symbol(g, X);
  }

  return {result, uv, d};
}
std::tuple<vector<vec3d>, vec2d, vec3d>
isophotic_geodesic_along_principal_direction(const point_cloud &pc,
                                             const int vid, const bool flipped,
                                             const vec2d &uv_start,
                                             const vec3d &prev_dir) {
  auto &p = pc.patches[vid];
  auto &Q = p.quadric;
  auto [g, X] = isophotic_metric(p, uv_start);
  auto sectors = sorted_sectors(p.CH);
  Matrix2d I;
  Matrix2d II;
  auto N = X[0].cross(X[1]);
  N.normalize();
  I << X[0].dot(X[0]), X[0].dot(X[1]), X[0].dot(X[1]), X[1].dot(X[1]);
  II << N.dot(X[2]), N.dot(X[4]), N.dot(X[4]), N.dot(X[3]);
  Matrix2d S = -II * I.inverse();
  SelfAdjointEigenSolver<Matrix2d> dec(S);
  Vector2d dir0 = dec.eigenvectors().col(0);
  Vector2d dir1 = dec.eigenvectors().col(1);
  auto k1 = vec2d{dir0(0), dir0(1)};
  auto k2 = vec2d{dir1(0), dir1(1)};
  auto d = (k1.norm() > k2.norm()) ? k1 : k2;
  if (flipped)
    d *= -1;

  auto dir3d = d.x() * X[0] + d.y() * X[1];
  if (prev_dir.norm() > 0 && dir3d.dot(prev_dir) < 0)
    d *= -1;

  auto [result, uv, final_d] = isophotic_geodesic(p, vid, uv_start, d);
  std::tie(g, X) = isophotic_metric(p, uv);
  dir3d = final_d.x() * X[0] + final_d.y() * X[1];
  return {result, uv, dir3d};
}
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//                            Voronoi

int index_of_singularity_closed_form(const point_cloud &pc, const int vid,
                                     const vector<double> &f) {
  auto &nbr = pc.nbrs[vid];
  auto &p = pc.patches[vid];
  auto k = nbr.size();
  VectorXd local_field(k);
  for (auto i = 0; i < k; ++i)
    local_field(i) = f[nbr[i]];
  VectorXd fit = p.C * local_field;
  auto r = 1e-2;
  auto a1 = fit(1);
  auto a2 = fit(2);
  auto a3 = fit(3);
  auto a4 = fit(4);
  auto a5 = fit(5);

  complex<double> z1(2 * a1, 2 * a2);
  complex<double> z1m(-2 * a1, -2 * a2);
  complex<double> z2(-a5 * r + a3 * r, 2 * a4 * r);
  complex<double> z4(a3 * r, 0);
  complex<double> z5(a5 * r, 0);

  complex<double> c4(4.0, 0.0);

  auto omega1 = (z1m - sqrt(pow(z1, 2) - c4 * z2 * (z4 + z5))) / (z4 + z5);
  auto omega2 = (z1m + sqrt(pow(z1, 2) - c4 * z2 * (z4 + z5))) / (z4 + z5);
  auto omega3 =
      (conj(z1m) - sqrt(pow(conj(z1), 2) - c4 * conj(z2) * (z4 + z5))) /
      conj(z2);

  auto omega4 =
      (conj(z1m) + sqrt(pow(conj(z1), 2) - c4 * conj(z2) * (z4 + z5))) /
      conj(z2);

  if (abs(omega1) >= 2 && abs(omega2) >= 2 && abs(omega3) < 2 &&
      abs(omega4) < 2)
    return -1;
  else if (abs(omega1) < 2 && abs(omega2) < 2 && abs(omega3) >= 2 &&
           abs(omega4) >= 2)
    return 1;

  return 0;
}

vector<vec2i> singularities(const point_cloud &pc, const vector<double> &f) {
  auto sing = vector<vec2i>{};
  sing.reserve(pc.positions.size());
  vector<bool> added(f.size(), false);
  auto grad = compute_grad(pc, f);
  for (auto i = 0; i < f.size(); ++i) {
    auto ind = index_of_singularity_closed_form(pc, i, f);
    if (ind != 0)
      sing.push_back(vec2i{i, ind});
  }
  auto it =
      std::remove_if(sing.begin(), sing.end(), [&pc, &grad](const vec2i &s) {
        auto vid = s.x();
        auto nbr = pc.nbrs[vid];

        //  if (s.y() == 1) {
        //    for (auto nei : nbr) {
        //      if (std::abs(lap[vid]) < std::abs(lap[nei]))
        //        return true;
        //    }
        //  } else if (s.y() == -1) {
        for (auto nei : nbr) {
          if (grad[vid].norm() > grad[nei].norm())
            return true;
        }
        //}

        return false;
      });
  sing.erase(it, sing.end());

  return sing;
}
bool is_local_maximum(const vector<double> &f, const vector<uint> &nbr,
                      const int vid) {
  auto f_vid = f[vid];
  bool is_max = true;
  for (auto &nei : nbr) {
    if (f[nei] > f_vid)
      is_max = false;
  }

  return is_max;
}
bool is_local_miminum(const vector<double> &f, const vector<uint> &nbr,
                      const int vid) {
  auto f_vid = f[vid];
  bool is_min = true;
  for (auto &nei : nbr) {
    if (f[nei] < f_vid)
      is_min = false;
  }

  return is_min;
}
bool is_saddle(const DrawableTrimesh<> &m, const vector<double> &f,
               const int vid) {
  auto f_vid = f[vid];
  auto sum = 0;
  for (auto &tid : m.adj_v2p(vid)) {
    auto off = m.poly_vert_offset(tid, vid);
    auto vid0 = m.poly_vert_id(tid, (off + 1) % 3);
    auto vid1 = m.poly_vert_id(tid, (off + 2) % 3);
    if (min(f[vid0], f[vid1]) < f_vid && max(f[vid0], f[vid1]) > f_vid)
      ++sum;
  }

  return (sum >= 4);
}
vector<vec2i> singularities(const DrawableTrimesh<> &m,
                            const vector<double> &f) {
  auto sing = vector<vec2i>{};
  auto n = m.num_verts();
  sing.reserve(n);
  for (auto i = 0; i < n; ++i) {
    auto nbr = m.adj_v2v(i);
    if (is_local_maximum(f, nbr, i))
      sing.push_back(vec2i{i, 1});
    else if (is_local_miminum(f, nbr, i))
      sing.push_back(vec2i{i, -1});
    else if (is_saddle(m, f, i))
      sing.push_back(vec2i{i, 0});
  }

  return sing;
}
vector<vec2i> singularities(const point_cloud &pc, const vector<double> &f,
                            const vector<bool> &bad) {
  auto sing = vector<vec2i>{};
  sing.reserve(pc.positions.size());
  vector<bool> added(f.size(), false);
  auto grad = compute_grad(pc, f);
  for (auto i = 0; i < f.size(); ++i) {
    auto ind = index_of_singularity_closed_form(pc, i, f);
    if (ind != 0)
      sing.push_back(vec2i{i, ind});
  }
  auto it = std::remove_if(sing.begin(), sing.end(),
                           [&pc, &grad, &bad](const vec2i &s) {
                             auto vid = s.x();
                             auto nbr = pc.nbrs[vid];

                             //  if (s.y() == 1) {
                             //    for (auto nei : nbr) {
                             //      if (std::abs(lap[vid]) <
                             //      std::abs(lap[nei]))
                             //        return true;
                             //    }
                             //  } else if (s.y() == -1) {
                             for (auto nei : nbr) {
                               if (grad[vid].norm() > grad[nei].norm())
                                 return true;
                             }
                             //}

                             return false;
                           });
  sing.erase(it, sing.end());

  return sing;
}
vector<int> singular_vertices(const DrawableTrimesh<> &m,
                              const vector<double> &f) {
  auto sing = vector<int>{};
  auto n = m.num_verts();
  sing.reserve(n);

  for (auto i = 0; i < f.size(); ++i) {
    auto nbr = m.adj_v2v(i);
    if (is_local_maximum(f, nbr, i))
      sing.push_back(i);
    else if (is_local_miminum(f, nbr, i))
      sing.push_back(i);
    else if (is_saddle(m, f, i))
      sing.push_back(i);
  }

  return sing;
}
void clean_singularities(const DrawableTrimesh<> &m, const vector<vec3d> &grad,
                         vector<int> &singularities) {
  auto it = singularities.begin();
  while (it != singularities.end()) {
    auto curr = *it;
    auto nbr = m.vert_n_ring(curr, 5);
    bool found = false;

    for (auto nei : nbr) {
      if (nei == curr)
        continue;
      if (find(singularities.begin(), singularities.end(), (int)nei) !=
              singularities.end() &&
          grad[curr].norm() >= grad[nei].norm()) {
        it = singularities.erase(it);
        found = true;
        break;
      }
    }

    if (!found)
      ++it;
  }
}
vector<int> singular_vertices(const point_cloud &pc, const vector<double> &f) {
  auto sing = vector<int>{};
  sing.reserve(pc.positions.size());
  vector<bool> added(f.size(), false);
  auto grad = compute_grad(pc, f);
  for (auto i = 0; i < f.size(); ++i) {
    auto ind = index_of_singularity_closed_form(pc, i, f);
    if (ind != 0)
      sing.push_back(i);
  }
  auto it =
      std::remove_if(sing.begin(), sing.end(), [&pc, &grad](const int &vid) {
        auto nbr = pc.nbrs[vid];

        for (auto nei : nbr) {
          if (grad[vid].norm() > grad[nei].norm())
            return true;
        }

        return false;
      });
  sing.erase(it, sing.end());

  return sing;
}
vector<int> singular_vertices(const point_cloud &pc, const vector<double> &f,
                              const vector<bool> &bad) {
  auto sing = vector<int>{};
  sing.reserve(pc.positions.size());
  vector<bool> added(f.size(), false);
  auto grad = compute_grad(pc, f);
  for (auto i = 0; i < f.size(); ++i) {
    if (bad[i])
      continue;
    auto ind = index_of_singularity_closed_form(pc, i, f);
    if (ind != 0)
      sing.push_back(i);
  }
  auto it = std::remove_if(sing.begin(), sing.end(),
                           [&pc, &grad, &bad](const int &vid) {
                             auto nbr = pc.nbrs[vid];

                             for (auto nei : nbr) {
                               if (bad[nei])
                                 continue;
                               if (grad[vid].norm() > grad[nei].norm())
                                 return true;
                             }

                             return false;
                           });
  sing.erase(it, sing.end());

  return sing;
}

vector<int> compute_candidates(const vector<double> &phi, const int V,
                               const int nf) {
  auto result = vector<int>{};
  result.reserve(nf * 2);

  auto minima = vector<int>(nf);
  auto maxima = vector<int>(nf);

  for (auto i = 0; i < nf; ++i) {
    auto curr_min = DBL_MAX;
    auto curr_max = DBL_MIN;
    for (auto j = 0; j < V; ++j) {
      auto curr_value = phi[i * V + j];
      if (curr_value < curr_min) {
        curr_min = curr_value;
        minima[i] = j;
      }

      if (curr_value > curr_max) {
        curr_max = curr_value;
        maxima[i] = j;
      }
    }
  }
  for (auto i = 0; i < nf; ++i) {
    if (std::find(result.begin(), result.end(), minima[i]) == result.end())
      result.push_back(minima[i]);

    if (std::find(result.begin(), result.end(), maxima[i]) == result.end())
      result.push_back(maxima[i]);
  }

  return result;
}
vector<double> deserialize_field(const vector<double> &phi, const int V,
                                 const int index) {
  auto result = vector<double>(V);
  for (auto i = 0; i < V; ++i)
    result[i] = phi[index * V + i];

  return result;
}
vector<int> compute_candidates(const vector<double> &phi, const int V,
                               const int nf, const point_cloud &pc) {
  auto result = vector<int>{};
  result.reserve(10 * nf);
  auto sing = vector<vec2i>{};
  sing.reserve(10 * nf);

  for (auto i = 1; i < nf; ++i) {
    auto f = deserialize_field(phi, V, i);
    auto curr_sing = singularities(pc, f);
    sing.insert(sing.end(), curr_sing.begin(), curr_sing.end());
  }
  for (auto i = 0; i < sing.size(); ++i) {
    if (std::find(result.begin(), result.end(), sing[i].x()) == result.end() /*&&
        sing[i].y() == 1*/)
      result.push_back(sing[i].x());
  }

  return result;
}
template <typename Update, typename Stop, typename Exit>
void visit_geodesic_graph(vector<double> &field, const geodesic_solver &solver,
                          const vector<int> &sources, const int type,
                          Update &&update, Stop &&stop, Exit &&exit) {
  /*
     This algortithm uses the heuristic Small Label Fisrt and Large Label Last
     https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm

     Large Label Last (LLL): When extracting nodes from the queue, pick the
     front one. If it weights more than the average weight of the queue, put
     on the back and check the next node. Continue this way.
     Sometimes average_weight is less than every value due to floating point
     errors (doesn't happen with double precision).

     Small Label First (SLF): When adding a new node to queue, instead of
     always pushing it to the end of the queue, if it weights less than the
     front node of the queue, it is put on front. Otherwise the node is put at
     the end of the queue.
  */

  auto in_queue = vector<bool>(solver.graph.size(), false);

  // Cumulative weights of elements in queue. Used to keep track of the
  // average weight of the queue.
  auto cumulative_weight = 0.0;

  // setup queue
  auto queue = deque<int>();
  for (auto source : sources) {
    in_queue[source] = true;
    cumulative_weight += field[source];
    queue.push_back(source);
  }

  while (!queue.empty()) {
    auto node = queue.front();
    auto average_weight = (float)(cumulative_weight / queue.size());

    // Large Label Last (see comment at the beginning)
    for (auto tries = 0; tries < queue.size() + 1; tries++) {
      if (field[node] <= average_weight)
        break;
      queue.pop_front();
      queue.push_back(node);
      node = queue.front();
    }

    // Remove node from queue.
    queue.pop_front();
    in_queue[node] = false;
    cumulative_weight -= field[node];

    // Check early exit condition.
    if (exit(node))
      break;
    if (stop(node))
      continue;

    for (auto i = 0; i < (int)solver.graph[node].size(); i++) {
      // Distance of neighbor through this node
      double new_distance;
      if (type == geodesic)
        new_distance = field[node] + solver.graph[node][i].length;
      else if (type == isophotic)
        new_distance = field[node] + solver.graph[node][i].isophotic_length;
      else
        new_distance =
            field[node] + std::abs(solver.graph[node][i].isophotic_length -
                                   solver.graph[node][i].length);

      auto neighbor = solver.graph[node][i].node;

      auto old_distance = field[neighbor];
      if (new_distance >= old_distance)
        continue;

      if (in_queue[neighbor]) {
        // If neighbor already in queue, don't add it.
        // Just update cumulative weight.
        cumulative_weight += new_distance - old_distance;
      } else {
        // If neighbor not in queue, add node to queue using Small Label
        // First (see comment at the beginning).
        if (queue.empty() || (new_distance < field[queue.front()]))
          queue.push_front(neighbor);
        else
          queue.push_back(neighbor);

        // Update queue information.
        in_queue[neighbor] = true;
        cumulative_weight += new_distance;
      }

      // Update distance of neighbor.
      field[neighbor] = new_distance;
      update(neighbor);
    }
  }
}
template <typename Update, typename Stop, typename Exit>
void visit_geodesic_graph_blended(vector<double> &field,
                                  const geodesic_solver &solver,
                                  const vector<int> &sources,
                                  const double &alpha, Update &&update,
                                  Stop &&stop, Exit &&exit) {
  /*
     This algortithm uses the heuristic Small Label Fisrt and Large Label Last
     https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm

     Large Label Last (LLL): When extracting nodes from the queue, pick the
     front one. If it weights more than the average weight of the queue, put
     on the back and check the next node. Continue this way.
     Sometimes average_weight is less than every value due to floating point
     errors (doesn't happen with double precision).

     Small Label First (SLF): When adding a new node to queue, instead of
     always pushing it to the end of the queue, if it weights less than the
     front node of the queue, it is put on front. Otherwise the node is put at
     the end of the queue.
  */

  auto in_queue = vector<bool>(solver.graph.size(), false);

  // Cumulative weights of elements in queue. Used to keep track of the
  // average weight of the queue.
  auto cumulative_weight = 0.0;

  // setup queue
  auto queue = deque<int>();
  for (auto source : sources) {
    in_queue[source] = true;
    cumulative_weight += field[source];
    queue.push_back(source);
  }

  while (!queue.empty()) {
    auto node = queue.front();
    auto average_weight = (float)(cumulative_weight / queue.size());

    // Large Label Last (see comment at the beginning)
    for (auto tries = 0; tries < queue.size() + 1; tries++) {
      if (field[node] <= average_weight)
        break;
      queue.pop_front();
      queue.push_back(node);
      node = queue.front();
    }

    // Remove node from queue.
    queue.pop_front();
    in_queue[node] = false;
    cumulative_weight -= field[node];

    // Check early exit condition.
    if (exit(node))
      break;
    if (stop(node))
      continue;

    for (auto i = 0; i < (int)solver.graph[node].size(); i++) {
      // Distance of neighbor through this node
      double geo_len = solver.graph[node][i].length;
      double iso_len = solver.graph[node][i].isophotic_length;
      double len = (1 - alpha) * geo_len + alpha * iso_len;

      auto new_distance = field[node] + len;

      auto neighbor = solver.graph[node][i].node;

      auto old_distance = field[neighbor];
      if (new_distance >= old_distance)
        continue;

      if (in_queue[neighbor]) {
        // If neighbor already in queue, don't add it.
        // Just update cumulative weight.
        cumulative_weight += new_distance - old_distance;
      } else {
        // If neighbor not in queue, add node to queue using Small Label
        // First (see comment at the beginning).
        if (queue.empty() || (new_distance < field[queue.front()]))
          queue.push_front(neighbor);
        else
          queue.push_back(neighbor);

        // Update queue information.
        in_queue[neighbor] = true;
        cumulative_weight += new_distance;
      }

      // Update distance of neighbor.
      field[neighbor] = new_distance;
      update(neighbor);
    }
  }
}
void update_geodesic_distances(vector<double> &distances,
                               const geodesic_solver &solver,
                               const vector<int> &sources, const int type,
                               double max_distance = DBL_MAX) {

  auto update = [](int node) {};
  auto stop = [&](int node) { return distances[node] > max_distance; };
  auto exit = [](int node) { return false; };
  for (auto source : sources)
    distances[source] = 0.0;
  visit_geodesic_graph(distances, solver, sources, type, update, stop, exit);
}
void update_euclidean_distances(vector<double> &distances,
                                const vector<int> &sources,
                                const DrawableTrimesh<> &m) {
  for (auto i = 0; i < sources.size(); ++i) {
    auto pos = m.vert(sources[i]);
    for (auto j = 0; j < m.num_verts(); ++j)
      distances[i] = min(distances[i], (m.vert(j) - pos).norm());
  }
}
double minimum_biharmonic_distance(const MatrixXd &D,
                                   const vector<int> &sources, const int vid) {
  auto d = DBL_MAX;
  for (auto i = 0; i < sources.size(); ++i) {
    d = min(d, D(sources[i], vid));
  }

  return d;
}
void update_biharmonic_distances(vector<double> &distances,
                                 const vector<int> &sources,
                                 const MatrixXd &D) {
  auto N = D.rows();

  for (auto j = 0; j < N; ++j) {
    distances[j] = minimum_biharmonic_distance(D, sources, j);
  }
}

vector<double> spectral_embedding(const vector<double> &phi, const int vid,
                                  const int V) {
  auto n = (int)phi.size();
  if (n % V != 0)
    std::cout << "Error! the number of entries should be divisible by the "
                 "number of verts"
              << std::endl;
  auto nf = n / V;
  auto result = vector<double>(nf);
  for (auto i = 0; i < nf; ++i) {
    result[i] = phi[i * V + vid];
  }

  return result;
}
double norm(const vector<double> &v0, const vector<double> &v1) {
  auto result = 0.0;
  for (auto i = 0; i < v0.size(); ++i) {
    result += pow(v0[i] - v1[i], 2);
  }

  return sqrt(result);
}
double norm_squared(const vector<double> &v0, const vector<double> &v1) {
  auto result = 0.0;
  for (auto i = 0; i < v0.size(); ++i) {
    result += pow(v0[i] - v1[i], 2);
  }

  return result;
}
double norm_infty(const vector<double> &v0, const vector<double> &v1) {
  auto result = DBL_MIN;
  for (auto i = 0; i < v0.size(); ++i) {
    result = max(result, std::abs(v0[i] - v1[i]));
  }

  return result;
}
double norm_infty(const vector<pair<double, int>> &v0,
                  const vector<pair<double, int>> &v1) {
  auto result = DBL_MIN;
  for (auto i = 0; i < v0.size(); ++i) {

    result = max(result, std::abs(v0[i].first - v1[i].first));
  }

  return result;
}
double norm_squared(const vector<pair<double, int>> &v0,
                    const vector<pair<double, int>> &v1) {
  auto result = 0.0;
  for (auto i = 0; i < v0.size(); ++i) {

    result += pow(v0[i].first - v1[i].first, 2);
  }

  return result;
}
double weighted_norm(const vector<double> &v0, const vector<double> &v1) {
  auto result = 0.0;
  for (auto i = 0; i < v0.size(); ++i) {
    result += exp(-(v0[i] + v1[i])) * pow(v0[i] - v1[i], 2);
  }

  return sqrt(result);
}
vector<double> diff_vec(const vector<double> &v0, const vector<double> &v1) {
  auto s = v0.size();
  auto result = vector<double>(s);
  for (auto i = 0; i < s; ++i)
    result[i] = std::abs(v0[i] - v1[i]);

  return result;
}
double norm(const vector<int> &v0, const vector<int> &v1) {
  auto result = 0.0;

  for (auto i = 0; i < 5; ++i) {
    result += pow(v0[i] - v1[i], 2);
  }

  return sqrt(result);
}
double norm(const vector<pair<double, int>> &v0,
            const vector<pair<double, int>> &v1) {

  auto s = min(v0.size(), v1.size());
  auto v0t = vector<int>(s);
  auto v1t = vector<int>(s);
  for (auto i = 0; i < s; ++i) {
    // if (v0[i].second != v1[i].second)
    //   return DBL_MAX;
    v0t[i] = v0[i].first;
    v1t[i] = v1[i].first;
  }

  return norm(v0t, v1t);
}
void update_spectral_distances(vector<double> &distances,
                               const vector<int> &sources,
                               const vector<double> &phi) {
  auto V = (int)distances.size();

  for (auto i = 0; i < sources.size(); ++i) {
    auto curr = spectral_embedding(phi, sources[i], V);
    for (auto j = 0; j < V; ++j)
      distances[i] =
          min(distances[i], norm(curr, spectral_embedding(phi, j, V)));
    ;
  }
}
vector<int> update_distances(vector<double> &distances,
                             const geodesic_solver &solver,
                             const vector<int> &sources, const int type,
                             double max_distance = DBL_MAX) {
  auto voronoi = vector<int>{};
  auto update = [&voronoi](int node) { voronoi.push_back(node); };
  auto stop = [&](int node) { return distances[node] > max_distance; };
  auto exit = [](int node) { return false; };
  visit_geodesic_graph(distances, solver, sources, type, update, stop, exit);
  return voronoi;
}
vector<int> update_euclidean_distances(vector<double> &distances,
                                       const DrawableTrimesh<> &m,
                                       const int source) {
  auto voronoi = vector<int>{};
  voronoi.reserve(m.num_verts());
  auto pos = m.vert(source);
  for (auto i = 0; i < m.num_verts(); ++i) {
    auto curr_d = (m.vert(i) - pos).norm();
    if (curr_d < distances[i]) {
      distances[i] = curr_d;
      voronoi.push_back(i);
    }
  }

  return voronoi;
}
vector<int> update_biharmonic_distances(vector<double> &distances,
                                        const MatrixXd &D, const int source) {
  auto voronoi = vector<int>{};
  auto N = D.rows();
  voronoi.reserve(N);

  for (auto i = 0; i < N; ++i) {
    auto curr_d = D(source, i);
    if (curr_d < distances[i]) {
      distances[i] = curr_d;
      voronoi.push_back(i);
    }
  }

  return voronoi;
}
vector<int> update_spectral_distances(vector<double> &distances,
                                      const DrawableTrimesh<> &m,
                                      const int source,
                                      const vector<double> &phi) {
  auto voronoi = vector<int>{};
  auto V = m.num_verts();
  voronoi.reserve(V);
  auto pos = spectral_embedding(phi, source, V);

  for (auto i = 0; i < V; ++i) {
    auto curr_d = norm(pos, spectral_embedding(phi, i, V));
    if (curr_d < distances[i]) {
      distances[i] = curr_d;
      voronoi.push_back(i);
    }
  }

  return voronoi;
}
vector<int> update_geodesic_distances_blended(vector<double> &distances,
                                              const geodesic_solver &solver,
                                              const vector<int> &sources,
                                              const double &alpha,
                                              double max_distance = DBL_MAX) {

  auto voronoi = vector<int>{};
  auto update = [&voronoi](int node) { voronoi.push_back(node); };
  auto stop = [&](int node) { return distances[node] > max_distance; };
  auto exit = [](int node) { return false; };
  visit_geodesic_graph_blended(distances, solver, sources, alpha, update, stop,
                               exit);

  return voronoi;
}
vector<int> update_voronoi_verts(iVd &voronoi, const DrawableTrimesh<> &mesh,
                                 const vector<int> &curr_region) {
  auto updated = vector<int>{};
  auto parsed = vector<bool>(mesh.num_polys(), false);
  for (auto vert : curr_region) {
    auto star = mesh.adj_v2p(vert);
    for (auto tri : star) {
      if (parsed[tri])
        continue;
      parsed[tri] = true;
      auto it = find(voronoi.voronoi_verts.begin(), voronoi.voronoi_verts.end(),
                     (int)tri);

      auto tag_x = voronoi.voronoi_tags[mesh.poly_vert_id(tri, 0)];
      auto tag_y = voronoi.voronoi_tags[mesh.poly_vert_id(tri, 1)];
      auto tag_z = voronoi.voronoi_tags[mesh.poly_vert_id(tri, 2)];

      if (tag_x != tag_y && tag_x != tag_z && tag_y != tag_z) {
        if (it == voronoi.voronoi_verts.end())
          voronoi.voronoi_verts.push_back(tri);
        updated.push_back(tri);
      } else if (it != voronoi.voronoi_verts.end()) {
        voronoi.voronoi_verts.erase(it);
      }
    }
  }
  return updated;
}
void update_voronoi_regions(iVd &vor, const vector<int> &voronoi_centers,
                            const DrawableTrimesh<> &m) {
  vor.voronoi_regions.clear();
  vor.voronoi_regions.resize(voronoi_centers.size());
  for (auto &region : vor.voronoi_regions) {
    region.clear();
    region.reserve(m.num_verts());
  }

  for (auto i = 0; i < m.num_verts(); ++i) {
    auto tag = vor.voronoi_tags[i];
    auto it = find(voronoi_centers.begin(), voronoi_centers.end(), tag);
    if (it == voronoi_centers.end())
      std::cout << "Error! This point should be a center" << std::endl;

    auto entry = distance(voronoi_centers.begin(), it);
    vor.voronoi_regions[entry].push_back(i);
  }
}
pair<double, int> max_normal_deviation(const iVd &vor,
                                       const vector<int> &centers,
                                       const int entry,
                                       const DrawableTrimesh<> &m) {

  auto n = m.vert_data(centers[entry]).normal;
  auto max_dev = DBL_MIN;
  auto vid = -1;
  for (auto vert : vor.voronoi_regions[entry]) {
    auto curr_dev = n.angle_rad(m.vert_data(vert).normal);
    if (curr_dev > max_dev) {
      max_dev = curr_dev;
      vid = vert;
    }
  }

  return {max_dev, vid};
}
void update_normal_deviation(iVd &vor, const vector<int> &voronoi_centers,
                             const DrawableTrimesh<> &m) {

  for (auto i = 0; i < voronoi_centers.size(); ++i) {
    vor.region_normal_deviation[i] =
        max_normal_deviation(vor, voronoi_centers, i, m);
  }
}

void update_normal_deviation(iVd &vor, const vector<int> &voronoi_centers,
                             const DrawableTrimesh<> &m, const int entry) {
  vor.region_normal_deviation[entry] =
      max_normal_deviation(vor, voronoi_centers, entry, m);
}

vector<double> compute_euclidean_distances(const vector<int> &sources,
                                           const DrawableTrimesh<> &m) {

  auto field = vector<double>(m.num_verts(), DBL_MAX);
  for (auto source : sources)
    field[source] = 0.0;

  update_euclidean_distances(field, sources, m);

  return field;
}
vector<double> compute_biharmonic_distances(const vector<int> &sources,
                                            const MatrixXd &D) {

  auto field = vector<double>(D.rows(), DBL_MAX);
  for (auto source : sources)
    field[source] = 0.0;

  update_biharmonic_distances(field, sources, D);

  return field;
}
vector<double> compute_spectral_distances(const vector<int> &sources,
                                          const vector<double> &phi,
                                          const DrawableTrimesh<> &m) {

  auto field = vector<double>(m.num_verts(), DBL_MAX);
  for (auto source : sources)
    field[source] = 0.0;

  update_spectral_distances(field, sources, phi);

  return field;
}
SparseMatrix<double> invert_diag_matrix(const SparseMatrix<double> &M) {

  SparseMatrix<double> M_inv;
  // http://www.alecjacobson.com/weblog/?p=2552
  if (&M_inv != &M) {
    M_inv = M;
  }
  // Iterate over outside
  for (int k = 0; k < M_inv.outerSize(); ++k) {
    // Iterate over inside
    for (typename Eigen::SparseMatrix<double>::InnerIterator it(M_inv, k); it;
         ++it) {
      if (it.col() == it.row()) {
        double v = it.value();
        assert(v != 0);
        v = 1.0 / v;
        M_inv.coeffRef(it.row(), it.col()) = v;
      }
    }
  }
  return M_inv;
}
vector<double> cotangent_laplacian(const SparseMatrix<double> &L,
                                   const SparseMatrix<double> &M,
                                   const vector<double> &field) {

  auto M_inv = invert_diag_matrix(M);
  auto F = wrapper(field);
  Eigen::VectorXd Lap = M_inv * F;
  vector<double> laplacian(field.size());
  for (auto i = 0; i < Lap.size(); ++i) {
    laplacian[i] = (float)Lap(i);
  }
  return laplacian;
}
MatrixXd compute_biharmonic_distance(const DrawableTrimesh<> &m) {
  int N = m.num_verts();
  MatrixXd D(N, N);

  SparseMatrix<double> A = mass_matrix(m);
  auto A_inv = invert_diag_matrix(A);
  auto Lc = laplacian(m, COTANGENT);

  SparseMatrix<double> Ld;
  Ld = A_inv * Lc;

  // compute Lc * A.inverse * Lc
  MatrixXd LcA_Lc = Lc * Ld;
  LcA_Lc.row(0).setZero();
  LcA_Lc.col(0).setZero();
  LcA_Lc(0, 0) = 1;

  // construct the J matrix (defined in paper)
  MatrixXd J;
  J.resize(N, N);
  VectorXd Ones = VectorXd::Ones(N);
  J = MatrixXd::Identity(N, N) - (1.0 / N) * (Ones * Ones.transpose());
  J.row(0).setZero();

  // LcA_Lc is now invertible
  MatrixXd Gd;
  Gd = LcA_Lc.llt().solve(J);

  VectorXd off = (1.0 / N) * Gd.colwise().sum();
  MatrixXd offset = (off * Ones.transpose()).transpose();
  Gd -= offset;

  // D(i, j)^2 = Gd(i, i) + Gd(j, j) - 2 * Gd(i, j)
  VectorXd diag = Gd.diagonal();
  MatrixXd dd = diag * Ones.transpose();
  D = sqrt((dd + dd.transpose() - 2 * Gd).array());

  return D;
}
std::pair<vector<double>, vector<int>>
exact_geodesic_wrapper(const vector<vector<uint>> &triangles,
                       const vector<vec3d> &positions) {
  int V = (int)positions.size();
  int F = (int)triangles.size();
  vector<double> points(3 * V);
  vector<int> faces(3 * F);
  vector<double> f(V);

  for (int i = 0; i < V; ++i) {
    points[3 * i] = positions[i].x();
    points[3 * i + 1] = positions[i].y();
    points[3 * i + 2] = positions[i].z();
  }
  for (int i = 0; i < F; ++i) {
    faces[3 * i] = triangles[i][0];
    faces[3 * i + 1] = triangles[i][1];
    faces[3 * i + 2] = triangles[i][2];
  }

  return std::make_pair(points, faces);
}
vector<double> exact_geodesic_distance(const DrawableTrimesh<> &m,
                                       const int &source) {
  int V = m.num_verts();
  vector<double> f(V);
  auto [points, faces] =
      exact_geodesic_wrapper(m.vector_polys(), m.vector_verts());
  geodesic_VTP::Mesh mesh;
  mesh.initialize_mesh_data(points, faces);
  geodesic_VTP::GeodesicAlgorithmExact algorithm(&mesh);
  algorithm.propagate(source);
  vector<geodesic_VTP::Vertex> verts = mesh.vertices();
  for (int j = 0; j < V; ++j) {
    geodesic_VTP::Vertex v = verts[j];
    f[j] = v.geodesic_distance();
  }

  return f;
}
vector<double> compute_geodesic_distances(const geodesic_solver &solver,
                                          const vector<int> &sources,
                                          const int type) {

  auto field = vector<double>(solver.graph.size(), DBL_MAX);
  for (auto source : sources)
    field[source] = 0.0;

  update_geodesic_distances(field, solver, sources, type);

  return field;
}
vector<double> compute_geodesic_distances_blended(const geodesic_solver &solver,
                                                  const vector<int> &sources,
                                                  const double &alpha) {

  auto field = vector<double>(solver.graph.size(), DBL_MAX);
  for (auto source : sources)
    field[source] = 0.0;

  update_geodesic_distances_blended(field, solver, sources, alpha);

  return field;
}
vector<vec3d> isophotic_geodesic(const point_cloud &pc, const int start,
                                 const int target, const double &alpha) {
  auto d = compute_geodesic_distances_blended(pc.solver, {target}, alpha);
  auto curr = start;
  auto result = vector<vec3d>{};
  result.reserve(pc.positions.size());
  result.push_back(pc.positions[start]);
  while (curr != target) {
    auto nbr = pc.solver.graph[curr];
    auto dist = DBL_MAX;
    auto new_one = -1;
    for (auto &nei : nbr) {
      if (d[nei.node] < dist) {
        dist = d[nei.node];
        new_one = nei.node;
      }
    }
    curr = new_one;
    result.push_back(pc.positions[curr]);
  }

  return result;
}
vector<int> add_point_to_sampling(iVd &vor, const geodesic_solver &solver,
                                  const DrawableTrimesh<> &mesh,
                                  const vector<double> &phi, const MatrixXd &D,
                                  const int vid, const int type) {
  vor.distances[vid] = 0;
  vor.voronoi_tags[vid] = vid;
  auto normal_dev = DBL_MIN;
  auto vid0 = -1;
  auto n = mesh.vert_data(vid).normal;
  auto curr_voronoi = vector<int>{};
  if (type == Euclidean)
    curr_voronoi = update_euclidean_distances(vor.distances, mesh, vid);
  else if (type == Spectral)
    curr_voronoi = update_spectral_distances(vor.distances, mesh, vid, phi);
  else if (type == Biharmonic)
    curr_voronoi = update_biharmonic_distances(vor.distances, D, vid);
  else
    curr_voronoi = update_distances(vor.distances, solver, {vid}, type, vor.R);

  for (auto vert : curr_voronoi) {
    vor.voronoi_tags[vert] = vid;
    auto dev = mesh.vert_data(vert).normal.angle_rad(n);
    if (dev > normal_dev) {
      normal_dev = dev;
      vid0 = vert;
    }
  }
  vor.region_normal_deviation.push_back(make_pair(normal_dev, vid0));

  return curr_voronoi;
}
vector<int> add_point_to_sampling(iVd &vor, const geodesic_solver &solver,
                                  const DrawableTrimesh<> &mesh,
                                  const int vid) {
  vor.distances[vid] = 0;
  vor.voronoi_tags[vid] = vid;
  auto vid0 = -1;
  auto n = mesh.vert_data(vid).normal;
  auto curr_voronoi = vector<int>{};

  curr_voronoi =
      update_distances(vor.distances, solver, {vid}, geodesic, vor.R);

  for (auto vert : curr_voronoi)
    vor.voronoi_tags[vert] = vid;

  return curr_voronoi;
}
vector<int> add_point_to_sampling_blended(iVd &vor,
                                          const geodesic_solver &solver,
                                          const DrawableTrimesh<> &mesh,
                                          const int vid, const double &alpha) {
  vor.distances[vid] = 0;
  vor.voronoi_tags[vid] = vid;
  auto normal_dev = DBL_MIN;
  auto vid0 = -1;
  auto n = mesh.vert_data(vid).normal;
  auto curr_voronoi = update_geodesic_distances_blended(vor.distances, solver,
                                                        {vid}, alpha, vor.R);
  for (auto vert : curr_voronoi) {
    vor.voronoi_tags[vert] = vid;
    auto dev = mesh.vert_data(vert).normal.angle_rad(n);
    if (dev > normal_dev) {
      normal_dev = dev;
      vid0 = vert;
    }
  }
  vor.region_normal_deviation.push_back(make_pair(normal_dev, vid0));

  return curr_voronoi;
}
vector<int> farthest_point_sampling(const geodesic_solver &solver,
                                    const DrawableTrimesh<> &mesh,
                                    const int seed, const int k) {

  auto result = vector<int>(k);
  result[0] = seed;
  auto distances = compute_geodesic_distances(solver, {result[0]});
  auto R = DBL_MAX;
  for (auto i = 1; i < k; ++i) {
    result[i] = distance(distances.begin(),
                         std::max_element(distances.begin(), distances.end()));
    update_geodesic_distances(distances, solver, {result[i]}, geodesic, R);

    R = *std::max_element(distances.begin(), distances.end());
  }

  return result;
}
int farthest_admissible_point(const vector<double> &distances,
                              const vector<int> &gt) {
  auto len = DBL_MIN;
  auto result = -1;
  for (auto &vid : gt) {
    if (distances[vid - 1] > len) {
      len = distances[vid - 1];
      result = vid - 1;
    }
  }

  return result;
}
vector<int> constrained_farthest_point_sampling(const geodesic_solver &solver,
                                                const DrawableTrimesh<> &mesh,
                                                const vector<int> &gt,
                                                const int seed, const int k) {

  auto result = vector<int>(k);
  result[0] = seed;
  auto distances = compute_geodesic_distances(solver, {result[0]});
  auto R = DBL_MAX;
  for (auto i = 1; i < k; ++i) {
    result[i] = farthest_admissible_point(distances, gt);
    update_geodesic_distances(distances, solver, {result[i]}, geodesic, R);

    R = *std::max_element(distances.begin(), distances.end());
  }

  return result;
}

pair<iVd, vector<int>> farthest_point_sampling(const geodesic_solver &solver,
                                               const DrawableTrimesh<> &mesh,
                                               const vector<double> &phi,
                                               const MatrixXd &D,
                                               const vector<int> seeds,
                                               const int type, const int k) {
  iVd vor = {};
  vor.voronoi_tags = vector<int>(solver.graph.size(), seeds[0]);
  auto voronoi_center = vector<int>{seeds[0]};
  vor.region_normal_deviation.push_back({DBL_MAX, -1});
  if (type == Euclidean)
    vor.distances = compute_euclidean_distances(seeds, mesh);
  else if (type == Spectral)
    vor.distances = compute_spectral_distances(seeds, phi, mesh);
  else if (type == Biharmonic)
    vor.distances = compute_biharmonic_distances(seeds, D);
  else
    vor.distances = compute_geodesic_distances(solver, {seeds[0]}, type);

  update_voronoi_regions(vor, voronoi_center, mesh);
  vor.R = DBL_MAX;

  for (auto i = 1; i < seeds.size(); ++i) {

    voronoi_center.push_back(seeds[i]);
    auto curr_voronoi =
        add_point_to_sampling(vor, solver, mesh, phi, D, seeds[i], type);
    update_voronoi_verts(vor, mesh, curr_voronoi);
    update_voronoi_regions(vor, voronoi_center, mesh);
    vor.R = *std::max_element(vor.distances.begin(), vor.distances.end());
  }
  // update_voronoi_regions(vor, voronoi_center, mesh);
  // update_normal_deviation(vor, voronoi_center, mesh, 0);
  // vor.R = *std::max_element(vor.distances.begin(), vor.distances.end());
  return {vor, voronoi_center};
}
pair<iVd, vector<int>> farthest_point_sampling(const geodesic_solver &solver,
                                               const DrawableTrimesh<> &mesh,
                                               const vector<int> seeds) {
  iVd vor = {};
  vor.voronoi_tags = vector<int>(solver.graph.size(), seeds[0]);
  auto voronoi_center = vector<int>{seeds[0]};
  vor.region_normal_deviation.push_back({DBL_MAX, -1});

  vor.distances = compute_geodesic_distances(solver, {seeds[0]}, geodesic);

  update_voronoi_regions(vor, voronoi_center, mesh);
  vor.R = DBL_MAX;

  for (auto i = 1; i < seeds.size(); ++i) {

    voronoi_center.push_back(seeds[i]);
    auto curr_voronoi = add_point_to_sampling(vor, solver, mesh, seeds[i]);
    update_voronoi_verts(vor, mesh, curr_voronoi);
    update_voronoi_regions(vor, voronoi_center, mesh);
    vor.R = *std::max_element(vor.distances.begin(), vor.distances.end());
  }
  // update_voronoi_regions(vor, voronoi_center, mesh);
  // update_normal_deviation(vor, voronoi_center, mesh, 0);
  // vor.R = *std::max_element(vor.distances.begin(), vor.distances.end());
  return {vor, voronoi_center};
}
pair<iVd, vector<int>>
farthest_point_sampling_blended(const geodesic_solver &solver,
                                const DrawableTrimesh<> &mesh,
                                const vector<int> seeds, const double &alpha) {
  iVd vor = {};
  vor.voronoi_tags = vector<int>(solver.graph.size(), seeds[0]);
  auto voronoi_center = vector<int>{seeds[0]};
  vor.region_normal_deviation.push_back({DBL_MAX, -1});
  vor.distances = compute_geodesic_distances_blended(solver, {seeds[0]}, alpha);

  vor.R = DBL_MAX;
  auto euler = 2 * (1 - mesh.genus());
  auto k = 9;
  for (auto i = 1; i < seeds.size(); ++i) {
    auto curr = seeds[i];
    voronoi_center.push_back(curr);
    auto curr_voronoi =
        add_point_to_sampling_blended(vor, solver, mesh, curr, alpha);
    update_voronoi_verts(vor, mesh, curr_voronoi);
  }
  update_voronoi_regions(vor, voronoi_center, mesh);
  update_normal_deviation(vor, voronoi_center, mesh, 0);
  vor.R = *std::max_element(vor.distances.begin(), vor.distances.end());
  return {vor, voronoi_center};
}
int find_new_point(const iVd &vor, const int center, const DrawableTrimesh<> &m,
                   const point_cloud &pc) {
  auto max_curv = DBL_MIN;
  auto vid = -1;
  for (auto i = 0; i < m.num_polys(); ++i) {
    auto vid0 = m.poly_vert_id(i, 0);
    auto vid1 = m.poly_vert_id(i, 1);
    auto vid2 = m.poly_vert_id(i, 2);
    auto tagx = vor.voronoi_tags[vid0];
    auto tagy = vor.voronoi_tags[vid1];
    auto tagz = vor.voronoi_tags[vid2];
    if (tagx == tagy && tagx == tagz)
      continue;
    else if (tagx == center) {
      auto curv = std::abs(gaussian_curvature(pc, vid0));
      if (curv > max_curv) {
        max_curv = curv;
        vid = vid0;
      }
    }
    if (tagy == center) {
      auto curv = std::abs(gaussian_curvature(pc, vid1));
      if (curv > max_curv) {
        max_curv = curv;
        vid = vid1;
      }
    }
    if (tagz == center) {
      auto curv = std::abs(gaussian_curvature(pc, vid2));
      if (curv > max_curv) {
        max_curv = curv;
        vid = vid2;
      }
    }
  }

  return vid;
}
void refine_voronoi_diagram(iVd &vor, vector<int> &vor_centers,
                            const geodesic_solver &solver,
                            const DrawableTrimesh<> &mesh,
                            const vector<double> &phi, const MatrixXd &D,
                            const point_cloud &pc, const int type) {
  // auto it = max_element(vor.distances.begin(), vor.distances.end());

  auto it = max_element(vor.region_normal_deviation.begin(),
                        vor.region_normal_deviation.end());
  if (it->first < M_PI_2) {
    std::cout << "No refinement needed" << std::endl;
    return;
  }
  while (it->first > M_PI_2) {
    auto new_center =
        // distance(vor.distances.begin(),
        //          std::max_element(vor.distances.begin(),
        //          vor.distances.end()));
        it->second;

    auto curr_voronoi =
        add_point_to_sampling(vor, solver, mesh, phi, D, new_center, type);
    update_voronoi_verts(vor, mesh, curr_voronoi);
    vor_centers.push_back(new_center);
    update_voronoi_regions(vor, vor_centers, mesh);
    update_normal_deviation(vor, vor_centers, mesh);
    it = max_element(vor.region_normal_deviation.begin(),
                     vor.region_normal_deviation.end());
    vor.R = *std::max_element(vor.distances.begin(), vor.distances.end());
  }
  printf("NUmber of centers is %d", (int)vor_centers.size());
}
void refine_voronoi_diagram(iVd &vor, vector<int> &vor_centers,
                            const geodesic_solver &solver,
                            const DrawableTrimesh<> &mesh,
                            const vector<double> &phi, const MatrixXd &D,
                            const point_cloud &pc, const int type,
                            const int k) {
  // auto it = max_element(vor.distances.begin(), vor.distances.end());

  while (vor_centers.size() < k) {
    auto new_center =
        distance(vor.distances.begin(),
                 std::max_element(vor.distances.begin(), vor.distances.end()));
    // it->second;
    // distance(vor.distances.begin(), it);
    auto curr_voronoi =
        add_point_to_sampling(vor, solver, mesh, phi, D, new_center, type);
    update_voronoi_verts(vor, mesh, curr_voronoi);
    vor_centers.push_back(new_center);
    update_voronoi_regions(vor, vor_centers, mesh);
    update_normal_deviation(vor, vor_centers, mesh);
    vor.R = *std::max_element(vor.distances.begin(), vor.distances.end());
  }
}
void refine_voronoi_diagram_blended(iVd &vor, vector<int> &vor_centers,
                                    const geodesic_solver &solver,
                                    const DrawableTrimesh<> &mesh,
                                    const vector<double> &phi,
                                    const MatrixXd &D, const point_cloud &pc,
                                    const double &alpha, const int k) {
  auto it = max_element(vor.distances.begin(), vor.distances.end());
  auto new_center = *it;
  // auto it = max_element(vor.region_normal_deviation.begin(),
  //                       vor.region_normal_deviation.end());
  // if (it->first < M_PI_2) {
  //   std::cout << "No refinement needed" << std::endl;
  //   return;
  // }
  // while (it->first > M_PI_2) {
  while (vor_centers.size() < k) {
    // auto new_center =
    //     // distance(vor.distances.begin(),
    //     //          std::max_element(vor.distances.begin(),
    //     //          vor.distances.end()));
    //     it->second;
    // distance(vor.distances.begin(), it);
    auto curr_voronoi =
        add_point_to_sampling(vor, solver, mesh, phi, D, new_center, alpha);
    update_voronoi_verts(vor, mesh, curr_voronoi);
    vor_centers.push_back(new_center);
    update_voronoi_regions(vor, vor_centers, mesh);
    update_normal_deviation(vor, vor_centers, mesh);
    // it = max_element(vor.region_normal_deviation.begin(),
    //                  vor.region_normal_deviation.end());
    it = max_element(vor.distances.begin(), vor.distances.end());
    vor.R = *std::max_element(vor.distances.begin(), vor.distances.end());
  }
  printf("NUmber of centers is %d", (int)vor_centers.size());
}
void build_tangent_spaces(iVd &vor, const point_cloud &pc,
                          const vector<int> &voronoi_centers) {
  vor.parametric_nbr.resize(voronoi_centers.size());
  for (auto nbr : vor.parametric_nbr)
    nbr.reserve(pc.positions.size());
  vor.basis.resize(voronoi_centers.size());
  for (auto i = 0; i < voronoi_centers.size(); ++i) {
    vor.basis[i] = pc.basis[voronoi_centers[i]];
    auto [p_nbr, vids] =
        project_points(voronoi_centers[i], pc, vor.voronoi_regions[i]);
    for (auto j = 0; j < p_nbr.size(); ++j)
      vor.parametric_nbr[i].push_back(make_pair(p_nbr[j], vids[j]));
  }
}

pair<iVd, vector<int>> intrinsic_voronoi_diagram(
    const geodesic_solver &solver, const DrawableTrimesh<> &mesh,
    const vector<double> &phi, const MatrixXd &D, const point_cloud &pc,
    const int type, const vector<int> seeds) {
  auto [vor, vor_centers] =
      farthest_point_sampling(solver, mesh, phi, D, seeds, type);
  // refine_voronoi_diagram(vor, vor_centers, solver, mesh, phi, D, pc, type,
  // k);
  //   build_tangent_spaces(vor, pc, vor_centers);
  return {vor, vor_centers};
}
iVd intrinsic_voronoi_diagram(const geodesic_solver &solver,
                              const DrawableTrimesh<> &mesh,
                              const vector<int> seeds) {
  auto [vor, vor_centers] = farthest_point_sampling(solver, mesh, seeds);
  // refine_voronoi_diagram(vor, vor_centers, solver, mesh, phi, D, pc, type,
  // k);
  //   build_tangent_spaces(vor, pc, vor_centers);
  return vor;
}
pair<iVd, vector<int>> intrinsic_voronoi_diagram_blended(
    const geodesic_solver &solver, const DrawableTrimesh<> &mesh,
    const point_cloud &pc, const float &alpha, const vector<int> &seeds) {

  auto [vor, vor_centers] =
      farthest_point_sampling_blended(solver, mesh, seeds, (double)alpha);
  // refine_voronoi_diagram_blended(vor, vor_centers, solver, mesh, pc, alpha);
  // build_tangent_spaces(vor, pc, vor_centers);
  return {vor, vor_centers};
}
int closest_point_to_centroid(const vector<vec2d> &p_nbr,
                              const vector<int> &nbr, const vec2d &c) {
  auto d = DBL_MAX;
  auto result = -1;
  for (auto i = 0; i < p_nbr.size(); ++i) {
    auto curr_d = (c - p_nbr[i]).norm();
    if (curr_d < d) {
      d = curr_d;
      result = nbr[i];
    }
  }

  return result;
}
void move_voronoi_centers(iVd &vor, vector<int> &vor_centers,
                          const geodesic_solver &solver,
                          const DrawableTrimesh<> &mesh,
                          const vector<double> &phi, const MatrixXd &D,
                          const point_cloud &pc, const int type) {
  for (auto i = 0; i < vor_centers.size(); ++i) {

    auto [p_nbr, vids] =
        project_points(vor_centers[i], pc, vor.voronoi_regions[i]);
    auto new_center = closest_point_to_centroid(p_nbr, vids, centroid(p_nbr));
    if (new_center == -1)
      continue;
    vor_centers[i] = new_center;
  }

  std::tie(vor, vor_centers) =
      intrinsic_voronoi_diagram(pc.solver, mesh, phi, D, pc, type, vor_centers);
}
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//                                 Gradient
vec3d polar_to_cartesian(const point_cloud &pc, int vid, const double x,
                         const double y) {
  vec3d g = vec3d{0, 0, 0};
  vec2f sol = vec2f{(float)x, (float)y};
  double phi = atan2(y, x);

  float mag = sol.norm();
  vec3d e = pc.patches[vid].xu;
  g = rot_vec(e, pc.patches[vid].Monge_normal, phi);
  g.normalize();
  g *= mag;

  return g;
}
vector<vec3d> compute_grad_MP(const point_cloud &pc, const VectorXd &f) {
  auto V = (int)f.rows();
  vector<vec3d> g(V);

  VectorXd Grad = pc.G * f;

  for (int i = 0; i < V; ++i) {
    g[i] = polar_to_cartesian(pc, i, Grad(i), Grad(V + i));
  }
  return g;
}
vector<vec3d> compute_grad_MP(const point_cloud &pc,
                              const vector<double> &field) {
  auto f = wrapper(field);

  return compute_grad_MP(pc, f);
}
vector<vec3d> compute_grad(const point_cloud &pc, const VectorXd &f) {
  auto V = (int)f.rows();
  vector<vec3d> g(V);

  VectorXd Grad = pc.G * f;

  for (int i = 0; i < V; ++i) {
    g[i] = vec3d{Grad(i), Grad(V + i), Grad(2 * V + i)};
  }
  return g;
}
vector<vec3d> compute_grad(const point_cloud &pc, const vector<double> &field) {
  auto f = wrapper(field);

  return compute_grad(pc, f);
}
vector<vec3d> compute_grad_cino(const SparseMatrix<double> &G,
                                const vector<double> &field) {
  auto V = (int)field.size();
  vector<vec3d> g(V);
  auto f = wrapper(field);
  VectorXd Grad = G * f;
  for (int i = 0; i < V; ++i) {
    g[i] = vec3d{Grad(3 * i), Grad(3 * i + 1), Grad(3 * i + 2)};
  }
  return g;
}
VectorXd local_field(const point_cloud &pc, const int vid, const VectorXd &f) {
  auto &nbr = pc.nbrs[vid];
  VectorXd result(nbr.size());
  for (auto i = 0; i < nbr.size(); ++i)
    result(i) = f[nbr[i]];

  return result;
}
vector<vec3d> compute_grad_slow(const point_cloud &pc, const VectorXd &f) {
  auto V = (int)f.rows();
  vector<vec3d> grad(V);

  for (int i = 0; i < V; ++i) {
    auto lf = local_field(pc, i, f);
    auto &p = pc.patches[i];
    VectorXd q = p.C * lf;
    auto fu = evaluate_quadric_du(q, vec2d{0, 0});
    auto fv = evaluate_quadric_dv(q, vec2d{0, 0});
    Vector2d partials;
    partials << fu, fv;
    auto g = first_fund(p.quadric, vec2d{0, 0});
    auto xu = evaluate_quadric_du(p.quadric, vec2d{0, 0});
    auto xv = evaluate_quadric_dv(p.quadric, vec2d{0, 0});
    Vector2d components = g.inverse() * partials;
    grad[i] = components(0) * xu + components(1) * xv;
  }
  return grad;
}
vector<vec3d> compute_grad_slow(const point_cloud &pc,
                                const vector<double> &field) {
  auto f = wrapper(field);

  return compute_grad_slow(pc, f);
}
vector<double> compute_laplacian(const Eigen::SparseMatrix<double> &L,
                                 const Eigen::VectorXd &field) {
  vector<double> laplacian(field.size());
  Eigen::VectorXd Lap = L * field;
  for (auto i = 0; i < Lap.size(); ++i) {
    laplacian[i] = Lap(i);
  }
  return laplacian;
}
vector<double> compute_laplacian(const Eigen::SparseMatrix<double> &L,
                                 const vector<double> &field) {

  return compute_laplacian(L, wrapper(field));
}
vec2d parameter_correction(const vec2d &p, const MatrixXd &Q,
                           const vec3d &res) {
  auto xu = evaluate_quadric_du(Q, p);
  auto xv = evaluate_quadric_dv(Q, p);
  Matrix2d g;
  Vector2d b;
  g << xu.dot(xu), xu.dot(xv), xu.dot(xv), xv.dot(xv);
  b << res.dot(xu), res.dot(xv);
  Vector2d v = g.inverse() * b;
  return p + vec2d{v(0), v(1)};
}
void parameter_correction(vector<pair<vec2d, int>> &p_nbr, const MatrixXd &Q,
                          const MatrixXd &res) {
  for (auto i = 1; i < p_nbr.size(); ++i) {
    Vector3d curr_res = res.row(i);
    p_nbr[i].first = parameter_correction(
        p_nbr[i].first, Q, vec3d{curr_res(0), curr_res(1), curr_res(2)});
  }
}
void update_L(MatrixXd &L, const vector<pair<vec2d, int>> &p_nbr) {
  for (auto i = 1; i < p_nbr.size(); ++i) {
    auto par_pos = p_nbr[i].first;
    L(i, 0) = 1;
    L(i, 1) = par_pos[0];
    L(i, 2) = par_pos[1];
    L(i, 3) = 1.f / 2 * pow(par_pos[0], 2);
    L(i, 4) = par_pos[0] * par_pos[1];
    L(i, 5) = 1.f / 2 * pow(par_pos[1], 2);
  }
}
std::tuple<MatrixXd, MatrixXd, MatrixXd, vec2d>
IRLS(MatrixXd &L, vector<pair<vec2d, int>> &p_nbr, const MatrixXd &d,
     const double &p = 1, const double epsilon = 1e-6) {
  auto k = (int)d.rows();
  MatrixXd W(k, k);

  W.setIdentity();
  ColPivHouseholderQR<MatrixXd> dec(L);
  MatrixXd x0(6, 3);
  for (auto i = 0; i < 3; ++i)
    x0.col(i) = dec.solve(d.col(i));

  MatrixXd res = d - L * x0;
  auto epsilon_w = compute_epsilon(d);
  auto it = 0;
  auto max_err = max_residual(res);
  vec2d weights_range;
  while (max_err > epsilon && it < 50) {

    weights_range = update_weights(W, epsilon_w, res, p);
    MatrixXd A = W * L;
    ColPivHouseholderQR<MatrixXd> dec(A);
    MatrixXd xk(6, 3);
    for (auto i = 0; i < 3; ++i)
      xk.col(i) = dec.solve(W * res.col(i));

    x0 = x0 + xk;
    res = d - L * x0;

    // parameter_correction(p_nbr, x0, res);
    // update_L(L, p_nbr);
    // res = d - L * x0;

    auto curr_err = max_residual(res);
    if (std::abs(curr_err - max_err) < 1e-8)
      break;
    max_err = curr_err;
    ++it;
  }
  return {x0, W, res, weights_range};
}
vector<double> diagonal_matrix_as_vector(const MatrixXd &W) {
  int k = W.rows();
  auto result = vector<double>(k);
  for (auto i = 0; i < k; ++i)
    result[i] = W(i, i);

  return result;
}
void clean_convex_hull(vector<vec2d> &ch) {
  auto new_ch = vector<vec2d>{};
  auto s = ch.size();
  auto perimeter = 0.0;
  for (auto i = 0; i < s; ++i) {
    auto v0 = ch[(s + i - 1) % s] - ch[i];
    auto v1 = ch[(i + 1) % s] - ch[i];
    perimeter += v1.norm();
  }

  auto target_len = 0.25 * perimeter / s;
  for (auto i = 0; i < ch.size(); ++i) {
    auto v0 = ch[(s + i - 1) % s] - ch[i];
    auto v1 = ch[(i + 1) % s] - ch[i];
    if (v1.norm() < target_len)
      continue;
    if (M_PI - v0.angle_rad(v1) > 2 * 1e-2)
      new_ch.push_back(ch[i]);
  }
  ch = new_ch;
}

MatrixXd Moore_Penrose_inverse(const MatrixXd &C,
                               const double &epsilon = 1e-6) {

  JacobiSVD<MatrixXd> svd(C, ComputeFullU | ComputeFullV);

  double tol = epsilon * std::max(C.cols(), C.rows()) *
               svd.singularValues().array().abs()(0);
  return svd.matrixV() *
         (svd.singularValues().array().abs() > tol)
             .select(svd.singularValues().array().inverse(), 0)
             .matrix()
             .asDiagonal() *
         svd.matrixU().adjoint();
}
void fill_riemannian_gradient_entries(vector<Triplet<double>> &entries,
                                      const vector<int> &nbr, const Matrix2d &g,
                                      const double &det, const VectorXd &c1,
                                      const VectorXd &c2, const vec3d &xu,
                                      const vec3d &xv, const int n) {
  int vid = nbr[0];
  int s = (int)nbr.size();
  typedef Triplet<double> T;
  auto g_nabla_u = 1 / det * (g(1, 1) * xu - g(0, 1) * xv);
  auto g_nabla_v = 1 / det * (g(0, 0) * xv - g(0, 1) * xu);
  for (int i = 0; i < s; ++i) {
    auto entry = nbr[i];
    entries.push_back(
        T(vid, entry, (g_nabla_u[0] * c1(i) + g_nabla_v[0] * c2(i))));
    entries.push_back(
        T(n + vid, entry, (g_nabla_u[1] * c1(i) + g_nabla_v[1] * c2(i))));
    entries.push_back(
        T(2 * n + vid, entry, (g_nabla_u[2] * c1(i) + g_nabla_v[2] * c2(i))));
  }
}
void laplacian_entries(vector<Eigen::Triplet<double>> &entries,
                       const Eigen::Matrix2d &g, const float &det,
                       const vector<int> &ring, const Eigen::MatrixXd &C,
                       const vec3d &xu, const vec3d &xv, const vec3d &xuu,
                       const vec3d &xuv, const vec3d &xvv) {

  typedef Eigen::Triplet<double> T;
  auto vid = ring[0];
  Eigen::MatrixXd Coeff(5, ring.size());
  Coeff.row(0) = C.row(1);
  Coeff.row(1) = C.row(2);
  Coeff.row(2) = C.row(3);
  Coeff.row(3) = C.row(4);
  Coeff.row(4) = C.row(5);
  auto g_deltau =
      -(g(0, 0) * (g(1, 1) * xu.dot(xvv) - g(0, 1) * xv.dot(xvv)) +
        2 * g(0, 1) * (g(0, 1) * xv.dot(xuv) - g(1, 1) * xu.dot(xuv)) +
        g(1, 1) * (g(1, 1) * xu.dot(xuu) - g(0, 1) * xv.dot(xuu))) /
      pow(det, 2);
  auto g_deltav =
      -(g(0, 0) * (g(0, 0) * xv.dot(xvv) - g(0, 1) * xu.dot(xvv)) +
        2 * g(0, 1) * (g(0, 1) * xu.dot(xuv) - g(0, 0) * xv.dot(xuv)) +
        g(1, 1) * (g(0, 0) * xv.dot(xuu) - g(0, 1) * xu.dot(xuu))) /
      pow(det, 2);
  auto g_deltauu = g(1, 1) / det;
  auto g_deltauv = -2 * g(0, 1) / det;
  auto g_deltavv = g(0, 0) / det;
  Eigen::VectorXd w(5);
  w << g_deltau, g_deltav, g_deltauu, g_deltauv, g_deltavv;
  Eigen::VectorXd b = w.transpose() * Coeff;
  for (int i = 0; i < b.rows(); ++i) {
    int entry = ring[i];
    entries.push_back(T(vid, entry, -b(i)));
  }
}
std::tuple<MatrixXd, MatrixXd>
matrices_primitves_fitting(const iVd &vor, const point_cloud &pc,
                           const vector<int> &centers, const int entry,
                           const int type_of_primitives) {
  auto &par_nbr = vor.parametric_nbr[entry];
  auto s = par_nbr.size();
  MatrixXd L;
  MatrixXd d;
  switch (type_of_primitives) {
  case bilinear_patch: {
    L.resize(s, 4);
    d.resize(s, 3);
    L.setZero();
    for (auto i = 0; i < s; ++i) {
      auto curr_pos = pc.positions[par_nbr[i].second];
      auto par_pos = par_nbr[i].first;
      for (auto h = 0; h < 3; ++h)
        d(i, h) = curr_pos[h];

      L(i, 0) = 1;
      L(i, 1) = par_pos[0];
      L(i, 2) = par_pos[1];
      L(i, 3) = par_pos[0] * par_pos[1];
    }
  } break;

  case parabolic_cylinder: {
    L.resize(3 * s, 6);
    d.resize(3 * s, 1);
    L.setZero();
    for (auto i = 0; i < s; ++i) {
      auto curr_pos = pc.positions[par_nbr[i].second];
      auto par_pos = par_nbr[i].first;

      for (auto h = 0; h < 3; ++h)
        d(3 * i + h) = curr_pos[h];

      L(3 * i, 0) = 1;
      L(3 * i, 1) = par_pos[1];
      L(3 * i + 1, 2) = 1;
      L(3 * i + 1, 3) = par_pos[0];
      L(3 * i + 2, 4) = 1;
      L(3 * i + 2, 5) = pow(par_pos[1], 2);
    }
  } break;

  case elliptic_paraboloid: {
    L.resize(3 * s, 6);
    d.resize(3 * s, 1);
    L.setZero();
    for (auto i = 0; i < s; ++i) {
      auto curr_pos = pc.positions[par_nbr[i].second];
      auto par_pos = par_nbr[i].first;

      for (auto h = 0; h < 3; ++h)
        d(3 * i + h) = curr_pos[h];

      L(3 * i, 0) = 1;
      L(3 * i, 1) = par_pos[0] * cos(par_pos[1]);
      L(3 * i + 1, 2) = 1;
      L(3 * i + 1, 3) = par_pos[0] * sin(par_pos[1]);
      L(3 * i + 2, 4) = 1;
      L(3 * i + 2, 5) = pow(par_pos[1], 2);
    }
  } break;

  case hyperbolic_paraboloid: {
    L.resize(3 * s, 8);
    d.resize(3 * s, 1);
    L.setZero();
    for (auto i = 0; i < s; ++i) {
      auto curr_pos = pc.positions[par_nbr[i].second];
      auto par_pos = par_nbr[i].first;

      for (auto h = 0; h < 3; ++h)
        d(3 * i + h) = curr_pos[h];

      L(3 * i, 0) = 1;
      L(3 * i, 1) = par_pos[0];
      L(3 * i, 2) = par_pos[1];
      L(3 * i + 1, 3) = 1;
      L(3 * i + 1, 4) = par_pos[1];
      L(3 * i + 2, 5) = 1;
      L(3 * i + 2, 6) = par_pos[0] * par_pos[1];
      L(3 * i + 2, 7) = pow(par_pos[0], 2);
    }

  } break;
  }

  return {L, d};
}
patch fittin_primitives(const iVd &vor, const point_cloud &pc,
                        const vector<int> &centers, const int entry,
                        const int type_of_primitives) {
  auto [L, d] =
      matrices_primitves_fitting(vor, pc, centers, entry, type_of_primitives);
  patch result;
  if (type_of_primitives == bilinear_patch) {
    auto [x0, W, res, weights_range] = IRLS_Claerbout(L, d);
    result.quadric = x0;
    result.weights = diagonal_matrix_as_vector(W);
    result.res = res;
    result.residual = max_residual(res);
    result.weights_range = weights_range;
  } else {
    ColPivHouseholderQR<MatrixXd> dec(L);
    MatrixXd x = dec.solve(d);
    MatrixXd res = L * x - d;

    result.quadric = x;
    result.res = res;
    result.residual = max_residual(res);
  }

  return result;
}

patch bilinear_patch_fitting(point_cloud &pc, const int vid,
                             const vector<int> &nbr) {
  auto s = nbr.size();
  auto basis = pc.basis[vid];
  auto u = basis[1];
  auto n = basis[3];
  auto pos = pc.positions[vid];
  MatrixXd L(s, 4);
  MatrixXd d(s, 3);
  patch result;
  auto &p_nbr = result.parametric_nbr;
  p_nbr.resize(s);
  double u_max, v_max, u_min, v_min, w_max, w_min;
  u_max = v_max = w_max = __DBL_MIN__;
  u_min = v_min = w_min = __DBL_MAX__;

  for (auto i = 0; i < s; ++i) {
    auto curr_pos = pc.positions[nbr[i]];
    auto len = (curr_pos - pos).norm();
    auto par_pos = vec2d{0, 0};
    if (len != 0) {
      auto p = project_vec(curr_pos - pos, n);
      auto theta = u.angle_rad(p);
      if ((u.cross(p).dot(n)) < 0)
        theta = 2 * M_PI - theta;
      par_pos = vec2d{len * std::cos(theta), len * std::sin(theta)};
    }

    p_nbr[i] = make_pair(par_pos, nbr[i]);
    u_max = std::max(u_max, par_pos.x());
    u_min = std::min(u_min, par_pos.x());
    v_max = std::max(v_max, par_pos.y());
    v_min = std::min(v_min, par_pos.y());
    for (auto h = 0; h < 3; ++h)
      d(i, h) = curr_pos[h];

    L(i, 0) = 1;
    L(i, 1) = par_pos[0];
    L(i, 2) = par_pos[1];
    L(i, 3) = par_pos[0] * par_pos[1];
  }

  auto [x0, W, res, weights_range] = IRLS_Claerbout(L, d);

  result.quadric = x0;
  result.weights = diagonal_matrix_as_vector(W);
  result.res = res;
  result.residual = max_residual(res);
  result.weights_range = weights_range;
  result.domain_range = {vec2d{u_min, u_max}, vec2d{v_min, v_max}};
  pc.residuals[vid].clear();
  // for (auto j = 0; j < s; ++j) {
  //   auto vid = nbr[j];
  //   pc.residuals[vid].push_back(result.weights[j]);
  // }
  return result;
}
double primitive_residual(const MatrixXd &Q,
                          const vector<pair<vec2d, int>> &nbr,
                          const vector<vec3d> &positions, const int type) {
  auto res = DBL_MIN;
  for (auto i = 0; i < nbr.size(); ++i) {
    auto pos_on_quad = evaluate_primitive(Q, nbr[i].first, type);
    auto pos = positions[nbr[i].second];
    res = max(res, (pos_on_quad - pos).norm());
  }

  return res;
}
// patch parabolic_cylinder_fitting(point_cloud &pc, const vector<vec3d> &e,
//                                  const vector<int> &nbr) {

//   auto s = nbr.size();
//   auto origin = e[0];
//   auto u = e[1];
//   auto v = e[2];
//   auto n = e[3];
//   patch result;
//   auto &p_nbr = result.parametric_nbr;
//   p_nbr.resize(s);
//   double u_max, v_max, u_min, v_min, w_max, w_min;
//   u_max = v_max = w_max = __DBL_MIN__;
//   u_min = v_min = w_min = __DBL_MAX__;
//   VectorXd d(s);
//   MatrixXd L(s, 6);
//   for (auto i = 0; i < s; ++i) {
//     auto curr_pos = pc.positions[nbr[i]] - origin;
//     auto par_pos = vec2d{0, 0};
//     par_pos.x() = u.dot(curr_pos);
//     par_pos.y() = v.dot(curr_pos);
//     d(i) = n.dot(curr_pos);
//     p_nbr[i] = make_pair(par_pos, nbr[i]);
//     u_max = std::max(u_max, par_pos.x());
//     u_min = std::min(u_min, par_pos.x());
//     v_max = std::max(v_max, par_pos.y());
//     v_min = std::min(v_min, par_pos.y());

//     L(i, 0) = 1;
//     L(i, 1) = par_pos[0];
//     L(i, 2) = par_pos[1];
//     L(i, 3) = pow(par_pos[0], 2);
//     L(i, 4) = 2 * par_pos[0] * par_pos[1];
//     L(i, 5) = pow(par_pos[1], 2);
//   }

//   ColPivHouseholderQR<MatrixXd> dec(L);
//   VectorXd x = dec.solve(d);
//   MatrixXd res = L * x - d;
//   result.quadric = x;
//   result.residual = max_residual(res);
//   result.domain_range = {vec2d{u_min, u_max}, vec2d{v_min, v_max}};
//   result.CH = GrahamScan(p_nbr);
//   result.e = e;

//   // for (auto j = 0; j < s; ++j) {
//   //   auto vid = nbr[j];
//   //   pc.residuals[vid].push_back(result.weights[j]);
//   // }
//   return result;
// }
patch parabolic_cylinder_fitting(point_cloud &pc, const vector<vec3d> &e,
                                 const vector<int> &nbr) {

  auto s = nbr.size();
  auto origin = e[0];
  auto u = e[1];
  auto v = e[2];
  auto n = e[3];
  patch result;
  auto &p_nbr = result.parametric_nbr;
  p_nbr.resize(s);
  double u_max, v_max, u_min, v_min, w_max, w_min;
  u_max = v_max = w_max = __DBL_MIN__;
  u_min = v_min = w_min = __DBL_MAX__;
  VectorXd d(s);
  MatrixXd L(s, 3);
  for (auto i = 0; i < s; ++i) {
    auto curr_pos = pc.positions[nbr[i]] - origin;
    auto par_pos = vec2d{0, 0};
    par_pos.x() = u.dot(curr_pos);
    par_pos.y() = v.dot(curr_pos);
    d(i) = n.dot(curr_pos);
    p_nbr[i] = make_pair(par_pos, nbr[i]);
    u_max = std::max(u_max, par_pos.x());
    u_min = std::min(u_min, par_pos.x());
    v_max = std::max(v_max, par_pos.y());
    v_min = std::min(v_min, par_pos.y());

    L(i, 0) = 1;
    L(i, 1) = par_pos[1];
    L(i, 2) = pow(par_pos[1], 2);
  }

  ColPivHouseholderQR<MatrixXd> dec(L);
  VectorXd x = dec.solve(d);
  // auto [x, W, res] = IRLS_Claerbout_1d(L, d);
  result.Monge_quadric = x;
  result.domain_range = {vec2d{u_min, u_max}, vec2d{v_min, v_max}};
  result.CH = GrahamScan(p_nbr);
  result.e = e;
  // for (auto j = 0; j < s; ++j) {
  //   auto vid = nbr[j];
  //   pc.residuals[vid].push_back(result.weights[j]);
  // }
  return result;
}

patch elliptic_paraboloid_fitting(point_cloud &pc, const int vid,
                                  const vector<int> &nbr) {
  auto s = nbr.size();
  auto basis = pc.basis[vid];
  auto u = basis[1];
  auto n = basis[3];
  auto pos = pc.positions[vid];
  MatrixXd L(3 * s, 6);
  MatrixXd d(3 * s, 1);
  L.setZero();
  patch result;
  auto &p_nbr = result.parametric_nbr;
  p_nbr.resize(s);
  double u_max, v_max, u_min, v_min, w_max, w_min;
  u_max = v_max = w_max = __DBL_MIN__;
  u_min = v_min = w_min = __DBL_MAX__;

  for (auto i = 0; i < s; ++i) {
    auto curr_pos = pc.positions[nbr[i]];
    auto len = (curr_pos - pos).norm();
    auto par_pos = vec2d{0, 0};
    if (len != 0) {
      auto p = project_vec(curr_pos - pos, n);
      auto theta = u.angle_rad(p);
      if ((u.cross(p).dot(n)) < 0)
        theta = 2 * M_PI - theta;
      par_pos = vec2d{len * std::cos(theta), len * std::sin(theta)};
    }
    p_nbr[i] = make_pair(par_pos, nbr[i]);
    u_max = std::max(u_max, par_pos.x());
    u_min = std::min(u_min, par_pos.x());
    v_max = std::max(v_max, par_pos.y());
    v_min = std::min(v_min, par_pos.y());
    for (auto h = 0; h < 3; ++h)
      d(3 * i + h) = curr_pos[h];

    L(3 * i, 0) = 1;
    L(3 * i, 1) = par_pos[0] * cos(par_pos[1]);
    L(3 * i + 1, 2) = 1;
    L(3 * i + 1, 3) = par_pos[0] * sin(par_pos[1]);
    L(3 * i + 2, 4) = 1;
    L(3 * i + 2, 5) = pow(par_pos[1], 2);
  }

  ColPivHouseholderQR<MatrixXd> dec(L);
  VectorXd x = dec.solve(d);
  VectorXd res = L * x - d;

  result.quadric = x;
  result.residual = primitive_residual(x, result.parametric_nbr, pc.positions,
                                       elliptic_paraboloid);
  result.domain_range = {vec2d{u_min, u_max}, vec2d{v_min, v_max}};
  pc.residuals[vid].clear();
  // for (auto j = 0; j < s; ++j) {
  //   auto vid = nbr[j];
  //   pc.residuals[vid].push_back(result.weights[j]);
  // }
  return result;
}
patch hyperbolic_paraboloid_fitting(point_cloud &pc, const int vid,
                                    const vector<int> &nbr) {
  auto s = nbr.size();
  auto basis = pc.basis[vid];
  auto u = basis[1];
  auto n = basis[3];
  auto pos = pc.positions[vid];
  MatrixXd L(3 * s, 8);
  MatrixXd d(3 * s, 1);
  L.setZero();
  patch result;
  auto &p_nbr = result.parametric_nbr;
  p_nbr.resize(s);
  double u_max, v_max, u_min, v_min, w_max, w_min;
  u_max = v_max = w_max = __DBL_MIN__;
  u_min = v_min = w_min = __DBL_MAX__;

  for (auto i = 0; i < s; ++i) {
    auto curr_pos = pc.positions[nbr[i]];
    auto len = (curr_pos - pos).norm();
    auto par_pos = vec2d{0, 0};
    if (len != 0) {
      auto p = project_vec(curr_pos - pos, n);
      auto theta = u.angle_rad(p);
      if ((u.cross(p).dot(n)) < 0)
        theta = 2 * M_PI - theta;
      par_pos = vec2d{len * std::cos(theta), len * std::sin(theta)};
    }
    p_nbr[i] = make_pair(par_pos, nbr[i]);
    u_max = std::max(u_max, par_pos.x());
    u_min = std::min(u_min, par_pos.x());
    v_max = std::max(v_max, par_pos.y());
    v_min = std::min(v_min, par_pos.y());
    for (auto h = 0; h < 3; ++h)
      d(3 * i + h) = curr_pos[h];

    L(3 * i, 0) = 1;
    L(3 * i, 1) = par_pos[0];
    L(3 * i, 2) = par_pos[1];
    L(3 * i + 1, 3) = 1;
    L(3 * i + 1, 4) = par_pos[1];
    L(3 * i + 2, 5) = 1;
    L(3 * i + 2, 6) = par_pos[0] * par_pos[1];
    L(3 * i + 2, 7) = pow(par_pos[0], 2);
  }

  ColPivHouseholderQR<MatrixXd> dec(L);
  VectorXd x = dec.solve(d);
  VectorXd res = L * x - d;

  result.quadric = x;
  result.residual = primitive_residual(x, result.parametric_nbr, pc.positions,
                                       hyperbolic_paraboloid);
  result.domain_range = {vec2d{u_min, u_max}, vec2d{v_min, v_max}};
  pc.residuals[vid].clear();
  // for (auto j = 0; j < s; ++j) {
  //   auto vid = nbr[j];
  //   pc.residuals[vid].push_back(result.weights[j]);
  // }
  return result;
}
double residual_of_quadric_fitting(const vector<vec3d> &positions,
                                   const vector<int> &nbr,
                                   const MatrixXd &quadric) {
  auto res = 0.0;
  auto s = nbr.size();
  for (auto i = 0; i < s; ++i) {
    auto pos = positions[nbr[i]];
    Vector4d p;
    p << pos.x(), pos.y(), pos.z(), 1;
    double curr_res = p.transpose() * quadric * p;
    res += pow(curr_res, 2);
  }

  return res / s;
}
vec3d signature(const VectorXd &eigs) {
  auto sign = vec3d{0, 0, 0};
  for (auto i = 0; i < eigs.rows(); ++i) {
    auto curr_eig = eigs(i);
    if (eigs(i) > 1e-8)
      sign[0] += 1;
    else if (eigs(i) < -1e-8)
      sign[1] += 1;
    else
      sign[2] += 1;
  }
  return sign;
}
int type_of_quadric(const MatrixXd &A) {
  auto B = A.block(0, 0, 3, 3);
  ColPivHouseholderQR<MatrixXd> decA(A);
  ColPivHouseholderQR<MatrixXd> decB(B);
  SelfAdjointEigenSolver<MatrixXd> eigB(B);
  auto rkA = decA.rank();
  auto rkB = decB.rank();
  auto detA = A.determinant();
  auto detB = B.determinant();
  auto signB = signature(eigB.eigenvalues());
  if (rkB == 3 && rkA == 4) {
    if ((signB[0] == 3 || signB[1] == 3) && detA < 0)
      return ellipsoid;
    if (((signB[0] == 2 && signB[1] == 1) ||
         (signB[0] == 1 && signB[1] == 2)) &&
        detA > 0)
      return hyperboloid;

  } else if (rkB == 2 && rkA == 4) {
    if (((signB[0] == 2 && signB[2] == 1) ||
         (signB[1] == 2 && signB[2] == 1)) &&
        detA < 0)
      return elliptic_paraboloid;
    if (signB[0] == 1 && signB[1] == 1 && signB[2] == 1 && detA > 0)
      return hyperbolic_paraboloid;

  } else if (rkB == 2 && rkA == 3) {
    if (((signB[0] == 2 && signB[2] == 1) || (signB[1] == 2 && signB[2] == 1)))
      return elliptic_cylinder;
    if (signB[0] == 1 && signB[1] == 1 && signB[2] == 1)
      return hyperbolic_cylinder;
  } else if (rkB == 1 && rkA == 3)
    return parabolic_cylinder;
  else if (rkB == 1 && rkA == 1)
    return bilinear_patch;

  return no_good;
}
patch quadric_fitting(point_cloud &pc, const vector<int> &nbr) {
  auto s = nbr.size();
  patch result;
  MatrixXd A(4 * s, 10);
  A.setZero();
  VectorXd d(4 * s);
  d.setZero();
  for (auto i = 0; i < s; ++i) {
    auto pos = pc.positions[nbr[i]];
    auto n = pc.basis[nbr[i]].back();
    A(i, 0) = pow(pos.x(), 2);
    A(i, 1) = pow(pos.y(), 2);
    A(i, 2) = pow(pos.z(), 2);
    A(i, 3) = 2 * pos.x() * pos.y();
    A(i, 4) = 2 * pos.x() * pos.z();
    A(i, 5) = 2 * pos.y() * pos.z();
    A(i, 6) = 2 * pos.x();
    A(i, 7) = 2 * pos.y();
    A(i, 8) = 2 * pos.z();
    A(i, 9) = 1;

    A(3 * i + s, 0) = 2 * pos.x();
    A(3 * i + s, 3) = 2 * pos.y();
    A(3 * i + s, 4) = 2 * pos.z();
    A(3 * i + s, 6) = 2;

    A(3 * i + s + 1, 1) = 2 * pos.y();
    A(3 * i + s + 1, 3) = 2 * pos.x();
    A(3 * i + s + 1, 5) = 2 * pos.z();
    A(3 * i + s + 1, 7) = 2;

    A(3 * i + s + 2, 2) = 2 * pos.z();
    A(3 * i + s + 2, 4) = 2 * pos.x();
    A(3 * i + s + 2, 5) = 2 * pos.y();
    A(3 * i + s + 2, 8) = 2;

    d(3 * i + s) = n.x();
    d(3 * i + s + 1) = n.y();
    d(3 * i + s + 2) = n.z();
  }

  ColPivHouseholderQR<MatrixXd> dec(A);
  VectorXd x = dec.solve(d);
  MatrixXd quadric(4, 4);
  quadric << x(0), x(3), x(4), x(6), x(3), x(1), x(5), x(7), x(4), x(5), x(2),
      x(8), x(6), x(7), x(8), x(9);

  result.quadric = quadric;
  result.residual = residual_of_quadric_fitting(pc.positions, nbr, quadric);
  result.type = type_of_quadric(quadric);
  return result;
}
patch fitting_primitive(point_cloud &pc, const int vid, const vector<int> &nbr,
                        const int type) {
  patch result;
  // switch (type) {
  // case bilinear_patch: {
  //   result = bilinear_patch_fitting(pc, vid, nbr);
  // } break;
  // case parabolic_cylinder: {
  //   result = parabolic_cylinder_fitting(pc, vid, nbr);
  // } break;
  // case hyperbolic_paraboloid: {
  //   result = hyperbolic_paraboloid_fitting(pc, vid, nbr);
  // } break;
  // case elliptic_paraboloid: {
  //   result = elliptic_paraboloid_fitting(pc, vid, nbr);
  // } break;
  // }

  return result;
}

std::tuple<patch, int> best_fitting_primitive(point_cloud &pc, const int vid) {
  patch result;
  auto type = -1;
  result.residual = DBL_MAX;
  auto nbr = pc.nbrs[vid];

  // auto curr_patch =
  //     fitting_primitive(pc, vid, pc.nbrs[vid], parabolic_cylinder);
  // if (curr_patch.residual < result.residual) {
  //   result = curr_patch;
  //   type = parabolic_cylinder;
  // }
  auto curr_patch = quadric_fitting(pc, pc.nbrs[vid]);
  if (curr_patch.residual < result.residual) {
    result = curr_patch;
    type = type_of_quadric(result.quadric);
  }

  return {result, type};
}
// std::tuple<patch, int> best_fitting_primitive(point_cloud &pc, const int
// vid)
// {
//   patch result;
//   auto type = -1;
//   result.residual = DBL_MAX;
//   auto nbr = pc.nbrs[vid];
//   for (auto i = 0; i < 4; ++i) {
//     auto curr_patch = fitting_primitive(pc, vid, pc.nbrs[vid], 0.005, i);
//     if (curr_patch.residual < result.residual) {
//       result = curr_patch;
//       type = i;
//     }
//   }
//   return {result, type};
// }
// std::tuple<patch, int>
// fitting_primitive_to_voronoi_region(const iVd &vor, point_cloud &pc,
//                                     const vector<int> &voronoi_centers,
//                                     const int center) {
//   patch result;
//   auto type = -1;
//   result.residual = DBL_MAX;
//   auto entry =
//       distance(voronoi_centers.begin(),
//                find(voronoi_centers.begin(), voronoi_centers.end(), center));

//   auto nbr = vor.voronoi_regions[entry];
//   auto curr_patch = bilinear_patch_fitting(pc, center, nbr);

//   result = curr_patch;
//   type = bilinear_patch;
//   curr_patch = quadric_fitting(pc, nbr, center);
//   if (curr_patch.residual < result.residual) {
//     result = curr_patch;
//     type = type_of_quadric(result.quadric);
//   }

//   result.CH = GrahamScan(vor.parametric_nbr[entry]);
//   return {result, type};
// }

patch patch_fitting(point_cloud &pc, vector<Triplet<double>> &G_entries,
                    vector<Triplet<double>> &L_entries, const int vid,
                    const vector<int> &nbr, const bool isophotic = false) {
  auto s = nbr.size();
  auto basis = pc.basis[vid];
  auto u = basis[1];
  auto n = basis[3];
  auto pos = pc.positions[vid];
  MatrixXd L(s, 6);
  MatrixXd d(s, 3);
  patch result;
  auto &p_nbr = result.parametric_nbr;
  p_nbr.resize(s);
  double u_max, v_max, u_min, v_min, w_max, w_min;
  u_max = v_max = w_max = __DBL_MIN__;
  u_min = v_min = w_min = __DBL_MAX__;
  for (auto i = 0; i < s; ++i) {
    auto curr_pos = pc.positions[nbr[i]];
    auto len = (curr_pos - pos).norm();
    auto par_pos = vec2d{0, 0};
    if (len != 0) {
      auto p = project_vec(curr_pos - pos, n);
      auto theta = u.angle_rad(p);
      if ((u.cross(p).dot(n)) < 0)
        theta = 2 * M_PI - theta;
      par_pos = vec2d{len * std::cos(theta), len * std::sin(theta)};
    }
    p_nbr[i] = make_pair(par_pos, nbr[i]);
    u_max = std::max(u_max, par_pos.x());
    u_min = std::min(u_min, par_pos.x());
    v_max = std::max(v_max, par_pos.y());
    v_min = std::min(v_min, par_pos.y());
    for (auto h = 0; h < 3; ++h)
      d(i, h) = curr_pos[h];

    L(i, 0) = 1;
    L(i, 1) = par_pos[0];
    L(i, 2) = par_pos[1];
    L(i, 3) = 1.f / 2 * pow(par_pos[0], 2);
    L(i, 4) = par_pos[0] * par_pos[1];
    L(i, 5) = 1.f / 2 * pow(par_pos[1], 2);
  }

  // auto [x0, W, res] = IRLS_Claerbout_splitted(L, d);
  auto [x0, W, res, weights_range] = IRLS_Claerbout(L, d);
  auto Lt = Transpose<MatrixXd>(L);
  auto B = Lt * L;
  ColPivHouseholderQR<MatrixXd> dec(B);
  MatrixXd C;
  if (dec.isInvertible()) {
    MatrixXd inv = B.inverse();
    C = inv * Lt;
  } else {
    MatrixXd Bi = Moore_Penrose_inverse(B);
    C = Bi * Lt;
  }

  // auto [x0, W, res, weights_range] = IRLS(L, p_nbr, d, 1.2);

  result.quadric = x0;
  result.C = C;
  result.weights = diagonal_matrix_as_vector(W);
  result.res = res;
  result.residual = max_residual(res);
  result.weights_range = weights_range;
  result.domain_range = {vec2d{u_min, u_max}, vec2d{v_min, v_max}};
  pc.residuals[vid].clear();
  // for (auto j = 0; j < s; ++j) {
  //   auto vid = nbr[j];
  //   pc.residuals[vid].push_back(result.weights[j]);
  // }
  // Matrix2d g;
  // vector<vec3d> X;
  // X.reserve(5);

  // auto g = first_fund(result.quadric, vec2d{0, 0});
  // X.push_back(evaluate_quadric_du(x0, vec2d{0, 0}));
  // X.push_back(evaluate_quadric_dv(x0, vec2d{0, 0}));
  // X.push_back(evaluate_quadric_duu(x0, vec2d{0, 0}));
  // X.push_back(evaluate_quadric_duv(x0, vec2d{0, 0}));
  // X.push_back(evaluate_quadric_dvv(x0, vec2d{0, 0}));

  // fill_riemannian_gradient_entries(G_entries, nbr, g, g.determinant(),
  // C.row(1),
  //                                  C.row(2), X[0], X[1],
  //                                  (int)pc.positions.size());
  // laplacian_entries(L_entries, g, g.determinant(), nbr, C, X[0], X[1], X[2],
  //                   X[3], X[4]);
  auto xu = evaluate_quadric_du(x0, vec2d{0, 0});
  auto xv = evaluate_quadric_dv(x0, vec2d{0, 0});
  Matrix2d g;
  g << xu.dot(xu), xu.dot(xv), xu.dot(xv), xv.dot(xv);
  double det = g.determinant();

  fill_riemannian_gradient_entries(G_entries, nbr, g, det, C.row(1), C.row(2),
                                   xu, xv, (int)pc.positions.size());

  // result.CH = GrahamScan(result.parametric_nbr);
  return result;
}

patch patch_fitting(const DrawableTrimesh<> &m,
                    vector<Triplet<double>> &G_entries,
                    vector<Triplet<double>> &L_entries, const int vid,
                    const vector<int> &nbr, const bool isophotic = false) {
  auto s = nbr.size();
  auto n = m.vert_data(vid).normal;
  auto pos = m.vert(vid);
  auto u = project_vec(m.vert(nbr[1]) - pos, n);
  u.normalize();
  MatrixXd L(s, 6);
  MatrixXd d(s, 3);
  patch result;
  auto &p_nbr = result.parametric_nbr;
  p_nbr.resize(s);
  double u_max, v_max, u_min, v_min, w_max, w_min;
  u_max = v_max = w_max = __DBL_MIN__;
  u_min = v_min = w_min = __DBL_MAX__;
  for (auto i = 0; i < s; ++i) {
    auto curr_pos = m.vert(nbr[i]);
    auto len = (curr_pos - pos).norm();
    auto par_pos = vec2d{0, 0};
    if (len != 0) {
      auto p = project_vec(curr_pos - pos, n);
      auto theta = u.angle_rad(p);
      if ((u.cross(p).dot(n)) < 0)
        theta = 2 * M_PI - theta;
      par_pos = vec2d{len * std::cos(theta), len * std::sin(theta)};
    }
    p_nbr[i] = make_pair(par_pos, nbr[i]);
    u_max = std::max(u_max, par_pos.x());
    u_min = std::min(u_min, par_pos.x());
    v_max = std::max(v_max, par_pos.y());
    v_min = std::min(v_min, par_pos.y());
    for (auto h = 0; h < 3; ++h)
      d(i, h) = curr_pos[h];

    L(i, 0) = 1;
    L(i, 1) = par_pos[0];
    L(i, 2) = par_pos[1];
    L(i, 3) = 1.f / 2 * pow(par_pos[0], 2);
    L(i, 4) = par_pos[0] * par_pos[1];
    L(i, 5) = 1.f / 2 * pow(par_pos[1], 2);
  }

  // auto [x0, W, res] = IRLS_Claerbout_splitted(L, d);
  auto [x0, W, res, weights_range] = IRLS_Claerbout(L, d);
  auto Lt = Transpose<MatrixXd>(L);
  auto B = Lt * L;
  ColPivHouseholderQR<MatrixXd> dec(B);
  MatrixXd C;
  if (dec.isInvertible()) {
    MatrixXd inv = B.inverse();
    C = inv * Lt;
  } else {
    MatrixXd Bi = Moore_Penrose_inverse(B);
    C = Bi * Lt;
  }

  // auto [x0, W, res, weights_range] = IRLS(L, p_nbr, d, 1.2);

  result.quadric = x0;
  result.C = C;
  result.weights = diagonal_matrix_as_vector(W);
  result.res = res;
  result.residual = max_residual(res);
  result.weights_range = weights_range;
  result.domain_range = {vec2d{u_min, u_max}, vec2d{v_min, v_max}};

  auto xu = evaluate_quadric_du(x0, vec2d{0, 0});
  auto xv = evaluate_quadric_dv(x0, vec2d{0, 0});
  Matrix2d g;
  g << xu.dot(xu), xu.dot(xv), xu.dot(xv), xv.dot(xv);
  double det = g.determinant();

  fill_riemannian_gradient_entries(G_entries, nbr, g, det, C.row(1), C.row(2),
                                   xu, xv, (int)m.num_verts());

  // result.CH = GrahamScan(result.parametric_nbr);
  return result;
}
patch patch_fitting(point_cloud &pc, const int vid, const vector<int> &nbr) {
  auto s = nbr.size();
  auto basis = pc.basis[vid];
  auto u = basis[1];
  auto n = basis[3];
  auto pos = pc.positions[vid];
  MatrixXd L(s, 6);
  MatrixXd d(s, 3);
  patch result;
  auto &p_nbr = result.parametric_nbr;
  p_nbr.resize(s);
  double u_max, v_max, u_min, v_min, w_max, w_min;
  u_max = v_max = w_max = __DBL_MIN__;
  u_min = v_min = w_min = __DBL_MAX__;

  for (auto i = 0; i < s; ++i) {
    auto curr_pos = pc.positions[nbr[i]];
    auto len = (curr_pos - pos).norm();
    auto par_pos = vec2d{0, 0};
    if (len != 0) {
      auto p = project_vec(curr_pos - pos, n);
      auto theta = u.angle_rad(p);
      if ((u.cross(p).dot(n)) < 0)
        theta = 2 * M_PI - theta;
      par_pos = vec2d{len * std::cos(theta), len * std::sin(theta)};
    }

    p_nbr[i] = make_pair(par_pos, nbr[i]);
    u_max = std::max(u_max, par_pos.x());
    u_min = std::min(u_min, par_pos.x());
    v_max = std::max(v_max, par_pos.y());
    v_min = std::min(v_min, par_pos.y());
    for (auto h = 0; h < 3; ++h)
      d(i, h) = curr_pos[h];

    L(i, 0) = 1;
    L(i, 1) = par_pos[0];
    L(i, 2) = par_pos[1];
    L(i, 3) = 1.f / 2 * pow(par_pos[0], 2);
    L(i, 4) = par_pos[0] * par_pos[1];
    L(i, 5) = 1.f / 2 * pow(par_pos[1], 2);
  }

  // auto [x0, W, res] = IRLS_Claerbout_splitted(L, d);
  auto [x0, W, res, weights_range] = IRLS_Claerbout(L, d);
  auto Lt = Transpose<MatrixXd>(L);
  auto B = Lt * L;
  ColPivHouseholderQR<MatrixXd> dec(B);
  MatrixXd C;
  if (dec.isInvertible()) {
    MatrixXd inv = B.inverse();
    C = inv * Lt;
  } else {
    MatrixXd Bi = Moore_Penrose_inverse(B);
    C = Bi * Lt;
  }

  // auto [x0, W, res, weights_range] = IRLS(L, p_nbr, d, 1.2);

  result.quadric = x0;
  result.C = C;
  result.weights = diagonal_matrix_as_vector(W);
  result.res = res;
  result.residual = max_residual(res);
  result.weights_range = weights_range;
  result.domain_range = {vec2d{u_min, u_max}, vec2d{v_min, v_max}};
  pc.residuals[vid].clear();
  for (auto j = 0; j < s; ++j) {
    auto vid = nbr[j];
    pc.residuals[vid].push_back(result.weights[j]);
  }

  result.CH = GrahamScan(result.parametric_nbr);
  return result;
}
std::tuple<patch, int>
fitting_primitive_to_voronoi_region(const iVd &vor, point_cloud &pc,
                                    const vector<int> &voronoi_centers,
                                    const int center) {
  patch result;
  auto type = -1;
  result.residual = DBL_MAX;
  auto entry =
      distance(voronoi_centers.begin(),
               find(voronoi_centers.begin(), voronoi_centers.end(), center));
  for (auto i = 1; i < 2; ++i) {
    auto nbr = vor.voronoi_regions[entry];
    auto curr_patch = fitting_primitive(pc, center, nbr, i);
    if (curr_patch.residual < result.residual) {
      result = curr_patch;
      type = i;
    }
  }

  // auto nbr = vor.voronoi_regions[entry];
  // result = patch_fitting(pc, center, nbr);
  result.CH = GrahamScan(result.parametric_nbr);
  return {result, type};
}
MatrixXd rhs_Monge_patch(int s) {
  MatrixXd E(s, s + 1);
  MatrixXd X = MatrixXd::Constant(s, 1, -1);
  E.topLeftCorner(s, 1) = X;
  MatrixXd I(s, s);
  I.setIdentity();
  E.topRightCorner(s, s) = I;
  return E;
}
vec3d average_normal(const point_cloud &pc, const vector<int> &nbr,
                     const int vid) {
  auto result = vec3d{0, 0, 0};
  auto s = nbr.size();
  for (auto i = 0; i < s; ++i) {
    result += pc.basis[nbr[i]][3];
  }
  result += pc.basis[vid][3];

  return result / (s + 1);
}
std::tuple<vector<int>, vector<double>, vector<double>, vec3d>
filtered_ring_stencil(const point_cloud &pc, const int vid) {

  auto nbr = pc.nbrs[vid];
  auto N = average_normal(pc, nbr, vid);
  auto n = pc.basis[vid][3];
  auto vert = pc.positions[vid];
  auto it = remove_if(nbr.begin(), nbr.end(),
                      [&](const int curr) { return (n.dot(N) < 1e-3); });
  nbr.erase(it, nbr.end());

  auto lens = vector<double>{};
  auto thetas = vector<double>{};

  lens.resize(nbr.size());
  thetas.resize(nbr.size());

  auto e = pc.basis[vid][1];

  for (auto i = 0; i < nbr.size(); ++i) {
    auto curr_pos = pc.positions[nbr[i]];
    auto len = (curr_pos - vert).norm();
    auto p = project_vec(curr_pos - vert, n);
    auto theta = e.angle_rad(p);
    if ((e.cross(p).dot(n)) < 0)
      theta = 2 * M_PI - theta;
    lens[i] = len;
    thetas[i] = theta;
  }
  return {nbr, lens, thetas, vec3d{N.x(), N.y(), N.z()}};
}
void fill_riemannian_gradient_entries_MP(vector<Triplet<double>> &entries,
                                         const vector<int> &nbr,
                                         const VectorXd &c, const VectorXd &a0,
                                         const VectorXd &a1, const int n) {
  int vid = nbr[0];
  int s = (int)nbr.size() - 1;
  double c0_squared = pow(c[0], 2);
  double c1_squared = pow(c[1], 2);
  Matrix2d g_inv;
  double det = 1 + c0_squared + c1_squared;
  g_inv << 1 + c1_squared, -c[0] * c[1], -c[0] * c[1], 1 + c0_squared;
  g_inv /= det;
  typedef Triplet<double> T;
  for (int i = 1; i < s; ++i) {
    int entry = nbr[i];
    entries.push_back(
        T(vid, entry, (g_inv(0, 0) * a0(i) + g_inv(0, 1) * a1(i))));
    entries.push_back(
        T(n + vid, entry, (g_inv(1, 0) * a0(i) + g_inv(1, 1) * a1(i))));
  }
}

patch Monge_patch_fitting(point_cloud &pc, vector<Triplet<double>> &G_entries,
                          const int vid, const vector<int> &nbr) {
  auto V = pc.positions.size();
  auto basis = pc.basis[vid];
  auto u = basis[1];
  auto v = basis[2];
  auto n = basis[3];
  auto pos = pc.positions[vid];

  patch result;
  SparseMatrix<double> G;
  auto &p_nbr = result.parametric_nbr;

  double u_max, v_max, u_min, v_min, w_max, w_min;
  u_max = v_max = w_max = __DBL_MIN__;
  u_min = v_min = w_min = __DBL_MAX__;

  // auto [filtered_nbr, lens, thetas, N] = filtered_ring_stencil(pc, vid);
  auto s = int(nbr.size());
  p_nbr.resize(s);
  MatrixXd L(s, 6);
  VectorXd h(s);
  result.Monge_normal = n;
  pc.nbrs[vid] = nbr;
  for (auto i = 0; i < s; ++i) {

    auto curr_pos = pc.positions[nbr[i]] - pos;

    auto par_pos = vec2d{0, 0};
    par_pos.x() = curr_pos.dot(u);
    par_pos.y() = curr_pos.dot(v);
    h(i) = curr_pos.dot(n);
    p_nbr[i] = make_pair(par_pos, nbr[i]);
    u_max = std::max(u_max, par_pos.x());
    u_min = std::min(u_min, par_pos.x());
    v_max = std::max(v_max, par_pos.y());
    v_min = std::min(v_min, par_pos.y());

    L(i, 0) = 1;
    L(i, 1) = par_pos.x();
    L(i, 2) = par_pos.y();
    L(i, 3) = 1.f / 2 * pow(par_pos.x(), 2);
    L(i, 4) = par_pos.x() * par_pos.y();
    L(i, 5) = 1.f / 2 * pow(par_pos.y(), 2);
  }
  // MatrixXd Lt = Transpose<MatrixXd>(L);

  // MatrixXd A = Lt * L;
  MatrixXd E = rhs_Monge_patch(s);
  ColPivHouseholderQR<MatrixXd> dec(L);
  VectorXd c(5);
  MatrixXd a(6, s);
  // if (dec.isInvertible()) {
  //   MatrixXd inv = A.inverse();
  //   c = inv * Lt * h;
  //   a = inv * Lt * E;

  // } else {
  // MatrixXd Rhsc = Lt * h;
  // MatrixXd Rhsa = Lt * E;
  c = dec.solve(h);
  a = dec.solve(E);
  //}
  // auto [c, W, res] = IRLS_Claerbout_1d(L, h);

  result.Monge_quadric = c;
  result.xu = vec3d{1, 0, c(0)};
  result.xv = vec3d{0, 1, c(1)};
  result.xuu = vec3d{0, 0, c(2)};
  result.xuv = vec3d{0, 0, c(3)};
  result.xvv = vec3d{0, 0, c(4)};
  // fill_riemannian_gradient_entries_MP(G_entries, nbr, c, a.row(0), a.row(1),
  // V);
  result.domain_range = {vec2d{u_min, u_max}, vec2d{v_min, v_max}};
  result.CH = GrahamScan(result.parametric_nbr);
  return result;
}
patch Monge_patch_fitting(point_cloud &pc, const vector<vec3d> &e,
                          const vector<int> &nbr) {
  auto V = pc.positions.size();
  auto u = e[1];
  auto v = e[2];
  auto n = e[3];
  auto pos = e[0];

  patch result;
  SparseMatrix<double> G;
  auto &p_nbr = result.parametric_nbr;

  double u_max, v_max, u_min, v_min, w_max, w_min;
  u_max = v_max = w_max = __DBL_MIN__;
  u_min = v_min = w_min = __DBL_MAX__;

  // auto [filtered_nbr, lens, thetas, N] = filtered_ring_stencil(pc, vid);
  auto s = int(nbr.size());
  p_nbr.resize(s);
  MatrixXd L(s, 6);
  VectorXd h(s);
  result.Monge_normal = n;
  for (auto i = 0; i < s; ++i) {

    auto curr_pos = pc.positions[nbr[i]] - pos;

    auto par_pos = vec2d{0, 0};
    par_pos.x() = curr_pos.dot(u);
    par_pos.y() = curr_pos.dot(v);
    h(i) = curr_pos.dot(n);
    p_nbr[i] = make_pair(par_pos, nbr[i]);
    u_max = std::max(u_max, par_pos.x());
    u_min = std::min(u_min, par_pos.x());
    v_max = std::max(v_max, par_pos.y());
    v_min = std::min(v_min, par_pos.y());

    L(i, 0) = 1;
    L(i, 1) = par_pos.x();
    L(i, 2) = par_pos.y();
    L(i, 3) = 1.f / 2 * pow(par_pos.x(), 2);
    L(i, 4) = par_pos.x() * par_pos.y();
    L(i, 5) = 1.f / 2 * pow(par_pos.y(), 2);
  }
  // MatrixXd Lt = Transpose<MatrixXd>(L);

  // MatrixXd A = Lt * L;
  // MatrixXd E = rhs_Monge_patch(s);
  // ColPivHouseholderQR<MatrixXd> dec(L);
  // VectorXd c(6);
  // MatrixXd a(6, s);
  // if (dec.isInvertible()) {
  //   MatrixXd inv = A.inverse();
  //   c = inv * Lt * h;
  //   a = inv * Lt * E;

  // } else {
  // MatrixXd Rhsc = Lt * h;
  // MatrixXd Rhsa = Lt * E;
  // c = dec.solve(h);
  // a = dec.solve(E);
  //}
  auto [c, W, res] = IRLS_Claerbout_1d(L, h);

  result.Monge_quadric = c;
  result.xu = vec3d{1, 0, c(0)};
  result.xv = vec3d{0, 1, c(1)};
  result.xuu = vec3d{0, 0, c(2)};
  result.xuv = vec3d{0, 0, c(3)};
  result.xvv = vec3d{0, 0, c(4)};
  // fill_riemannian_gradient_entries_MP(G_entries, nbr, c, a.row(0), a.row(1),
  // V);
  result.domain_range = {vec2d{u_min, u_max}, vec2d{v_min, v_max}};
  result.CH = GrahamScan(result.parametric_nbr);
  result.e = e;
  return result;
}
patch direct_parabolic_cylinder_fitting(point_cloud &pc,
                                        const vector<int> &nbr) {
  auto e = PCA(nbr, pc.positions);
  auto p = Monge_patch_fitting(pc, e, nbr);
  VectorXd q = p.Monge_quadric;
  Matrix2d H;
  H << q(2), q(3), q(3), q(5);
  SelfAdjointEigenSolver<Matrix2d> eig(H);
  Matrix2d U = eig.eigenvectors();
  Vector2d u;
  Vector2d v;
  Vector2d lambda = eig.eigenvalues();
  // if (abs(lambda(0)) > abs(lambda(1))) {
  u = U.col(0);
  v = U.col(1);
  // } else {
  //   u1 = U.col(0);
  //   u0 = U.col(1);
  // }
  auto e_old = vec3d{1, 0, 0};
  auto n = vec3d{0, 0, 1};
  auto e0_prime = vec3d{u(0), u(1), 0};
  auto e1_prime = vec3d{v(0), v(1), 0};
  auto theta = e0_prime.angle_rad(e_old);
  if (e_old.cross(e0_prime).dot(n) < 0)
    theta *= -1;

  auto new_e = vector<vec3d>(4);
  new_e[0] = e[0];
  new_e[1] = rot_vec(e[1], e[3], theta);
  new_e[3] = e[3];
  new_e[2] = new_e[3].cross(new_e[1]);

  return parabolic_cylinder_fitting(pc, new_e, nbr);
}
vector<vec2d> update_domain_range(const vector<pair<vec2d, int>> &par_nbr) {
  double u_max, v_max, u_min, v_min;
  u_max = v_max = __DBL_MIN__;
  u_min = v_min = __DBL_MAX__;
  for (auto &par_pos : par_nbr) {
    u_max = std::max(u_max, par_pos.first.x());
    u_min = std::min(u_min, par_pos.first.x());
    v_max = std::max(v_max, par_pos.first.y());
    v_min = std::min(v_min, par_pos.first.y());
  }
  return {vec2d{u_min, u_max}, vec2d{v_min, v_max}};
}
double compute_tau(const point_cloud &pc) {
  auto tau = 0.0;
  auto P = pc.positions.size();
  for (auto i = 0; i < pc.positions.size(); ++i) {
    if (pc.patch_tagging[i].size() > 0)
      tau += pc.patches[pc.patch_tagging[i][0]].residual;
    else
      tau += pc.patches[i].residual;
  }

  tau /= P;
  return tau;
}
vector<double> subdivide_angles(const int number_of_subdivision) {
  auto thetas = vector<double>(number_of_subdivision);
  auto step = 2 * M_PI / number_of_subdivision;

  for (auto i = 0; i < number_of_subdivision; ++i) {
    thetas[i] = i * step;
  }
  return thetas;
}
vector<vec3d> local_grad(const point_cloud &pc, const int vid,
                         const vector<double> &f, const double &r) {
  auto &nbr = pc.nbrs[vid];
  auto &p = pc.patches[vid];
  auto k = nbr.size();
  VectorXd local_field(k);
  auto result = vector<vec3d>(10);
  for (auto i = 0; i < k; ++i)
    local_field(i) = f[nbr[i]];
  VectorXd fit = p.C * local_field;
  auto thetas = subdivide_angles(10);
  auto e = vec2d{1, 0};

  for (auto i = 0; i < 10; ++i) {

    auto curr_pos = vec2d{r * std::cos(thetas[i]), r * std::sin(thetas[i])};
    auto xu = evaluate_quadric_du(p.quadric, curr_pos);
    auto xv = evaluate_quadric_dv(p.quadric, curr_pos);
    auto fu = evaluate_quadric_du(fit, curr_pos);
    auto fv = evaluate_quadric_dv(fit, curr_pos);
    // auto [g, X] = isophotic_metric(p, curr_pos);
    auto g = first_fund(p.quadric, curr_pos);
    Vector2d partial;
    partial << fu, fv;
    Vector2d gradient = g.inverse() * partial;

    result[i] = xu * gradient(0) + xv * gradient(1);
  }

  return result;
}
double gaussian_curvature(const point_cloud &pc, const int &vid) {
  auto [g, Xuv] = metric_tensor(pc.patches[vid], vec2d{0, 0});
  auto xu = Xuv[0];
  auto xv = Xuv[1];
  auto xuu = Xuv[2];
  auto xvv = Xuv[3];
  auto xuv = Xuv[4];
  auto n = xu.cross(xv);
  n.normalize();
  auto L = xuu.dot(n);
  auto M = xuv.dot(n);
  auto N = xvv.dot(n);
  auto num = L * N - pow(M, 2);
  auto den = g.determinant();

  return num / den;
}
vector<double> curvature_field(const point_cloud &pc) {
  auto result = vector<double>(pc.positions.size());
  for (auto i = 0; i < pc.positions.size(); ++i)
    result[i] = gaussian_curvature(pc, i);

  return result;
}
std::tuple<vec2d, vector<vec2d>> principal_curv_and_dir(const patch &p,
                                                        const vec2d &uv) {
  auto S = shape_operator(p, uv);
  SelfAdjointEigenSolver<Matrix2d> dec(S);
  auto eivals = dec.eigenvalues();
  auto eivec = dec.eigenvectors();
  double k1 = eivals(0);
  double k2 = eivals(1);

  Vector2d K1 = eivec.col(0);
  Vector2d K2 = eivec.col(1);
  auto d1 = vec2d{K1(0), K1(1)};
  auto d2 = vec2d{K2(0), K2(1)};
  d1.normalize();
  d2.normalize();
  return {vec2d{k1, k2}, {d1, d2}};
}
vector<vec3d> local_principal_dir(const point_cloud &pc, const int vid,
                                  const double &r) {
  auto &nbr = pc.nbrs[vid];
  auto &p = pc.patches[vid];
  auto thetas = subdivide_angles(10);
  auto [k, d] = principal_curv_and_dir(p, vec2d{0, 0});
  auto result = vector<vec3d>(10);
  auto xu = evaluate_quadric_du(p.quadric, vec2d{0, 0});
  auto xv = evaluate_quadric_dv(p.quadric, vec2d{0, 0});

  for (auto i = 0; i < 10; ++i) {
    auto curr_pos = vec2d{r * std::cos(thetas[i]), r * std::sin(thetas[i])};
    std::tie(k, d) = principal_curv_and_dir(p, curr_pos);
    auto curr = d[1];

    result[i] = xu * curr.x() + xv * curr.y();
  }

  return result;
}
vector<int> parabolic_regions(const point_cloud &pc, const double &tau_m) {
  auto result = vector<int>{};
  result.reserve(pc.positions.size());
  for (auto i = 0; i < pc.positions.size(); ++i) {
    auto [k, d] = principal_curv_and_dir(pc.patches[i], vec2d{0, 0});
    auto kmax = std::max(std::abs(k.x()), std::abs(k.y()));
    auto kmin = std::min(std::abs(k.x()), std::abs(k.y()));
    auto tau = (kmax - kmin) / kmax;
    if (tau > tau_m)
      result.push_back(i);
  }

  return result;
}
int index_of_singularity(const point_cloud &pc, const int vid,
                         const vector<double> &f) {
  auto &nbr = pc.nbrs[vid];
  auto &p = pc.patches[vid];
  auto k = nbr.size();
  VectorXd local_field(k);
  for (auto i = 0; i < k; ++i)
    local_field(i) = f[nbr[i]];
  VectorXd fit = p.C * local_field;
  auto r = 1e-4;
  auto thetas = subdivide_angles(100);
  auto index = 0;
  auto total = 0.0;
  auto prev = vec2d{0, 0};
  auto first = vec2d{0, 0};
  auto curr = vec2d{0, 0};
  auto phi = 0.0;
  for (auto i = 0; i < 100; ++i) {
    auto curr_pos = vec2d{r * std::cos(thetas[i]), r * std::sin(thetas[i])};
    auto fu = evaluate_quadric_du(fit, curr_pos);
    auto fv = evaluate_quadric_dv(fit, curr_pos);
    // auto [g, X] = isophotic_metric(p, curr_pos);
    auto g = first_fund(p.quadric, curr_pos);
    Vector2d partial;
    partial << fu, fv;
    Vector2d gradient = g.inverse() * partial;
    curr = vec2d{gradient(0), gradient(1)};
    curr.normalize();

    // phi = e.angle_rad(grad);

    auto curr_angle = curr.angle_rad(prev);
    if (cross(prev, curr) < 0)
      curr_angle *= -1;
    if (i != 0)
      total += curr_angle;
    else
      first = curr;
    prev = curr;
  }
  phi = curr.angle_rad(first);
  if (cross(curr, first) < 0)
    phi *= -1;
  total += phi;

  if (std::abs(2 * M_PI - total) < 1e-2 || std::abs(2 * M_PI + total) < 1e-2)
    return 1;

  return 0;
}

int index_of_singularity(const point_cloud &pc, const int vid) {
  auto &nbr = pc.nbrs[vid];
  auto &p = pc.patches[vid];
  auto k = nbr.size();

  auto r = 1e-2;
  auto thetas = subdivide_angles(100);
  auto index = 0;
  auto total = 0.0;
  auto prev = vec2d{0, 0};
  auto first = vec2d{0, 0};
  auto curr = vec2d{0, 0};
  auto phi = 0.0;
  for (auto i = 0; i < 100; ++i) {
    auto curr_pos = vec2d{r * std::cos(thetas[i]), r * std::sin(thetas[i])};
    auto [k, d] = principal_curv_and_dir(p, curr_pos);
    curr = d[1];
    // phi = e.angle_rad(grad);

    auto curr_angle = curr.angle_rad(prev);
    if (cross(prev, curr) < 0)
      curr_angle *= -1;
    if (i != 0)
      total += curr_angle;
    else
      first = curr;
    prev = curr;
  }
  phi = curr.angle_rad(first);
  if (cross(curr, first) < 0)
    phi *= -1;
  total += phi;

  return round(total / (2 * M_PI));
}
vector<vec2i> singularities(const point_cloud &pc) {
  auto sing = vector<vec2i>{};
  sing.reserve(pc.positions.size());
  for (auto i = 0; i < pc.positions.size(); ++i) {
    auto ind = index_of_singularity(pc, i);
    if (ind != 0)
      sing.push_back(vec2i{i, ind});
  }

  return sing;
}

void patch_fitting(point_cloud &pc, const DrawableTrimesh<> &m,
                   const double &max_err, const bool isophotic) {
  // Profiler prof;
  // prof.push("Patch Fitting");
  auto k = pc.positions.size();
  pc.patches.resize(k);
  pc.patch_tagging.resize(k);
  vector<Triplet<double>> G;
  vector<Triplet<double>> L;
  for (auto i = 0; i < k; ++i) {
    auto nbr = n_ring(m, i, 2);
    pc.nbrs[i] = nbr;
    pc.patches[i] = patch_fitting(m, G, L, i, nbr);
  }
  pc.G.resize(3 * k, k);
  pc.G.setFromTriplets(G.begin(), G.end());
  pc.solver = compute_geodesic_solver(pc);
  // prof.pop();
}
void patch_fitting(point_cloud &pc, const DrawableTrimesh<> &m,
                   const vector<bool> &bad, const int goodones,
                   const double &max_err) {
  Profiler prof;
  prof.push("Patch Fitting");
  auto k = pc.positions.size();
  pc.patches.resize(k);
  pc.patch_tagging.resize(k);
  vector<Triplet<double>> G;
  vector<Triplet<double>> L;
  for (auto i = 0; i < k; ++i) {
    if (bad[i])
      continue;
    auto nbr = cleaned_k_ring(m, i, 35, bad);
    pc.nbrs[i] = nbr;

    pc.patches[i] = patch_fitting(pc, G, L, i, nbr);
    pc.patches[i].CH = GrahamScan(pc.patches[i].parametric_nbr);
  }

  pc.G.resize(3 * k, k);
  // pc.L.resize(k, k);
  // pc.L.setFromTriplets(L.begin(), L.end());

  pc.G.setFromTriplets(G.begin(), G.end());
  pc.solver = compute_geodesic_solver(pc, bad);

  // remove_outliers(pc, max_err);
  // enlarge_patches(pc, max_err);

  // // update_patch(pc);
  // trim_secondary_patches(pc, max_err);
  // kill_small_patches(pc, max_err);
  pc.tau = compute_tau(pc);
  //   for (auto i = 0; i < k; ++i) {
  //     pc.patches[i].CH = GrahamScan(pc.patches[i].parametric_nbr);
  //   }
  prof.pop();
}

bool my_matrix_eigenfunctions(const Eigen::SparseMatrix<double> &m,
                              const int nf, std::vector<double> &f,
                              std::vector<double> &f_min,
                              std::vector<double> &f_max) {
  int nc = m.cols();
  f.resize(m.cols() * nf);
  f_min.resize(nf);
  f_max.resize(nf);

  // https://github.com/yixuan/spectra/issues/149#issuecomment-1398594081
  Spectra::SparseGenMatProd<double> op(m);
  Spectra::GenEigsSolver<Spectra::SparseGenMatProd<double>> eigs(op, nf,
                                                                 2 * nf + 1);
  eigs.init();
  eigs.compute(Spectra::SortRule::LargestMagn); // largest beacuse it's inverted
  if (eigs.info() != Spectra::CompInfo::Successful) {
    std::cout << "did not manage to compute" << std::endl;
    return false;
  }
  // assemble output data
  auto basis_func = eigs.eigenvectors();
  for (int fid = 0; fid < basis_func.cols(); ++fid) {
    double min = max_double;
    double max = min_double;
    int off = nf - 1 - fid; // reverse the order
    for (int i = 0; i < nc; ++i) {
      auto coeff = basis_func(i, fid);
      if (coeff.imag() != 0) {
        std::cout << "Complex eigenvalues" << std::endl;
        return false;
      } // fail if there are complex coefficients
      f.at(off * nc + i) = coeff.real();
      max = std::max(max, std::fabs(coeff.real()));
      min = std::min(min, -max);
    }
    f_min.at(off) = min;
    f_max.at(off) = max;
  }

  return true;
}
vector<double> uniform_sampling(const double &min, const double &max,
                                const int n) {
  auto result = vector<double>(n);
  auto step = (max - min) / n;
  for (auto i = 0; i < n; ++i) {
    result[i] = min + (i + 1) * step;
  }

  return result;
}
vector<pair<double, int>> circle_descriptor(const DrawableTrimesh<> &m,
                                            const vector<double> &distances,
                                            const int center) {

  auto n = 16;
  auto rhos = uniform_sampling(0.0, 1, n);
  auto result = vector<pair<double, int>>(n);
  for (auto i = 0; i < n; ++i) {
    auto c = create_circle(m, center, rhos[i], distances);
    result[i] = make_pair(normalize_circle_length(m, c), (int)c.isoline.size());
  }

  return result;
}
void normalize_field(vector<double> &f, const double &M) {
  for (auto i = 0; i < f.size(); ++i)
    f[i] /= M;
}
vector<vector<pair<double, int>>>
compute_descriptors(const DrawableTrimesh<> &m, const vector<int> &centers,
                    const vector<vector<double>> &dist) {
  auto result = vector<vector<pair<double, int>>>(centers.size());
  for (auto i = 0; i < centers.size(); ++i) {
    result[i] = circle_descriptor(m, dist[i], centers[i]);
  }

  return result;
}
double p_descr(const vector<pair<double, int>> &Dx,
               const vector<pair<double, int>> &Dy, const double sigma) {

  auto diff = norm_squared(Dx, Dy) / pow(sigma, 2);

  return exp(-0.5 * diff);
}

double p_dist(const vector<vector<double>> &l0,
              const vector<vector<double>> &l1,
              const unordered_map<int, int> &mapping, const double &sigma,
              const int curr0, const int curr1) {
  auto result = 1.0;
  for (auto &c : mapping) {
    result *= (exp(-0.5 / pow(sigma, 2) *
                   pow(l0[c.first][curr0] - l1[c.second][curr1], 2)));
  }

  return result;
}
double p_samp(const vector<pair<double, int>> &Dx,
              const vector<pair<double, int>> &Dy,
              const vector<vector<double>> &l0,
              const vector<vector<double>> &l1,
              const unordered_map<int, int> &mapping, const double &sigmaD,
              const double &sigmag, const int curr0, const int curr1) {
  return p_descr(Dx, Dy, sigmaD) *
         p_dist(l0, l1, mapping, sigmag, curr0, curr1);
}

double proximity_norm(const vector<double> &v0, const vector<double> &v1) {
  auto n = v0.size();
  auto v0_tmp = vector<pair<double, int>>(n);
  auto v1_tmp = vector<pair<double, int>>(n);
  for (auto i = 0; i < n; ++i) {
    v0_tmp[i] = make_pair(v0[i], i);
    v1_tmp[i] = make_pair(v1[i], i);
  }

  sort(v0_tmp.begin(), v0_tmp.end());
  sort(v1_tmp.begin(), v1_tmp.end());
  auto result = 0.0;
  for (auto i = 0; i < n; ++i) {
    if (v0_tmp[i].second == v1_tmp[i].second) {
      result += ((n - i) / (std::abs(v0_tmp[i].first - v1_tmp[i].first)));
    }
  }

  return 1.0 / result;
}
double global_consistency_criterion(const vector<double> &v0,
                                    const vector<double> &v1) {
  auto result = DBL_MIN;
  for (auto i = 0; i < v1.size(); ++i) {
    auto curr = v0[i] / v1[i] + v1[i] / v0[i];
    result = max(result, curr);
  }

  return result;
}
pair<int, double> best_matching_center(const vector<double> &curr,
                                       const vector<vector<double>> &others) {
  auto diff = DBL_MAX;
  auto best = -1;
  for (auto i = 0; i < others.size(); ++i) {
    auto curr_norm = norm(curr, others[i]);
    if (curr_norm < diff) {
      diff = curr_norm;
      best = i;
    }
  }

  return std::make_pair(best, diff);
}

pair<int, double>
best_matching_center_GH(const vector<double> &curr,
                        const vector<vector<double>> &others) {
  auto diff = DBL_MAX;
  auto best = -1;
  for (auto i = 0; i < others.size(); ++i) {

    auto difference = diff_vec(curr, others[i]);
    auto m = *max_element(difference.begin(), difference.end());
    if (m < diff) {
      diff = m;
      best = i;
    }
  }

  return std::make_pair(best, diff);
}
vec2d max_dist_between_landmarks(const vector<vector<double>> &d0,
                                 const vector<vector<double>> &d1,
                                 const vector<int> &centers0,
                                 const vector<int> &centers1) {
  auto result = vec2d{DBL_MIN, DBL_MIN};
  for (auto i = 0; i < d0.size(); ++i) {
    for (auto j = 0; j < centers0.size(); ++j)
      result.x() = max(d0[i][centers0[j]], result.x());

    for (auto h = 0; h < centers1.size(); ++h)
      result.y() = max(d1[i][centers1[h]], result.y());
  }
  return result;
}
double mapping_error(const unordered_map<int, int> &mapping,
                     const vector<int> &centers0, const vector<int> &centers1,
                     const vector<vector<double>> &d0,
                     const vector<vector<double>> &d1) {
  auto n = centers0.size();
  auto err = 0.0;
  // for (auto i = 0; i < d0.size(); ++i) {
  //   for (auto j = 0; j < n; ++j) {
  //     auto entryfxi = mapping.at(j);
  //     err += std::abs(d1[entryfxi][centers1[entryfxi]] - d0[i][centers0[j]]);
  //   }
  // }
  for (auto &c0 : mapping) {
    for (auto &c1 : mapping) {
      if (c0.first == c1.first)
        continue;
      err += pow(d1[c0.second][centers1[c1.second]] -
                     d0[c0.first][centers0[c1.first]],
                 2);
    }
  }

  return sqrt(err) / n;
}
std::tuple<unordered_map<int, int>, vector<bool>, vector<bool>>
compute_pairing(const vector<vector<double>> &v0,
                const vector<vector<double>> &v1, const vector<int> &centers0,
                const vector<int> centers1, const bool reversed) {
  auto n0 = centers0.size();
  auto n1 = centers1.size();
  auto taken = vector<bool>(n1, false);
  auto discarded = vector<bool>(n0, false);
  unordered_map<int, int> result;
  unordered_map<int, double> residuals;
  for (auto i = 0; i < n0; ++i) {
    auto best_match = best_matching_center(v0[i], v1);
    if (!taken[best_match.first]) {
      if (!reversed) {
        result[centers0[i]] = centers1[best_match.first];
        residuals[centers0[i]] = best_match.second;
        taken[best_match.first] = true;
      } else {
        result[centers0[best_match.first]] = centers1[i];
        residuals[centers0[best_match.first]] = best_match.second;
        taken[best_match.first] = true;
      }
    } else if (!reversed) {
      auto it = find_if(std::begin(result), std::end(result),
                        [&centers1, &best_match](auto &&p) {
                          return p.second == centers1[best_match.first];
                        });
      if (it == std::end(result))
        std::cout << "we messed up somewhere" << std::endl;
      auto old_one = it->first;
      if (residuals.at(old_one) > best_match.second) {
        result.erase(old_one);
        residuals.erase(old_one);
        result[centers0[i]] = centers1[best_match.first];
        residuals[centers0[i]] = best_match.second;
        auto entry = distance(centers0.begin(),
                              find(centers0.begin(), centers0.end(), old_one));
        discarded[entry] = true;
      } else
        discarded[i] = true;
    } else {
      auto oldone = centers0[best_match.first];
      if (residuals.at(oldone) > best_match.second) {
        result.erase(oldone);
        residuals.erase(oldone);
        result[centers0[best_match.first]] = centers1[i];
        residuals[centers0[best_match.first]] = best_match.second;
        auto entry = distance(centers0.begin(),
                              find(centers0.begin(), centers0.end(), oldone));
        discarded[entry] = true;
      } else
        discarded[i] = true;
    }
  }

  return {result, taken, discarded};
}
std::unordered_map<int, int>
voronoi_mapping(vector<int> &centers0, vector<int> &centers1,
                const vector<int> &lndmarks, const DrawableTrimesh<> &m0,
                const DrawableTrimesh<> &m1, const double &A0, const double &A1,
                const vector<vector<double>> &d0,
                const vector<vector<double>> &d1, const bool GH) {
  if ((int)lndmarks.size() % 2) {
    std::cout << "Error! There should be an even number of landmarks"
              << std::endl;
  }
  auto result = unordered_map<int, int>{};
  auto residuals = unordered_map<int, double>{};
  auto k = (int)lndmarks.size() / 2;
  auto n0 = centers0.size();
  auto n1 = centers1.size();
  auto v0 = vector<vector<double>>(n0, vector<double>(k));
  auto v1 = vector<vector<double>>(n1, vector<double>(k));
  auto max_distances = max_dist_between_landmarks(d0, d1, centers0, centers1);

  for (auto i = 0; i < lndmarks.size(); i += 2) {
    for (auto j = 0; j < n0; ++j)
      v0[j][i / 2] = d0[i / 2][centers0[j]] / A0;

    for (auto h = 0; h < n1; ++h)
      v1[h][i / 2] = d1[i / 2][centers1[h]] / A1;
  }

  if (n0 == n1) {
    auto taken = vector<bool>(n1, false);
    auto discarded = vector<bool>(n0, false);
    std::tie(result, taken, discarded) =
        compute_pairing(v0, v1, centers0, centers1, false);
    if (result.size() != n0) {
      std::tie(result, taken, discarded) =
          compute_pairing(v1, v0, centers1, centers0, true);
      auto it = centers0.begin();
      while (it != centers0.end()) {
        auto curr_entry = distance(centers0.begin(), it);
        if (taken[curr_entry]) {
          ++it;
        } else {
          it = centers0.erase(it);
          taken.erase(taken.begin() + curr_entry);
        }
      }

      it = centers1.begin();
      while (it != centers1.end()) {
        auto curr_entry = distance(centers1.begin(), it);
        if (!discarded[curr_entry])
          ++it;
        else {
          it = centers1.erase(it);
          discarded.erase(discarded.begin() + curr_entry);
        }
      }
    } else {
      auto it = centers1.begin();
      while (it != centers1.end()) {
        auto curr_entry = distance(centers1.begin(), it);
        if (taken[curr_entry]) {
          ++it;
        } else {
          it = centers1.erase(it);
          taken.erase(taken.begin() + curr_entry);
        }
      }

      it = centers0.begin();
      while (it != centers0.end()) {
        auto curr_entry = distance(centers0.begin(), it);
        if (!discarded[curr_entry])
          ++it;
        else {
          it = centers0.erase(it);
          discarded.erase(discarded.begin() + curr_entry);
        }
      }
    }

  } else if (n0 < n1) {
    auto taken = vector<bool>(n1, false);
    auto discarded = vector<bool>(n0, false);
    for (auto i = 0; i < n0; ++i) {
      auto best_match = (GH) ? best_matching_center_GH(v0[i], v1)
                             : best_matching_center(v0[i], v1);
      if (!taken[best_match.first]) {
        result[centers0[i]] = centers1[best_match.first];
        residuals[centers0[i]] = best_match.second;
        taken[best_match.first] = true;
      } else {
        auto it = find_if(std::begin(result), std::end(result),
                          [&centers1, &best_match](auto &&p) {
                            return p.second == centers1[best_match.first];
                          });
        if (it == std::end(result))
          std::cout << "we messed up somewhere" << std::endl;
        auto old_one = it->first;
        if (residuals.at(old_one) > best_match.second) {
          result.erase(old_one);
          residuals.erase(old_one);
          result[centers0[i]] = centers1[best_match.first];
          residuals[centers0[i]] = best_match.second;
          auto entry =
              distance(centers0.begin(),
                       find(centers0.begin(), centers0.end(), old_one));
          discarded[entry] = true;
        } else
          discarded[i] = true;
      }
    }

    auto it = centers1.begin();
    while (it != centers1.end()) {
      auto curr_entry = distance(centers1.begin(), it);
      if (taken[curr_entry]) {
        ++it;
      } else {
        it = centers1.erase(it);
        taken.erase(taken.begin() + curr_entry);
      }
    }

    it = centers0.begin();
    while (it != centers0.end()) {
      auto curr_entry = distance(centers0.begin(), it);
      if (!discarded[curr_entry])
        ++it;
      else {
        it = centers0.erase(it);
        discarded.erase(discarded.begin() + curr_entry);
      }
    }
  } else {
    auto taken = vector<bool>(n0, false);
    auto discarded = vector<bool>(n1, false);
    for (auto i = 0; i < n1; ++i) {
      auto best_match = (GH) ? best_matching_center_GH(v1[i], v0)
                             : best_matching_center(v1[i], v0);
      if (!taken[best_match.first]) {
        result[centers0[best_match.first]] = centers1[i];
        residuals[centers0[best_match.first]] = best_match.second;
        taken[best_match.first] = true;
      } else {
        auto oldone = centers0[best_match.first];
        if (residuals.at(oldone) > best_match.second) {
          result.erase(oldone);
          residuals.erase(oldone);
          result[centers0[best_match.first]] = centers1[i];
          residuals[centers0[best_match.first]] = best_match.second;
          auto entry = distance(centers0.begin(),
                                find(centers0.begin(), centers0.end(), oldone));
          discarded[entry] = true;
        } else
          discarded[i] = true;
      }
    }

    auto it = centers0.begin();
    while (it != centers0.end()) {
      auto curr_entry = distance(centers0.begin(), it);
      if (taken[curr_entry]) {
        ++it;
      } else {
        it = centers0.erase(it);
        taken.erase(taken.begin() + curr_entry);
      }
    }

    it = centers1.begin();
    while (it != centers1.end()) {
      auto curr_entry = distance(centers1.begin(), it);
      if (!discarded[curr_entry])
        ++it;
      else {
        it = centers1.erase(it);
        discarded.erase(discarded.begin() + curr_entry);
      }
    }
  }
  if (centers0.size() != centers1.size())
    std::cout << "Error!This vector should have the same length" << std::endl;

  // for (auto i = 0; i < centers0.size(); ++i) {
  //   if (residuals.find(centers0[i]) == residuals.end())
  //     continue;
  //   std::cout << "Best matching center has been assigned with an error of "
  //             << residuals.at(centers0[i]) << std::endl;
  // }

  return result;
}
std::tuple<int, int, double>
best_pairing(const vector<vector<pair<double, int>>> &Des0,
             const vector<vector<pair<double, int>>> &Des1,
             const vector<int> &centers0, const vector<int> &centers1,
             const double &sigma,
             const unordered_map<int, vector<int>> &attemped) {
  auto best0 = -1;
  auto best1 = -1;
  auto res = DBL_MIN;
  for (auto j = 0; j < centers0.size(); ++j) {
    for (auto i = 0; i < centers1.size(); ++i) {

      auto it = attemped.find(centers0[j]);
      bool already_checked = false;
      if (it != attemped.end()) {
        auto checked = attemped.at(centers0[i]);
        already_checked = (find(checked.begin(), checked.end(), centers1[i]) !=
                           checked.end());
      }
      if (already_checked)
        continue;
      auto curr_res = p_descr(Des0[j], Des1[i], sigma);
      if (curr_res > res) {
        res = curr_res;
        best0 = j;
        best1 = i;
      }
    }
  }

  return {best0, best1, res};
}
std::tuple<int, int, double>
best_pairing(const vector<vector<pair<double, int>>> &Des0,
             const vector<vector<pair<double, int>>> &Des1,
             const vector<int> &centers0, const vector<int> &centers1,
             const double &sigma_des, const double &sigma_dist,
             const vector<vector<double>> &d0, const vector<vector<double>> &d1,
             const unordered_map<int, vector<int>> &attemped,
             const unordered_map<int, int> &mapping, const vector<bool> &taken,
             const vector<bool> &added) {
  auto best0 = -1;
  auto best1 = -1;
  auto res = DBL_MIN;
  for (auto j = 0; j < centers0.size(); ++j) {
    if (added[j])
      continue;
    for (auto i = 0; i < centers1.size(); ++i) {
      if (taken[i])
        continue;
      auto it = attemped.find(centers0[j]);
      bool already_checked = false;
      if (it != attemped.end()) {
        auto checked = attemped.at(centers0[i]);
        already_checked = (find(checked.begin(), checked.end(), centers1[i]) !=
                           checked.end());
      }
      if (already_checked)
        continue;
      auto curr_res = p_samp(Des0[j], Des1[i], d0, d1, mapping, sigma_des,
                             sigma_dist, centers0[j], centers1[i]);

      if (curr_res > res) {
        res = curr_res;
        best0 = j;
        best1 = i;
      }
    }
  }

  return {best0, best1, res};
}
void add_pairing(unordered_map<int, vector<int>> &map,
                 const pair<int, int> &new_one) {
  auto it = map.find(new_one.first);
  if (it == map.end())
    map[new_one.first] = vector<int>{new_one.second};
  else {
    auto &v = map[new_one.first];
    v.push_back(new_one.second);
  }
}
std::unordered_map<int, int> voronoi_mapping(vector<int> &centers0,
                                             vector<int> &centers1,
                                             const DrawableTrimesh<> &m0,
                                             const DrawableTrimesh<> &m1,
                                             const vector<vector<double>> &c0,
                                             const vector<vector<double>> &c1) {

  auto n0 = centers0.size();
  auto n1 = centers1.size();

  auto Des0 = compute_descriptors(m0, centers0, c0);
  auto Des1 = compute_descriptors(m1, centers1, c1);
  auto sigma_des = 0.1;
  auto sigma_dist = 0.1;
  unordered_map<int, vector<int>> attempted;
  unordered_map<int, int> result;
  auto count = 0;
  auto err = DBL_MAX;
  auto prev_err = DBL_MAX;
  auto best_err = DBL_MAX;
  auto epsilon = 3 * m0.edge_avg_length();
  while (count < 150 && err > epsilon) {
    auto curr_map = unordered_map<int, int>{};

    if (n0 <= n1) {
      vector<bool> added(n0, false);
      vector<bool> taken(n1, false);
      if (curr_map.size() == 0) {
        auto [p0, p1, res] =
            best_pairing(Des0, Des1, centers0, centers1, sigma_des, attempted);
        curr_map[p0] = p1;
        added[p0] = true;
        taken[p1] = true;
        add_pairing(attempted, make_pair(p0, p1));
      }
      while (curr_map.size() != n0) {
        auto [p0, p1, res] =
            best_pairing(Des0, Des1, centers0, centers1, sigma_des, sigma_dist,
                         c0, c1, attempted, curr_map, taken, added);
        curr_map[p0] = p1;
        added[p0] = true;
        taken[p1] = true;
        add_pairing(attempted, make_pair(p0, p1));
      }

      err = mapping_error(curr_map, centers0, centers1, c0, c1);
      if (err < prev_err) {
        result = curr_map;
        best_err = err;
      }

      prev_err = err;

      ++count;
    }
  }
  std::cout << "we reached " << count << "iterations" << std::endl;
  std::cout << "Best error " << best_err << std::endl;
  std::cout << "Threshold error " << epsilon << std::endl;

  unordered_map<int, int> res;
  for (auto &p : result)
    res[centers0[p.first]] = centers1[p.second];

  if (n0 < n1) {
    auto it = centers1.begin();
    while (it != centers1.end()) {
      bool present = false;
      for (auto &p : res) {
        if (p.second == *it) {
          present = true;
          break;
        }
      }
      if (present)
        ++it;
      else
        it = centers1.erase(it);
    }
  } else
    std::cout << "You need to handle this case" << std::endl;
  return res;
}

double voronoi_region(const DrawableTrimesh<> &m, const int vid,
                      const int tid) {

  auto k = m.poly_vert_offset(tid, vid);
  auto tri = m.adj_p2v(tid);
  auto v = m.vert(tri[k]);
  auto v0 = m.vert(tri[(k + 1) % 3]);
  auto v1 = m.vert(tri[(k + 2) % 3]);
  auto alpha0 = (v0 - v).angle_rad(v0 - v1);
  auto alpha1 = (v1 - v).angle_rad(v1 - v);

  return 1.0 / 8.0 *
         ((v - v0).norm_sqrd() * cot(alpha0) +
          (v - v1).norm_sqrd() * cot(alpha1));
}
vector<double> vert_voronoi_mass(const DrawableTrimesh<> &m) {
  auto result = vector<double>(m.num_verts());
  for (auto i = 0; i < m.num_verts(); ++i) {
    auto sum = 0.0;
    for (auto tid : m.adj_v2p(i))
      sum += voronoi_region(m, i, tid);

    result[i] = sum;
  }

  return result;
}
MatrixXd geodesic_matrix(const geodesic_solver &solver) {
  auto n = solver.graph.size();
  MatrixXd result(n, n);
  for (auto j = 0; j < n; ++j) {
    auto d = compute_geodesic_distances(solver, {j}, geodesic);
    for (auto i = j; i < n; ++i) {
      result(i, j) = d[i];
      result(j, i) = d[i];
    }
  }
  return result;
}
double diameter(const MatrixXd &G) {
  auto result = DBL_MIN;
  for (auto i = 0; i < G.rows(); ++i) {
    for (auto j = i; j < G.cols(); ++j) {
      result = max(result, G(i, j));
    }
  }

  return result;
}
