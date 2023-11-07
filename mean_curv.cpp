// #include "diff_geo.h"
#include "utilities.h"
#include <cinolib/io/write_OBJ.h>
#include <cinolib/io/write_OFF.h>
#include <cinolib/mean_curv_flow.h>
// #include <cinolib/meshes/drawable_trimesh.h>
#include <fstream>
#include <iostream>
using namespace std;
using namespace cinolib;

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

double compute_theta(const DrawableTrimesh<> &m, const int tid, const int vid) {
  uint k = m.poly_vert_offset(tid, vid);
  uint vid1 = m.poly_vert_id(tid, (k + 1) % 3);
  uint vid2 = m.poly_vert_id(tid, (k + 2) % 3);

  vec3d v1 = m.vert(vid1)-m.vert(vid);
  vec3d v2 = m.vert(vid2)-m.vert(vid);

  double theta = v1.angle_rad(v2);

  return theta;
}

double gaussian_curvature(const DrawableTrimesh<> &m,const int vid) {
  double result = 0.;
  vector<uint> star = m.adj_v2p(vid);

  for (uint tid : star)
    result += compute_theta(m, tid, vid);

  return 2 * M_PI - result;
}

Eigen::VectorXd gaussian_curvature(const DrawableTrimesh<> &m) {
  Eigen::VectorXd result(m.num_verts());
  for (uint i = 0; i < m.num_verts(); ++i)
    result(i) = gaussian_curvature(m, i);
  return result;
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Eigen::VectorXd Laplacian_smooth_signal(const DrawableTrimesh<> & m, const Eigen::VectorXd & f,
         const double                   time_scalar)
{
    Eigen::SparseMatrix<double> L  = laplacian(m, COTANGENT);
    Eigen::SparseMatrix<double> MM = mass_matrix(m);

    // backward euler time integration of heat flow equation
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> LLT(MM - time_scalar * L);
    // return LLT.solve(MM * f);
    Eigen::VectorXd x = LLT.solve(MM * f);
    return x;
}

void Laplacian_smooth_mesh(DrawableTrimesh<> & m, const double time_scalar)
{
    Eigen::SparseMatrix<double> L  = laplacian(m, COTANGENT);
    Eigen::SparseMatrix<double> MM = mass_matrix(m);

    // backward euler time integration of heat flow equation
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> LLT(MM - time_scalar * L);

    uint nv = m.num_verts();
    Eigen::VectorXd x(nv);
    Eigen::VectorXd y(nv);
    Eigen::VectorXd z(nv);

    for(uint vid=0; vid<nv; ++vid) {
      vec3d pos = m.vert(vid);
      x[vid] = pos.x();
      y[vid] = pos.y();
      z[vid] = pos.z();
    }

    x = LLT.solve(MM * x);
    y = LLT.solve(MM * y);
    z = LLT.solve(MM * z);

    for(uint vid=0; vid<m.num_verts(); ++vid) {
      vec3d new_pos(x[vid], y[vid], z[vid]);
      m.vert(vid) = new_pos;
    }
}


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//                                utility

int main(int argc, char **argv) {

  vector<string> names = {"hippo", "lion"};
  double time_step = 0.00001;

   for (auto name : names) {
    vector<vec3d> pos;
    vector<vector<uint>> tris;
    auto s = "../data_extra/" + name + ".off";
    //read_OBJ(s.c_str(), pos, tris);
    read_OFF(s.c_str(), pos, tris);
    DrawableTrimesh<> m(pos, tris);
    // optimize position and scale to get better numerical precision
    m.normalize_bbox();
    m.center_bbox();        

    Eigen::VectorXd K = gaussian_curvature(m);
    cout << "Total defect orginal " << K.sum()/M_PI << " Pi\n"; 

    Eigen::VectorXd K_smoothed = K;
    // for (int i=0;i<100;i++) K_smoothed = Laplacian_smooth_signal(m, K_smoothed, time_step);
    // cout << "Total defect smoothed iter " << K_smoothed.sum()/M_PI << " Pi\n"; 
    K_smoothed = Laplacian_smooth_signal(m, K, time_step);
    cout << "Total defect smoothed single " << K_smoothed.sum()/M_PI << " Pi\n"; 

    // write output
    vector<double> K_vec(K_smoothed.data(),K_smoothed.data()+K_smoothed.size());
    auto filename = "../data_extra/" + name + "_field.dat";
    export_field(K_vec,filename);

    // Laplacian_smooth_mesh(m,time_step);
    // K = gaussian_curvature(m);
    // cout << "Total defect smoothed mesh " << K.sum()/M_PI << " Pi\n"; 
    // auto filename = "../data_extra/" + name + "_smoothed_L.obj";
    // write_OBJ(filename.c_str(),obj_wrapper(m.vector_verts()),m.vector_polys());
  }

  return 0;
}
