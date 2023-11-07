#include "utilities.h"
#include <cinolib/drawable_sphere.h>
#include <cinolib/geodesics.h>
#include <cinolib/gl/glcanvas.h>
#include <cinolib/gradient.h>
#include <cinolib/io/write_OBJ.h>
#include <cinolib/mean_curv_flow.h>
#include <fstream>
using namespace std;
using namespace cinolib;
// NOTES: 71 does not get the correspondence right on one hand
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//                                 GUI utility

void draw_pc(const point_cloud &pc, GLcanvas &gui) {
  for (auto &point : pc.points) {
    gui.push(&point, false);
  }
}

void remove_pc(const point_cloud &pc, GLcanvas &gui) {
  for (auto &point : pc.points) {
    gui.pop(&point);
  }
}

bool draw_combobox(const char *lbl, int &value, const vector<string> &labels) {
  if (!ImGui::BeginCombo(lbl, labels[value].c_str()))
    return false;
  auto old_val = value;
  for (auto i = 0; i < labels.size(); i++) {
    ImGui::PushID(i);
    if (ImGui::Selectable(labels[i].c_str(), value == i))
      value = i;
    if (value == i)
      ImGui::SetItemDefaultFocus();
    ImGui::PopID();
  }
  ImGui::EndCombo();
  return value != old_val;
}

//=========================== PROCESSING FUNCTIONS PROTOTYPES ==================

Eigen::VectorXd gaussian_curvature(const DrawableTrimesh<> &);
vector<Eigen::VectorXd> Build_discrete_scale_space(const DrawableTrimesh<> &, const Eigen::VectorXd &, 
                                                                                        double, const int);
Eigen::VectorXd Laplacian_smooth_signal(const DrawableTrimesh<> &, const Eigen::VectorXd &, const double);
void Laplacian_smooth_mesh(DrawableTrimesh<> &, const double);

Eigen::VectorXd Mean_curvature(const DrawableTrimesh<> &, const Eigen::SparseMatrix<double> &);

void InvertSparseMatrix(Eigen::SparseMatrix<double> &);
//=============================== MAIN =========================================

int main(int argc, char **argv) {
  if (argc<4) {cout << "Usage: Mesh_scale_space filename num_levels time_step\n"; return 1;}

  //INPUT MESH
  auto name = std::string(argv[1]);
  auto s = "../data/" + name + ".off";
  DrawableTrimesh<> m(s.c_str());
  uint nverts = m.num_verts();
  m.normalize_bbox();
  m.center_bbox();   
  m.vert_data(0).normal;
  m.vector_vert_normals();
   
  // OUTPUT FIELDS
  uint nlevels = stoi(argv[2]); 
  vector<vector<double>> fields(nlevels,vector<double>(nverts));

  // COMPUTE
  cout << "Computing discrete scale space: " << flush;
  double time_step = stod(argv[3]);
  // Eigen::VectorXd K = gaussian_curvature(m);
  Eigen::SparseMatrix<double> ML  = laplacian(m, COTANGENT);
  Eigen::SparseMatrix<double> MM = mass_matrix(m);
  InvertSparseMatrix(MM);
  ML = MM * ML;
  Eigen::VectorXd K = Mean_curvature(m,ML);
  vector<Eigen::VectorXd> efields = Build_discrete_scale_space(m,K,time_step,nlevels);
  for (auto i=0;i<efields.size();i++) {
    fields[i] = vector(efields[i].data(),efields[i].data()+efields[i].size());
    cout <<  "*" << flush << flush;
  }
  cout << endl;

  // GUI
  GLcanvas gui;
  ScalarField phi;
  float point_size = 0.002;
  int selected_entry = 0;

  bool show_sf = false;
 
  gui.draw_side_bar();
  gui.push(&m, false);
  gui.callback_app_controls = [&]() {
    if (ImGui::Checkbox("Show Scalar Field", &show_sf)) {
      if (show_sf) {
        phi = ScalarField(fields[selected_entry]);
//        phi.normalize_in_01();
        phi.copy_to_mesh(m);
        m.show_texture1D(TEXTURE_1D_HSV);
      } else {
        m.show_poly_color();
      }
    } 
    // if (ImGui::Button("Dummy button 1")) {
    //   string foldername = "../teaser/";
    //   // do something
    // }
    // ImGui::SameLine();
    // if (ImGui::Button("Dummy button 2")) {
    //   auto name0 = "CMFC" + name + ".obj";
    //   write_OBJ(name0.c_str(), obj_wrapper(m.vector_verts()),
    //             m.vector_polys());
    // }
    if (ImGui::SliderInt("Choose level", &selected_entry, 0,
                         nlevels - 1)) {
      if (show_sf) {
        phi = ScalarField(fields[selected_entry]);
        // phi.normalize_in_01();
        phi.copy_to_mesh(m);
        m.show_texture1D(TEXTURE_1D_HSV);
      }
    }
  };

  // gui.callback_mouse_left_click = [&](int modifiers) -> bool {
  //   if (modifiers & GLFW_MOD_SHIFT) {
  //     vec3d p;
  //     vec2d click = gui.cursor_pos();
  //     if (gui.unproject(click, p)) {
  //       uint vid = m.pick_vert(p);

  //       m.vert_data(vid).color = Color::RED();
  //       m.updateGL();
  //     }
  //   }
  //   return false;
  // };

  return gui.launch();
}

//===================================== PROCESSING FUNCTIONS ===================================

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

vector<Eigen::VectorXd> Build_discrete_scale_space(const DrawableTrimesh<> &m, const Eigen::VectorXd &f, 
                                                                    double time_scalar, const int levels) 
{
    Eigen::SparseMatrix<double> L  = laplacian(m, COTANGENT);
    Eigen::SparseMatrix<double> MM = mass_matrix(m);
    vector<Eigen::VectorXd> buf;
    buf.push_back(f);
    for (auto i=1;i<levels;i++) {
      // backward euler time integration of heat flow equation
      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> LLT(MM - time_scalar * L);
      buf.push_back(LLT.solve(MM * f));
      time_scalar *= 2;
    }
    return buf;
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

Eigen::VectorXd Mean_curvature(const DrawableTrimesh<> & m, const Eigen::SparseMatrix<double> &ML)
{
  uint nv = m.num_verts();
  Eigen::MatrixXd V(nv,3);
  for(uint vid=0; vid<nv; ++vid) {
      vec3d pos = m.vert(vid);
      V(vid,0) = pos.x();
      V(vid,1) = pos.y();
      V(vid,2) = pos.z();
  }
  Eigen::MatrixXd Hn = ML * V;
  Hn *= -0.5;
  Eigen::VectorXd H(nv);
  for(uint vid=0; vid<nv; ++vid) {
    H(vid)=Hn.row(vid).norm();
    vec3d Hni(Hn(vid,0),Hn(vid,1),Hn(vid,2));
    if (Hni.dot(m.vert_data(vid).normal())<0) H(vid) = -H(vid); 
  }
  return H;
}

void InvertSparseMatrix(Eigen::SparseMatrix<double> &Y)
{
    // Iterate over outside
  for(int k=0; k<Y.outerSize(); ++k)
  {
    // Iterate over inside
    for(typename MatY::InnerIterator it (Y,k); it; ++it)
    {
      if(it.col() == it.row())
      {
        Scalar v = it.value();
        assert(v != 0);
        v = ((Scalar)1.0)/v;
        Y.coeffRef(it.row(),it.col()) = v;
      }
    }
  }
}
