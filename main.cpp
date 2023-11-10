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

//::::::::::::::::::::::::::::::::::::GUI utilitIES ::::::::::::::::::::::::::::::::::::

inline void draw_cp(const vector<DrawableSphere> &cp, GLcanvas &gui) {
  for (auto &point : cp) //if (point.radius > 0) 
  gui.push(&point, false);
}

inline void remove_cp(const vector<DrawableSphere> &cp, GLcanvas &gui) {
  for (auto &point : cp) gui.pop(&point);
}

inline void set_critical_points(const vector<int> &c, vector<DrawableSphere> &cp, float s)
{
  for (uint i=0;i<c.size();i++) {
    if (c[i]==-1) continue;
    cp[i].radius = s;
    if (c[i]==0) cp[i].color = cinolib::Color::BLUE();
    else if (c[i]==1) cp[i].color = cinolib::Color::RED();
    else cp[i].color = cinolib::Color::GREEN();
  }
}

inline void reset_critical_points(vector<DrawableSphere> &cp)
{
  for (auto &point : cp) {
    point.radius = 0.0;
    point.color = cinolib::Color::BLACK();
  }
}

ScalarField Clamp_and_rescale_field(const vector<double> &f, const float cl[])
// f field is in [0,1] both in input and in output
// cl clamp limits: clamp f in [cl[0],cl[1]] and rescale it to [0,1]
{ 
  ScalarField sf(f);
  // cout << "Actual range [" << sf.minCoeff() << "," << sf.maxCoeff() << "] clapmed to [" << cl[0] << "," << cl[1] << "] ";
  for (auto i=0;i<sf.size();i++) 
    if (sf(i)<cl[0]) sf(i) = 0.0;
    else if (sf(i) > cl[1]) sf(i) = 1.0;
    else sf(i) = (sf(i)-cl[0])/(cl[1]-cl[0]);
  // cout << "rescaled in [" << sf.minCoeff() << "," << sf.maxCoeff() << "]\n";
  return sf;
}

//====================== GENERAL UTILITIES =============================================

void InvertSparseMatrix(Eigen::SparseMatrix<double> &Y)
{
  for(int k=0; k<Y.outerSize(); ++k) {
    for(typename Eigen::SparseMatrix<double>::InnerIterator it(Y,k); it; ++it) {
      if (it.col() == it.row()) {
        double v = it.value();
        v = 1.0/v;
        Y.coeffRef(it.row(),it.col()) = v;
      }
    }
  }
}

void normalize_in_01(Eigen::VectorXd &f)
{
    long double min = f.minCoeff();
    long double max = f.maxCoeff();
    long double delta = max - min;
    for(int i=0;i<f.size();i++) f[i] = (double)((f[i]-min) / delta);
}

//==================== FIELD GENERATORS ========================

double compute_theta(const DrawableTrimesh<> &m, const int tid, const int vid) {
  uint k = m.poly_vert_offset(tid, vid);
  uint vid1 = m.poly_vert_id(tid, (k + 1) % 3);
  uint vid2 = m.poly_vert_id(tid, (k + 2) % 3);

  vec3d v1 = m.vert(vid1)-m.vert(vid);
  vec3d v2 = m.vert(vid2)-m.vert(vid);

  double theta = v1.angle_rad(v2);

  return theta;
}

double gaussian_curvature(const DrawableTrimesh<> &m, const int vid) {
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
    vec3d n = m.vert_data(vid).normal;      //vert_data(vid).normal;
    if (Hni.dot(n)<0) H(vid) = -H(vid); 
  }
  return H;
}

//=============================== INPUT FIELD ==================================
Eigen::VectorXd Generate_field(const DrawableTrimesh<> &m)
{
  // field is mean curvature 
  Eigen::SparseMatrix<double> ML  = laplacian(m, COTANGENT);
  Eigen::SparseMatrix<double> MM = mass_matrix(m);
  Eigen::VectorXd buf;
  InvertSparseMatrix(MM);
  ML = MM * ML;
  buf = Mean_curvature(m,ML);
  // buf = gaussian_curvature(m);
  normalize_in_01(buf);
  return buf;
}

void Set_clamp_limits(const Eigen::VectorXd &f, int sigma_multiplier, float cl[]) 
{
  // set clamp limits to sigma_multiplier * sigma
  double mean = f.sum()/f.size();
  Eigen::VectorXd s(f);
  for (auto i=0;i<s.size();i++) s(i) = (s(i)-mean)*(s(i)-mean);
  double sigma = sqrt(s.sum()/s.size());
  // cl[0] = mean - sigma_multiplier * sigma;
  // cl[1] = mean + sigma_multiplier * sigma;
  cl[0] = std::max(mean - sigma_multiplier * sigma,0.0);
  cl[1] = std::min(mean + sigma_multiplier * sigma,1.0);
  // cout << "clamp limits: " << cl[0] << ", " << cl[1] << endl;
}

//=========================== PROCESSING FUNCTIONS =============================

vector<Eigen::VectorXd> Build_discrete_scale_space(const DrawableTrimesh<> &m, const Eigen::VectorXd &f, 
                                                    double time_scalar, double time_multiplier, int levels) 
{
  // DrawableTrimesh<> m(m1);
  // MCF(m,1);
  Eigen::SparseMatrix<double> L  = laplacian(m, COTANGENT);
  Eigen::SparseMatrix<double> MM = mass_matrix(m);
  vector<Eigen::VectorXd> buf(levels);
  buf[0]=f;
  normalize_in_01(buf[0]);
  for (auto i=1;i<levels;i++) {
    if (i%10==0) cout << "level "<< i << " completed\n";
    // backward Euler time integration of heat flow equation
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> LLT(MM - time_scalar * L);
    buf[i] = LLT.solve(MM * buf[i-1]);
    normalize_in_01(buf[i]);
    time_scalar *= time_multiplier;
  }
  return buf;
}

vector<int> Find_Critical_Points(const DrawableTrimesh<> &m, const vector<vector<uint>> &VV,
                                          const Eigen::VectorXd &f, vector<vector<int>> &ties)
{
  // -1 regular; 0 minimum; 1 maximum; k>1 (k-1)-saddle
  vector<int> buf(f.size());
  uint nv = m.num_verts();
  for(uint vid=0; vid<nv; vid++) {
    vector<uint> neigh = VV[vid];
    int nn = neigh.size();
    vector<bool> sign(nn);    // true iff neighbor is smaller
    for (uint j=0;j<nn;j++) { // cycle on neighbors
      if (f(neigh[j])==f(vid)) { // solve ties with vertex index
        sign[j] = (vid>neigh[j]); 
        vector<int> t(2);
        t[0]=vid; t[1]=neigh[j];
        ties.push_back(t);
      }
      else if (f(neigh[j])<f(vid)) sign[j] = true;
      else sign[j] = false;
    }
    uint count = 0;
    for (uint j=0; j<nn; j++)
      if (sign[j]!=sign[(j+1)%nn]) count++;
    if (count==0) buf[vid] = sign[0]?1:0;       // max or min depending on sign
    else if (count==2) buf[vid] = -1;           // regular
    else if (count%2 == 0) buf[vid] = count/2;  // k-saddle
    else cout << "Error at vertex " << vid << ": odd number of change sign\n";
  }
  return buf;
}

void print_statistics(const vector<vector<int>> &c)
{
  for (uint i=0;i<c.size();i++) {
    cout << "Field " << i << ": ";
    std::vector<int>::iterator cit;
    int k = *std::max_element(c[i].begin(),c[i].end());
    vector<int> counts(k+1,0);
    for (uint vid=0;vid<c[i].size();vid++)
      if (c[i][vid]!=-1) counts[c[i][vid]]++;
    cout << counts[0] << " minima, " << counts[1] << " maxima, ";
    for (uint j=2; j<=k; j++) cout << counts[j] << " " << j-1 << "-saddles ";
    cout << endl;
  }
}




//=============================== MAIN =========================================

int main(int argc, char **argv) {
  if (argc<4) {cout << "Usage: Mesh_scale_space filename num_levels time_step\n"; return 1;}

  //INPUT MESH
  auto name = std::string(argv[1]);
  auto s = "../data/" + name;
  DrawableTrimesh<> m(s.c_str());
  uint nverts = m.num_verts();
  vector<vector<uint>> VV(nverts); // Vertex-Vertex relation
  for (auto i=0;i<nverts;i++) VV[i]=m.vert_ordered_verts_link(i);
  // m.normalize_bbox();
  // m.center_bbox();   
  // MCF(m,1,0.0001);
 
  // OUTPUT FIELDS
  uint nlevels = stoi(argv[2]); 
  vector<vector<double>> fields(nlevels,vector<double>(nverts));

  // GENERATE FIELD
  Eigen::VectorXd f = Generate_field(m);
  // f *= 100000000;
  vector<float*> clamp_limits(nlevels);
  for (auto &l : clamp_limits) l = new float[2];

  // COMPUTE
  cout << "Computing discrete scale space: \n";
  double time_step = stod(argv[3]);
  double time_mult = stod(argv[4]);
  vector<Eigen::VectorXd> efields = Build_discrete_scale_space(m,f,time_step,time_mult,nlevels);
  for (auto i=0;i<efields.size();i++) {
    fields[i] = vector(efields[i].data(),efields[i].data()+efields[i].size());
  }
  cout << "done"<< endl;

  for (auto i=0;i<nlevels;i++) 
    Set_clamp_limits(efields[i], 2, clamp_limits[i]); // set clamp limits to sigma

  cout << "Finding critical points: \n";
  vector<vector<int>> critical(nlevels,vector<int>(nverts));
  vector<vector<int>> ties;
  for (auto i=0;i<efields.size();i++) {
    critical[i] = Find_Critical_Points(m,VV,efields[i],ties);
    if (ties.size()>0) {
      cout << "Found ties at level " << i << ": ";
      for (auto i=0;i<ties.size();i++) 
        cout << "(" << ties[i][0] << "," << ties[i][1] << "), ";
      cout << endl;
    }
    ties.resize(0);
  }
  cout << "done"<< endl;
  print_statistics(critical);

  // GUI
  GLcanvas gui;
  ScalarField phi;
  int selected_entry = 0;
  float curr_clamp[2];
  bool show_sf = false;
  bool show_cp = false;
  bool show_wf = false;
  gui.show_side_bar = true;

  // bullets for critical points
  float point_size = m.edge_avg_length()/2;
  vector<DrawableSphere> points(nverts);
  for (uint i=0;i<nverts;i++)
    points[i]=DrawableSphere(m.vert(i),0.0,cinolib::Color::BLACK());
 
  gui.push(&m);
  m.show_wireframe(false);
 
  gui.callback_app_controls = [&]() {
    if (ImGui::Checkbox("Show wireframe", &show_wf)) {
      if (show_wf) m.show_wireframe(true);
      else m.show_wireframe(false);
      m.updateGL();
    } 
    if (ImGui::Checkbox("Show Scalar Field", &show_sf)) {
      if (show_sf) {
        curr_clamp[0]=clamp_limits[selected_entry][0];
        curr_clamp[1]=clamp_limits[selected_entry][1];
        phi = Clamp_and_rescale_field(fields[selected_entry],curr_clamp);
        // phi.normalize_in_01();
        phi.copy_to_mesh(m);
        m.show_texture1D(TEXTURE_1D_HSV);
      } else {
        m.show_poly_color();
      }
    } 
    ImGui::SameLine();
    if (ImGui::Checkbox("Show Critical Points", &show_cp)) {
      if (show_cp) {
        set_critical_points(critical[selected_entry],points,point_size);
        draw_cp(points,gui);
      } else {
        reset_critical_points(points);
        remove_cp(points,gui);
      }
      m.updateGL();
    } 
    // if (ImGui::Button("Dummy button 2")) {
    //   auto name0 = "CMFC" + name + ".obj";
    //   write_OBJ(name0.c_str(), obj_wrapper(m.vector_verts()),
    //             m.vector_polys());
    // }
    if (ImGui::SliderFloat2("Clamp values", curr_clamp, f.minCoeff(), f.maxCoeff(),"%.4f",ImGuiSliderFlags_Logarithmic)) {
      if (show_sf) {
        phi = Clamp_and_rescale_field(fields[selected_entry],curr_clamp);
        phi.copy_to_mesh(m);
        m.updateGL();
      }
    }
    if (ImGui::SliderInt("Choose level", &selected_entry, 0, nlevels - 1)) {
      if (show_sf) {
        curr_clamp[0]=clamp_limits[selected_entry][0];
        curr_clamp[1]=clamp_limits[selected_entry][1];
        phi = Clamp_and_rescale_field(fields[selected_entry],curr_clamp);
        phi.copy_to_mesh(m);
      }
      if (show_cp) {
        reset_critical_points(points);
        set_critical_points(critical[selected_entry],points,point_size);
      }
      if (show_sf || show_cp) m.updateGL();
    }
  };

  gui.callback_mouse_left_click = [&](int modifiers) -> bool {
    if (modifiers & GLFW_MOD_SHIFT) {
      vec3d p;
      vec2d click = gui.cursor_pos();
      if (gui.unproject(click, p)) {
        uint vid = m.pick_vert(p);
        cout << "Picked vertex " << vid << " field value " 
              << std::setprecision(20) << fields[selected_entry][vid] 
              << " at level " << selected_entry << endl;
        // m.vert_data(vid).color = Color::RED();
        // m.updateGL();
      }
    }
    return false;
  };

  return gui.launch();
}
//=========================== UNUSED FUNCTIONS =======================

Eigen::VectorXd Laplacian_smooth_signal(const DrawableTrimesh<> & m, const Eigen::VectorXd & f,
                                        double time_scalar)
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




