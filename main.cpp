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

void draw_cp(const vector<DrawableSphere> &cp, GLcanvas &gui) {
  for (auto &point : cp) {
    gui.push(&point, false);
  }
}

void remove_cp(const vector<DrawableSphere> &cp, GLcanvas &gui) {
  for (auto &point : cp) {
    gui.pop(&point);
  }
}

void set_critical_points(const vector<int> &c, vector<DrawableSphere> &cp, float s)
{
  for (uint i=0;i<c.size();i++) {
    if (c[i]==-1) continue;
    cp[i].radius = s;
    if (c[i]==0) cp[i].color = cinolib::Color::BLUE();
    else if (c[i]==1) cp[i].color = cinolib::Color::RED();
    else cp[i].color = cinolib::Color::GREEN();
  }
}

void reset_critical_points(vector<DrawableSphere> &cp)
{
  for (auto &point : cp) {
    point.radius = 0.0;
    point.color = cinolib::Color::BLACK();
  }
}

//=========================== PROCESSING FUNCTIONS PROTOTYPES ==================

Eigen::VectorXd gaussian_curvature(const DrawableTrimesh<> &);
vector<Eigen::VectorXd> Build_discrete_scale_space(const DrawableTrimesh<> &, const Eigen::VectorXd &, 
                                                                                        double, const int);
Eigen::VectorXd Laplacian_smooth_signal(const DrawableTrimesh<> &, const Eigen::VectorXd &, const double);
void Laplacian_smooth_mesh(DrawableTrimesh<> &, const double);

Eigen::VectorXd Mean_curvature(const DrawableTrimesh<> &, const Eigen::SparseMatrix<double> &);

vector<int> Find_Critical_Points(const DrawableTrimesh<> &, const Eigen::VectorXd &);

//====================== UTILITIES =============================================

void InvertSparseMatrix(Eigen::SparseMatrix<double> &Y)
{
    // Iterate over outside
  for(int k=0; k<Y.outerSize(); ++k)
  {
    // Iterate over inside
    for(typename Eigen::SparseMatrix<double>::InnerIterator it(Y,k); it; ++it)
    {
      if (it.col() == it.row())
      {
        double v = it.value();
        v = 1.0/v;
        Y.coeffRef(it.row(),it.col()) = v;
      }
    }
  }
}

void progress_log(uint i) { std::cout <<  i << " "; std::cout.flush(); }

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


//=============================== INPUT FIELD ==================================
Eigen::VectorXd Generate_field(const DrawableTrimesh<> &m)
{
  // field is mean curvature 
  Eigen::SparseMatrix<double> ML  = laplacian(m, COTANGENT);
  Eigen::SparseMatrix<double> MM = mass_matrix(m);
  InvertSparseMatrix(MM);
  ML = MM * ML;
  return Mean_curvature(m,ML);
  // return gaussian_curvature(m);
}

void Set_clamp_limits(const Eigen::VectorXd &f, int sigma_multiplier, float cl[]) 
{
  // set clamp limits to sigma_multiplier * sigma
  double mean = f.sum()/f.size();
  Eigen::VectorXd s(f);
  for (auto i=0;i<s.size();i++) s(i) = (s(i)-mean)*(s(i)-mean);
  double sigma = sqrt(s.sum()/s.size());
  cl[0] = mean - sigma_multiplier * sigma;
  cl[1] = mean + sigma_multiplier * sigma;
  cout << "Field limits: " << f.minCoeff() << ", " << f.maxCoeff() 
      << "; clamp limits: " << cl[0] << ", " << cl[1] << endl;
}

ScalarField Rescale_field(const vector<double> & f, float cl[])
{
  ScalarField sf(f);
  cout << "Actual range [" << sf.minCoeff() << "," << sf.maxCoeff() << "] clapmed to [" << cl[0] << "," << cl[1] << "] ";
  for (auto i=0;i<sf.size();i++) 
    if (sf(i)<cl[0]) sf(i) = 0.0;
    else if (sf(i) > cl[1]) sf(i) = 1.0;
    else sf(i) = (sf(i)-cl[0])/(cl[1]-cl[0]);
  cout << "rescaled in [" << sf.minCoeff() << "," << sf.maxCoeff() << "]\n";
  return sf;
}

//=============================== MAIN =========================================

int main(int argc, char **argv) {
  if (argc<4) {cout << "Usage: Mesh_scale_space filename num_levels time_step\n"; return 1;}

  //INPUT MESH
  auto name = std::string(argv[1]);
  auto s = "../data/" + name;
  DrawableTrimesh<> m(s.c_str());
  uint nverts = m.num_verts();
  // m.normalize_bbox();
  // m.center_bbox();   
   
  // OUTPUT FIELDS
  uint nlevels = stoi(argv[2]); 
  vector<vector<double>> fields(nlevels,vector<double>(nverts));

  // GENERATE FIELD
  // MCF(m,1,0.0);
  Eigen::VectorXd f = Generate_field(m);
  // f *= 100000000;
  float clamp_limits[2];
  Set_clamp_limits(f, 1, clamp_limits); // set clamp limits to sigma

  // COMPUTE
  cout << "Computing discrete scale space: " << flush;
  double time_step = stod(argv[3]);
  vector<Eigen::VectorXd> efields = Build_discrete_scale_space(m,f,time_step,nlevels);
  for (auto i=0;i<efields.size();i++) {
    fields[i] = vector(efields[i].data(),efields[i].data()+efields[i].size());
    // progress_log(i);
  }
  cout << endl;

  cout << "Finding critical points: " << flush;
  vector<vector<int>> critical(nlevels,vector<int>(nverts));
  for (auto i=0;i<efields.size();i++) {
    cout << "Level " << i << ": ";
    critical[i] = Find_Critical_Points(m,efields[i]);
    cout << endl;
  }
  cout << endl;
  print_statistics(critical);

  // GUI
  GLcanvas gui;
  ScalarField phi;
  int selected_entry = 0;
  bool show_sf = false;
  bool show_cp = false;
  gui.show_side_bar = true;

  // bullets for critical points
  float point_size = m.edge_avg_length()/2;
  vector<DrawableSphere> points(nverts);
  for (uint i=0;i<nverts;i++)
    points[i]=DrawableSphere(m.vert(i),0.0,cinolib::Color::BLACK());
 
  gui.push(&m);
 
  gui.callback_app_controls = [&]() {
    if (ImGui::Checkbox("Show Scalar Field", &show_sf)) {
      if (show_sf) {
        phi = Rescale_field(fields[selected_entry],clamp_limits);
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
    if (ImGui::SliderFloat2("Clamp values", clamp_limits, f.minCoeff(), f.maxCoeff(),"%.4f",ImGuiSliderFlags_Logarithmic)) {
      if (show_sf) {
        phi = Rescale_field(fields[selected_entry],clamp_limits);
        phi.copy_to_mesh(m);
        m.updateGL();
      }
    }
    if (ImGui::SliderInt("Choose level", &selected_entry, 0, nlevels - 1)) {
      if (show_sf) {
        phi = Rescale_field(fields[selected_entry],clamp_limits);
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
              << std::setprecision(15) << fields[selected_entry][vid] 
              << " at level " << selected_entry << endl;
        // m.vert_data(vid).color = Color::RED();
        // m.updateGL();
      }
    }
    return false;
  };

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

vector<Eigen::VectorXd> Build_discrete_scale_space(const DrawableTrimesh<> &m1, const Eigen::VectorXd &f, 
                                                                    double time_scalar, const int levels) 
{
  DrawableTrimesh<> m(m1);
  // MCF(m,1);
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
    vec3d n = m.vert_data(vid).normal;      //vert_data(vid).normal;
    if (Hni.dot(n)<0) H(vid) = -H(vid); 
  }
  return H;
}

vector<int> Find_Critical_Points(const DrawableTrimesh<> &m, const Eigen::VectorXd &f)
{
  // -1 regular; 0 minimum; 1 maximum; k>1 (k-1)-saddle
  vector<int> buf(f.size());
  uint nv = m.num_verts();
  for(uint vid=0; vid<nv; vid++) {
    vector<uint> neigh = m.vert_ordered_verts_link(vid);
    int nn = neigh.size();
    vector<bool> sign(nn);    // true iff neighbor is smaller
    for (uint j=0;j<nn;j++) { // cycle on neighbors
      if (f(neigh[j])==f(vid)) {sign[j] = (vid>neigh[j]); 
      cout << "Tie: " << vid << ", " << neigh[j] << ", "; } // solve ties with vertex index
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


