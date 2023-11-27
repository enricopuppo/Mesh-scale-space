#include "utilities.h"
#include <cinolib/drawable_sphere.h>
#include <cinolib/geodesics.h>
#include <cinolib/gl/glcanvas.h>
#include <cinolib/gl/file_dialog_open.h>
#include <cinolib/gl/file_dialog_save.h>
#include <cinolib/gradient.h>
#include <cinolib/io/write_OBJ.h>
#include <cinolib/mean_curv_flow.h>
#include <fstream>
using namespace std;
using namespace cinolib;

//:::::::::::::::::::::::::::: GLOBAL VARIABLES (FOR GUI) ::::::::::::::::::::::::::::::
// Input
  DrawableTrimesh<> m;            // the input mesh
  uint nverts;                    // its #vertices
  vector<vector<uint>> VV;        // its VV relation
// Output
  vector<vector<double>> fields;  // the discrete scale-space
  uint nlevels;                   // # levels in the scale-space 
  vector<float*> clamp_limits;    // clamp limits for field visualization at all levels
// Processing
  Eigen::VectorXd f;              // base field on m
  double t_step;                  // base time step of diffusion
  double mult;                    // multiplicator/stride of time step in diffusion
  vector<vector<int>> critical;   // critical points at all levels
// GUI
  ScalarField phi;
  int selected_entry = 0;
  float curr_clamp[2];
  bool show_sf = false;
  bool show_cp = false;
  bool show_m = true;
  bool show_wf = false;
  vector<DrawableSphere> points;  // spheres for rendering critical points
  float point_size;               // base radius of spheres
  float point_multiplier = 1.0;   // multiplier of radius
 



//::::::::::::::::::::::::::::::::::::GUI utilities ::::::::::::::::::::::::::::::::::::

// functions to  render vertices as spheres
inline void draw_points(const vector<DrawableSphere> &cp, GLcanvas &gui) {
  for (auto &point : cp) //if (point.radius > 0) 
    gui.push(&point, false);
}

inline void remove_points(const vector<DrawableSphere> &cp, GLcanvas &gui) {
  for (auto &point : cp) gui.pop(&point);
}

inline void set_points(const DrawableTrimesh<> &m, const vector<int> &c, vector<DrawableSphere> &cp, float s)
{
  cp.resize(0);
  DrawableSphere buf;
  buf.radius = s;
  for (uint i=0;i<c.size();i++) {
    if (c[i]==-1) continue;
    if (c[i]==0) buf.color = cinolib::Color::BLUE();
    else if (c[i]==1) buf.color = cinolib::Color::RED();
    else buf.color = cinolib::Color::GREEN();
    buf.center = m.vert(i);
    cp.push_back(buf);
  }
}

inline void reset_points(vector<DrawableSphere> &cp)
{
  for (auto &point : cp) {
    point.radius = 0.0;
    point.color = cinolib::Color::BLACK();
  }
}

// ------------
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

//::::::::::::::::::::::::::::::::: GUI ::::::::::::::::::::::::::::::::::::::::::::::::

GLcanvas Init_GUI()
{
  GLcanvas gui(1500,700);
  gui.side_bar_width = 0.25;
  gui.show_side_bar = true;
  point_size = m.edge_avg_length()/2; // set initial radius of spheres for critical points
  return gui;
}

void Setup_GUI_Callbacks(GLcanvas & gui)
{
  gui.callback_app_controls = [&]() {
    ImGui::SeparatorText("Files");
    if (ImGui::Button("Load mesh")) {
      string filename = file_dialog_open();
    }
    ImGui::SameLine();
    if (ImGui::Button("Save scale-space")){
      string filename = file_dialog_save();
    }

    ImGui::SeparatorText("Field");

    ImGui::SeparatorText("Processing");

    ImGui::SeparatorText("View");
    if (ImGui::Checkbox("Show wireframe", &show_wf)) {
      if (show_wf) m.show_wireframe(true);
      else m.show_wireframe(false);
      m.updateGL();
    } 
    ImGui::SameLine();
    if (ImGui::Checkbox("Show mesh", &show_m)) {
      if (show_m) m.show_mesh(true);
      else m.show_mesh(false);
      m.updateGL();
    } 
    if (ImGui::Checkbox("Show Scalar Field", &show_sf)) {
      if (show_sf) {
        curr_clamp[0]=clamp_limits[selected_entry][0];
        curr_clamp[1]=clamp_limits[selected_entry][1];
        phi = Clamp_and_rescale_field(fields[selected_entry],curr_clamp);
        phi.copy_to_mesh(m);
        m.show_texture1D(TEXTURE_1D_HSV);
      } else {
        m.show_poly_color();
      }
    } 
    ImGui::SameLine();
    if (ImGui::Checkbox("Show Critical Points", &show_cp)) {
      if (show_cp) {
        set_points(m,critical[selected_entry],points,point_size*point_multiplier);
        draw_points(points,gui);
      } else 
        remove_points(points,gui);
      // m.updateGL();
    } 
    if (ImGui::SliderFloat("Point size", &point_multiplier, 0, 10.0, "%.1f")) {
      if (show_cp) {
        set_points(m,critical[selected_entry],points,point_size*point_multiplier);
        m.updateGL();
      }
    }
    if (ImGui::SliderFloat2("Clamp values", curr_clamp, f.minCoeff(), f.maxCoeff(),"%.4f",ImGuiSliderFlags_Logarithmic)) {
  //  if (ImGui::InputFloat2("Clamp values", curr_clamp,"%.4f")) {
      if (show_sf) {
        phi = Clamp_and_rescale_field(fields[selected_entry],curr_clamp);
        phi.copy_to_mesh(m);
        m.updateGL();
      }
    }
    // if (ImGui::SliderInt("Choose level", &selected_entry, 0, nlevels - 1)) {
    if (ImGui::InputInt("Level", &selected_entry)) {
      if (selected_entry>=nlevels) selected_entry = nlevels-1;
      else {
        if (show_sf) {
          curr_clamp[0]=clamp_limits[selected_entry][0];
          curr_clamp[1]=clamp_limits[selected_entry][1];
          phi = Clamp_and_rescale_field(fields[selected_entry],curr_clamp);
          phi.copy_to_mesh(m);
        }
        if (show_cp) {
          remove_points(points,gui);
          set_points(m,critical[selected_entry],points,point_size*point_multiplier);
          draw_points(points,gui);
        }
        if (show_sf || show_cp) m.updateGL();
      }
    }
    // JUST AN EXAMPLE OF BUTTON
    // if (ImGui::Button("Dummy button 2")) {
    //   auto name0 = "CMFC" + name + ".obj";
    //   write_OBJ(name0.c_str(), obj_wrapper(m.vector_verts()),
    //             m.vector_polys());
    // }
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

// rescale all values of vector to [0,1]
void normalize_in_01(Eigen::VectorXd &f)
{
    long double min = f.minCoeff();
    long double max = f.maxCoeff();
    long double delta = max - min;
    for(int i=0;i<f.size();i++) f[i] = (double)((f[i]-min) / delta);
}

//==================== FIELD GENERATORS ========================

// Gaussian curvature (angle defect)
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

// Mean curvature (Laplacian)
Eigen::VectorXd Mean_curvature(const DrawableTrimesh<> & m)
{
  uint nv = m.num_verts();
  Eigen::SparseMatrix<double> L  = laplacian(m, COTANGENT);
  Eigen::SparseMatrix<double> M = mass_matrix(m);
  InvertSparseMatrix(M);
  L = M * L;
  Eigen::MatrixXd V(nv,3);
  for(uint vid=0; vid<nv; ++vid) {
      vec3d pos = m.vert(vid);
      V(vid,0) = pos.x();
      V(vid,1) = pos.y();
      V(vid,2) = pos.z();
  }
  Eigen::MatrixXd Hn = L * V;
  Hn *= -0.5;
  Eigen::VectorXd H(nv);
  for(uint vid=0; vid<nv; ++vid) {
    H(vid)=Hn.row(vid).norm();
    vec3d Hni(Hn(vid,0),Hn(vid,1),Hn(vid,2));
    vec3d n = m.vert_data(vid).normal;
    if (Hni.dot(n)<0) H(vid) = -H(vid); 
  }
  return H;
}

// Eigenfunctions of the Laplacian
Eigen::VectorXd Laplacian_eigenfunction(const DrawableTrimesh<> & m, int maxe, int k)
// compute maxe eigenfunctions and return the k-th
// warning: result changes depending on the value of maxe>=k
{
  uint nv = m.num_verts();
  vector<double> eig;
  vector<double> f_min;
  vector<double> f_max;
  Eigen::SparseMatrix<double> L = laplacian(m, COTANGENT);
  matrix_eigenfunctions(L, true, maxe, eig, f_min, f_max);
  Eigen::VectorXd buf(nv);
  for (auto i=0;i<nv;i++) 
    buf[i]=eig[(k-1)*nv+i]; 
  return buf;
}

Eigen::VectorXd Coordinate(const DrawableTrimesh<> & m, int coord)
// return coordinate of vertices selected by coord: 0 -> x, 1 -> y, 2 ->z
{
  uint nv = m.num_verts();
  Eigen::VectorXd buf(nv);
  for (auto i=0;i<nv;i++) 
    buf[i]=m.vert(i)[coord]; 
  return buf;
   
}

Eigen::VectorXd Random_field(const DrawableTrimesh<> & m)
{
  uint nv = m.num_verts();
  Eigen::VectorXd buf(nv);
  srand(time(NULL));
  for (auto i=0;i<nv;i++) 
    buf[i]=(double)rand()/RAND_MAX; 
  return buf;
   
}



//=============================== INPUT FIELD ==================================
Eigen::VectorXd Generate_field(const DrawableTrimesh<> &m)
{
  Eigen::VectorXd buf;
  // buf = Mean_curvature(m);
  buf = gaussian_curvature(m);
  // buf =  Laplacian_eigenfunction(m,100,10);
  // buf = Coordinate(m,1);
  // buf = Random_field(m);
  // normalize_in_01(buf);
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
  cl[0] = std::max(mean - sigma_multiplier * sigma,f.minCoeff());
  cl[1] = std::min(mean + sigma_multiplier * sigma,f.maxCoeff());
  // cout << "clamp limits: " << cl[0] << ", " << cl[1] << endl;
}

//=========================== PROCESSING FUNCTIONS =============================

vector<Eigen::VectorXd> Build_disc_ss(const DrawableTrimesh<> &m, const Eigen::VectorXd &f, 
                                            int levels, double time_scalar = 0.01, double mult = 1, 
                                            bool linear = true, bool normalize = false) 
// compute a discrete scale-space with "levels" levels from input field f on shape m
// if linear applies diffusion flow levels * mult times with steady coefficient time_scalar
// else applies diffusion flow "levels" times with exponentially increasing diffusion coefficient:
// initial coefficient time_scalar is multiplied by mult at each iteration
// cumulative smoothing: input field at iteration i is the smoothed field at iteration i-1
// if normalize the result is normalizd in [0,1] at each iteration
{
  Eigen::SparseMatrix<double> L  = laplacian(m, COTANGENT);
  Eigen::SparseMatrix<double> MM = mass_matrix(m);
  vector<Eigen::VectorXd> buf(levels);
  Eigen::VectorXd f1(f);
  if (normalize) normalize_in_01(f1);
  buf[0]=f1;

  if (linear) cout << "linear progression\n"; else cout << "exponential progression\n";

  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> LLT(MM - time_scalar * L);
  for (auto i=1;i<levels;i++) {
    // iterated backward Euler time integration of heat flow equation
    if (linear) {
      int nit = (int)mult;
      for (int j=0;j<nit;j++) f1 = LLT.solve(MM * f1);
      buf[i] = f1;
    } else {
      if (i>0) Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> LLT(MM - time_scalar * L);
      buf[i] = LLT.solve(MM * buf[i-1]);
      time_scalar *= mult;
    }
    if (normalize) normalize_in_01(buf[i]);
    cout << "level "<< i << " completed\n";
  }
  return buf;
}

vector<int> Find_Critical_Points(const DrawableTrimesh<> &m, const vector<vector<uint>> &VV,
                                          const Eigen::VectorXd &f, vector<vector<int>> &ties)
// find critical points of field f on shape m
// return a vector indexed on vertices of m: -1 regular; 0 minimum; 1 maximum; k>1 (k-1)-saddle
// VV is vector of vertex-vertex topological relations of m
// ties reports edges with equal field values at both endpoints, if any, as pairs of vertex indices 
{
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
// print the number and types of critical points at all levels of the discrete ss
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
  if (argc<4) {cout << "Usage: Mesh_scale_space filename num_levels time_step multiplier/stride\n"; return 1;}

  //INPUT MESH::::::::::::::::::::::::::::::::::::
  string s = "../data/" + string(argv[1]);
  m = DrawableTrimesh<>(s.c_str());
  nverts = m.num_verts();
  VV.resize(nverts); // fill in Vertex-Vertex relation
  for (auto i=0;i<nverts;i++) VV[i]=m.vert_ordered_verts_link(i);
  // uncomment the following and adjust parameters if you want a smoother mesh
  // MCF(m,12,1e-5,true);
  m.normalize_bbox(); // rescale mesh to fit [0,1]^3 box
  m.updateGL();     

  // OUTPUT FIELDS::::::::::::::::::::::::::::::::
  nlevels = stoi(argv[2]); 
  fields = vector<vector<double>>(nlevels,vector<double>(nverts));
  clamp_limits = vector<float*>(nlevels);
  for (auto &l : clamp_limits) l = new float[2];

  // GENERATE FIELD:::::::::::::::::::::::::::::::
  // change this function if you want to generate a different field
  f = Generate_field(m);

  // COMPUTE DISCRETE SCALE SPACE:::::::::::::::::
  cout << "Computing discrete scale space: ";
  t_step = stod(argv[3]);
  mult = stod(argv[4]);
  vector<Eigen::VectorXd> efields = Build_disc_ss(m,f,nlevels,t_step,mult,true,true);
  for (auto i=0;i<efields.size();i++) // convert from Eigen to std::vector
    fields[i] = vector(efields[i].data(),efields[i].data()+efields[i].size());
  cout << "done"<< endl;

  for (auto i=0;i<nlevels;i++) // set clamp limits for visualization
    Set_clamp_limits(efields[i], 2, clamp_limits[i]); 

  cout << "Finding critical points: \n";
  critical = vector<vector<int>>(nlevels,vector<int>(nverts));
  vector<vector<int>> ties; // "flat" edges
  for (auto i=0;i<efields.size();i++) {
    critical[i] = Find_Critical_Points(m,VV,efields[i],ties);
    if (ties.size()>0) {    // report ties, if any
      cout << "Found ties at level " << i << ": ";
      for (auto i=0;i<ties.size();i++) 
        cout << "(" << ties[i][0] << "," << ties[i][1] << "), ";
      cout << endl;
    }
    ties.resize(0);
  }
  cout << "done"<< endl;
  print_statistics(critical);


  // setup GUI
  GLcanvas gui = Init_GUI();
  Setup_GUI_Callbacks(gui);

  // render the mesh
  gui.push(&m);
  m.show_wireframe(false);
 
  return gui.launch();
}
