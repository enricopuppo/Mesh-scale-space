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
// Global state
enum field_methods {GAUSS, MEAN, L_EIGEN, COORDX, COORDY, COORDZ, RANDOM};

struct State {
  bool MESH_IS_LOADED, FIELD_IS_PRESENT, SCALE_SPACE_IS_PRESENT;
  // Input
  string filename;               // name of input file
  DrawableTrimesh<> m;            // the input mesh
  uint nverts;                    // its #vertices
  vector<vector<uint>> VV;        // its VV relation
  // Output
  vector<vector<double>> fields;  // the discrete scale-space
  uint nlevels;                   // # levels in the scale-space 
  vector<float*> clamp_limits;    // clamp limits for field visualization at all levels
  // Field generation
  Eigen::VectorXd f;              // base field on m
  int current_field_method;       // current method to generate the field
  vector<string> field_method_labels; // labels for the combo box
  int max_eigenfunctions, selected_eigenfunction; // parameters for eigenfunction generator
  // Scale space
  double t_step;                  // base time step of diffusion
  double mult;                    // multiplicator/stride of time step in diffusion
  vector<vector<int>> critical;   // critical points at all levels
  // GUI
  int selected_entry;
  float curr_clamp[2];
  bool show_sf, show_cp, show_m, show_wf;
  vector<DrawableSphere> points;          // spheres for rendering critical points
  float point_size, point_multiplier;     // base radius and multiplier of spheres

  State() {
    MESH_IS_LOADED = FIELD_IS_PRESENT = SCALE_SPACE_IS_PRESENT = false;
    show_sf = show_cp = show_wf = false;
    show_m = true;
    // field generation
    current_field_method = GAUSS;
    field_method_labels = {"Gaussian curvature", "Mean curvature", "Laplacian eigenfunction", "Coordinate x", "Coordinate y", "Coordinate z", "Random"};
    max_eigenfunctions = 100;
    selected_eigenfunction = 1;
    // view
    selected_entry = 0;
    point_multiplier = 1.0;
  }
};

//::::::::::::::::::::::::::::::::::::GUI utilities ::::::::::::::::::::::::::::::::::::

void Load_mesh(State &gs)
{
  string filename = file_dialog_open();
  if (filename.size()!=0) {
    gs.m = DrawableTrimesh<>(filename.c_str());
    gs.nverts = gs.m.num_verts();
    gs.VV.resize(gs.nverts); // fill in Vertex-Vertex relation
    for (auto i=0;i<gs.nverts;i++) gs.VV[i]=gs.m.vert_ordered_verts_link(i);
    // uncomment the following and adjust parameters if you want a smoother mesh
    // MCF(m,12,1e-5,true);
    gs.m.normalize_bbox(); // rescale mesh to fit [0,1]^3 box
    gs.m.center_bbox();
    gs.m.show_wireframe(gs.show_wf);
    gs.m.updateGL();  
    gs.MESH_IS_LOADED = true;
    gs.FIELD_IS_PRESENT = false;
    gs.SCALE_SPACE_IS_PRESENT = false; 
    gs.point_size = gs.m.edge_avg_length()/2; // set initial radius of spheres for critical points
  }
}

void Save_scale_space(const State &gs)
{
  string filename = file_dialog_save();
  if (filename.size()==0) return;
  ofstream f(filename.c_str());
  // if (!f) {ImGui::OpenPopup("Output error"); return;}
  f << gs.nverts << " " << gs.nlevels << endl;
  for (uint i=0;i<gs.nlevels;i++) {
    for (uint j=0;j<gs.nverts;j++) f << gs.fields[i][j] << " ";
    f << endl;
  }
  f.close();
}

//-------------
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

const char* wrap_strings(const vector<string> &s)
{
  string ss;
  for (int i=0;i<s.size();i++) ss = ss + "\0" + s[i];
  ss += "\0";
  return ss.c_str();  
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
void Generate_field(State & gs)
{
  switch(gs.current_field_method) {
    case GAUSS: gs.f = gaussian_curvature(gs.m); break;
    case MEAN: gs.f = Mean_curvature(gs.m); break;
    case L_EIGEN: gs.f = Laplacian_eigenfunction(gs.m,gs.max_eigenfunctions,gs.selected_eigenfunction); break;
    case COORDX:
    case COORDY:
    case COORDZ:
    case RANDOM:
    default: cout << "Bad case in switch - This shouldn't happen!\n";
  }
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


//::::::::::::::::::::::::::::::::: GUI ::::::::::::::::::::::::::::::::::::::::::::::::

GLcanvas Init_GUI()
{
  GLcanvas gui(1500,700);
  gui.side_bar_width = 0.25;
  gui.show_side_bar = true;
  return gui;
}

void Setup_GUI_Callbacks(GLcanvas & gui, State &gs)
{
  gui.callback_app_controls = [&]() {
    // Files
    ImGui::SeparatorText("Files");
    if (ImGui::Button("Load mesh")) {
      if (gs.MESH_IS_LOADED) {
        ImGui::OpenPopup("Load mesh?");
      } else {
        Load_mesh(gs);
      }
    }
    // Modal popup for loading files
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center,ImGuiCond_Appearing,ImVec2(0.5f,0.5f));
    if (ImGui::BeginPopupModal("Load mesh?", NULL, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse))
    {
      ImGui::Text("All data structures will be reset - Load mesh anyway?\n\n");
      ImGui::Separator();           
      static bool dont_ask_me_next_time = false;
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0,0));
      ImGui::Checkbox("Don't ask me next time", &dont_ask_me_next_time);
      ImGui::PopStyleVar();
      if (ImGui::Button("OK", ImVec2(120,0))) {Load_mesh(gs); ImGui::CloseCurrentPopup();}
      ImGui::SameLine();
      if (ImGui::Button("Cancel", ImVec2(120,0))) {ImGui::CloseCurrentPopup();}
      ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Save scale-space")) Save_scale_space(gs);

    // Field
    ImGui::SeparatorText("Field");
    ImGui::Combo("Method",&gs.current_field_method,wrap_strings(gs.field_method_labels));
    if (ImGui::Button("Generate field")) Generate_field(gs);

    // Processing
    ImGui::SeparatorText("Processing");

    // View
    ImGui::SeparatorText("View");
    if (ImGui::Checkbox("Show wireframe", &gs.show_wf)) {
      if (gs.show_wf) gs.m.show_wireframe(true);
      else gs.m.show_wireframe(false);
      gs.m.updateGL();
    } 
    ImGui::SameLine();
    if (ImGui::Checkbox("Show mesh", &gs.show_m)) {
      if (gs.show_m) gs.m.show_mesh(true);
      else gs.m.show_mesh(false);
      gs.m.updateGL();
    } 
    if (ImGui::Checkbox("Show Scalar Field", &gs.show_sf)) {
      if (gs.show_sf) {
        gs.curr_clamp[0]=gs.clamp_limits[gs.selected_entry][0];
        gs.curr_clamp[1]=gs.clamp_limits[gs.selected_entry][1];
        ScalarField phi = Clamp_and_rescale_field(gs.fields[gs.selected_entry],gs.curr_clamp);
        phi.copy_to_mesh(gs.m);
        gs.m.show_texture1D(TEXTURE_1D_HSV);
      } else {
        gs.m.show_poly_color();
      }
    } 
    ImGui::SameLine();
    if (ImGui::Checkbox("Show Critical Points", &gs.show_cp)) {
      if (gs.show_cp) {
        set_points(gs.m,gs.critical[gs.selected_entry],gs.points,gs.point_size*gs.point_multiplier);
        draw_points(gs.points,gui);
      } else 
        remove_points(gs.points,gui);
      // m.updateGL();
    } 
    if (ImGui::SliderFloat("Point size", &gs.point_multiplier, 0, 10.0, "%.1f")) {
      if (gs.show_cp) {
        set_points(gs.m,gs.critical[gs.selected_entry],gs.points,gs.point_size*gs.point_multiplier);
        gs.m.updateGL();
      }
    }
    if (ImGui::SliderFloat2("Clamp values", gs.curr_clamp, gs.f.minCoeff(), gs.f.maxCoeff(),"%.4f",ImGuiSliderFlags_Logarithmic)) {
  //  if (ImGui::InputFloat2("Clamp values", curr_clamp,"%.4f")) {
      if (gs.show_sf) {
        ScalarField phi = Clamp_and_rescale_field(gs.fields[gs.selected_entry],gs.curr_clamp);
        phi.copy_to_mesh(gs.m);
        gs.m.updateGL();
      }
    }
    // if (ImGui::SliderInt("Choose level", &selected_entry, 0, nlevels - 1)) {
    if (ImGui::InputInt("Level", &gs.selected_entry)) {
      if (gs.selected_entry>=gs.nlevels) gs.selected_entry = gs.nlevels-1;
      else {
        if (gs.show_sf) {
          gs.curr_clamp[0]=gs.clamp_limits[gs.selected_entry][0];
          gs.curr_clamp[1]=gs.clamp_limits[gs.selected_entry][1];
          ScalarField phi = Clamp_and_rescale_field(gs.fields[gs.selected_entry],gs.curr_clamp);
          phi.copy_to_mesh(gs.m);
          gs.m.updateGL();
        }
        if (gs.show_cp) {
          remove_points(gs.points,gui);
          set_points(gs.m,gs.critical[gs.selected_entry],gs.points,gs.point_size*gs.point_multiplier);
          draw_points(gs.points,gui);
        }
      }
    }
   };

  gui.callback_mouse_left_click = [&](int modifiers) -> bool {
    if (modifiers & GLFW_MOD_SHIFT) {
      vec3d p;
      vec2d click = gui.cursor_pos();
      if (gui.unproject(click, p)) {
        uint vid = gs.m.pick_vert(p);
        cout << "Picked vertex " << vid << " field value " 
              << std::setprecision(20) << gs.fields[gs.selected_entry][vid] 
              << " at level " << gs.selected_entry << endl;
        // m.vert_data(vid).color = Color::RED();
        // m.updateGL();
      }
    }
    return false;
  };
}


//=============================== MAIN =========================================

int main(int argc, char **argv) {
  if (argc<4) {cout << "Usage: Mesh_scale_space filename num_levels time_step multiplier/stride\n"; return 1;}

  //SETUP GLOBAL STATE::::::::::::::::::::::::::::
  State gs;

  //INPUT MESH::::::::::::::::::::::::::::::::::::
  string s = "../data/" + string(argv[1]);
  gs.m = DrawableTrimesh<>(s.c_str());
  gs.nverts = gs.m.num_verts();
  gs.VV.resize(gs.nverts); // fill in Vertex-Vertex relation
  for (auto i=0;i<gs.nverts;i++) gs.VV[i]=gs.m.vert_ordered_verts_link(i);
  // uncomment the following and adjust parameters if you want a smoother mesh
  // MCF(m,12,1e-5,true);
  gs.m.normalize_bbox(); // rescale mesh to fit [0,1]^3 box
  gs.m.center_bbox();
  gs.m.show_wireframe(false);
  gs.m.updateGL();  
  gs.point_size = gs.m.edge_avg_length()/2; // set initial radius of spheres for critical points
  gs.MESH_IS_LOADED = true;   

  // OUTPUT FIELDS::::::::::::::::::::::::::::::::
  gs.nlevels = stoi(argv[2]); 
  gs.fields = vector<vector<double>>(gs.nlevels,vector<double>(gs.nverts));
  gs.clamp_limits = vector<float*>(gs.nlevels);
  for (auto &l : gs.clamp_limits) l = new float[2];

  // GENERATE FIELD:::::::::::::::::::::::::::::::
  // change this function if you want to generate a different field
  Generate_field(gs);

  // COMPUTE DISCRETE SCALE SPACE:::::::::::::::::
  cout << "Computing discrete scale space: ";
  gs.t_step = stod(argv[3]);
  gs.mult = stod(argv[4]);
  vector<Eigen::VectorXd> efields = Build_disc_ss(gs.m,gs.f,gs.nlevels,gs.t_step,gs.mult,true,true);
  for (auto i=0;i<efields.size();i++) // convert from Eigen to std::vector
    gs.fields[i] = vector(efields[i].data(),efields[i].data()+efields[i].size());
  gs.SCALE_SPACE_IS_PRESENT = true;
  cout << "done"<< endl;

  for (auto i=0;i<gs.nlevels;i++) // set clamp limits for visualization
    Set_clamp_limits(efields[i], 2, gs.clamp_limits[i]); 

  cout << "Finding critical points: \n";
  gs.critical = vector<vector<int>>(gs.nlevels,vector<int>(gs.nverts));
  vector<vector<int>> ties; // "flat" edges
  for (auto i=0;i<efields.size();i++) {
    gs.critical[i] = Find_Critical_Points(gs.m,gs.VV,efields[i],ties);
    if (ties.size()>0) {    // report ties, if any
      cout << "Found ties at level " << i << ": ";
      for (auto i=0;i<ties.size();i++) 
        cout << "(" << ties[i][0] << "," << ties[i][1] << "), ";
      cout << endl;
    }
    ties.resize(0);
  }
  cout << "done"<< endl;
  print_statistics(gs.critical);


  // setup GUI
  GLcanvas gui = Init_GUI();
  Setup_GUI_Callbacks(gui,gs);

  // render the mesh
  gui.push(&gs.m);
  
  return gui.launch();
}
