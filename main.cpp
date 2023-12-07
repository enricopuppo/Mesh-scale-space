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
const char * FIELD_METHOD_LABELS[] = {"Gaussian curvature", "Mean curvature", "Laplacian eigenfunction", "Coordinate x", "Coordinate y", "Coordinate z", "Random"};

struct State {
  // Program state ::::::::::::::::::::::::::::::::::::::::::::::::::::::
  bool MESH_IS_LOADED, FIELD_IS_PRESENT, SCALE_SPACE_IS_PRESENT, EIGENFUNCTIONS_COMPUTED;
  // Input
  DrawableTrimesh<> m;            // the input mesh
  uint nverts;                    // its #vertices
  vector<vector<uint>> VV;        // its VV relation
  // Field
  Eigen::VectorXd f;              // base field on m
  float f_clamp[2];               // clamp limits of base field
  vector<double> eigenfunctions;  // eigenfunctions of f as computed by SPECTRA
  // Scale-space
  vector<vector<double>> fields;  // the discrete scale-space
  vector<vector<int>> critical;   // critical points at all levels
  vector<DrawableSphere> points;  // spheres for rendering critical points
  float point_size;               // base radius of spheres
  // GUI state ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  // Field
  int current_field_method;       // current method to generate the field
  bool normalize_f;               // base field is normalized
  int max_eigenfunctions, selected_eigenfunction; // parameters for eigenfunction generator
  // Scale space
  bool normalize;                 // fields are normalized at all levels during diffusion
  int nlevels;                    // # levels in the scale-space 
  int method;                     // 1 diffusion; 2 smoothness optiization
  int diff_progression;           // 1 linear - 2 exponential
  float diff_lambda;              // base time step of diffusion
  int diff_stride;                // stride for linear method
  float diff_mult;                // multiplicator for exponential method
  int opt_progression;            // 1 linear - 2 exponential
  float opt_w;                    // base regularization of optimization
  int opt_stride;                 // stride for linear method
  float opt_div;                  // divisor for exponential method
  // View
  bool show_m, show_wf, show_cp;  // show mesh, wire-frame, critical points
  int show_field;                 // field to show: 0 no field; 1 base field; 2 scale-space field
  float point_multiplier;         // multiplier of sphere radius
  vector<float*> clamp_limits;    // clamp limits for field visualization at all levels
  float curr_clamp[2];            // current clamp limits
  int selected_entry;             // selected level for visualization

  State() {
    MESH_IS_LOADED = FIELD_IS_PRESENT = SCALE_SPACE_IS_PRESENT = false;
    EIGENFUNCTIONS_COMPUTED = false;
    // field generation
    current_field_method = GAUSS;
    normalize_f = false;
    max_eigenfunctions = 100;
    selected_eigenfunction = 1;
    f_clamp[0] = 0; f_clamp[1] = 1;
    // scale-space
    normalize = true;
    nlevels = 300;
    method = 1;           //diffusion
    diff_progression = 1;  //linear
    diff_lambda = 0.0001;
    diff_stride = 10;
    diff_mult = 1.05;
    opt_progression = 1;  //linear
     opt_w = 1e6;
    opt_stride = 10;
    opt_div = 1.05;
    // view
    show_m = true;
    show_cp = show_wf = false;
    show_field = 0;
    point_multiplier = 1.0;
    f.resize(2); f(0)=0.0; f(1)=1.0;  // init f to support clamp limits in gui
    curr_clamp[0]=0.0; curr_clamp[1]=1.0;
    selected_entry = 0;
  }
};

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

//::::::::::::::::::::::::::::::::::::I/O ::::::::::::::::::::::::::::::::::::

void Load_mesh(string filename, GLcanvas & gui, State &gs)
{
  gs.m = DrawableTrimesh<>(filename.c_str());
  gs.nverts = gs.m.num_verts();
  gs.VV.resize(gs.nverts); // fill in Vertex-Vertex relation
  for (auto i=0;i<gs.nverts;i++) gs.VV[i]=gs.m.vert_ordered_verts_link(i);
  // uncomment the following and adjust parameters if you want a smoother mesh
  // MCF(m,12,1e-5,true);
  gs.m.normalize_bbox(); // rescale mesh to fit [0,1]^3 box
  gs.m.center_bbox();
  gs.m.show_wireframe(gs.show_wf);
  gs.m.show_mesh(gs.show_m);    
  gs.m.updateGL();  
  gs.point_size = gs.m.edge_avg_length()/2; // set initial radius of spheres for critical points
  if (!gs.MESH_IS_LOADED) {
    gui.push(&gs.m);
    gs.MESH_IS_LOADED = true;
  }

  // reset field and scale-space
  gs.f.resize(2); gs.f(0)=0.0; gs.f(1)=1.0; // init f to support clamp limits in gui
  gs.curr_clamp[0]=0.0; gs.curr_clamp[1]=1.0;
  gs.show_field = 0;

  if (gs.show_cp) {
    remove_points(gs.points,gui);
    gs.show_cp = false;
  }

  gs.clamp_limits = vector<float*>(1); // only entry zero for the base field
  gs.clamp_limits[0] = new float[2];

  gs.FIELD_IS_PRESENT = gs.SCALE_SPACE_IS_PRESENT = gs.EIGENFUNCTIONS_COMPUTED = false; 
}

void Load_mesh(GLcanvas & gui, State &gs)
{
  string filename = file_dialog_open();
  if (filename.size()!=0) Load_mesh(filename,gui,gs);
}

void Save_scale_space(const State &gs)
{
  string filename = file_dialog_save();
  if (filename.size()==0) return;
  ofstream f(filename.c_str());
  f << gs.nverts << " " << gs.fields.size() << endl;
  for (uint i=0;i<gs.nlevels;i++) {
    for (uint j=0;j<gs.nverts;j++) f << gs.fields[i][j] << " ";
    f << endl;
  }
  f.close();
}

void Export_critical_points(const State &gs)
{
  string filename = file_dialog_save();
  if (filename.size()==0) return;
  ofstream f(filename.c_str());
  for (uint i=0;i<gs.nverts;i++)
    if (gs.critical[gs.selected_entry][i]!=-1) 
      f << gs.critical[gs.selected_entry][i] << " " << i << endl;
  f.close();
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

Eigen::SparseMatrix<double> Invert_diag_matrix(const Eigen::SparseMatrix<double> &M) 
{
  Eigen::SparseMatrix<double> M_inv;
  // http://www.alecjacobson.com/weblog/?p=2552
  if (&M_inv != &M) M_inv = M;
  // Iterate over outside
  for (int k = 0; k < M_inv.outerSize(); ++k) {
    // Iterate over inside
    for (typename Eigen::SparseMatrix<double>::InnerIterator it(M_inv, k); it; ++it) {
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
Eigen::VectorXd Laplacian_eigenfunction(State &gs)
// compute maxe eigenfunctions and return the k-th
// warning: result changes depending on the value of maxe>=k
{
  uint nv = gs.m.num_verts();
  Eigen::VectorXd buf(nv);
  if (!gs.EIGENFUNCTIONS_COMPUTED || gs.eigenfunctions.size() < nv*gs.max_eigenfunctions) {
    cout << "Computing eigenfunctions...\n";
    vector<double> f_min;
    vector<double> f_max;
    Eigen::SparseMatrix<double> L = laplacian(gs.m, COTANGENT);
    matrix_eigenfunctions(L, true, gs.max_eigenfunctions, gs.eigenfunctions, f_min, f_max);
    gs.EIGENFUNCTIONS_COMPUTED = true;
    cout << "done!\n";
  }
  for (auto i=0;i<nv;i++) 
    buf[i]=gs.eigenfunctions[gs.selected_eigenfunction*nv+i]; 
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
void Set_clamp_limits(const Eigen::VectorXd &f, int sigma_multiplier, float cl[]) 
{
  // set clamp limits to sigma_multiplier * sigma
  double mean = f.sum()/f.size();
  Eigen::VectorXd s(f);
  for (auto i=0;i<s.size();i++) s(i) = (s(i)-mean)*(s(i)-mean);
  double sigma = sqrt(s.sum()/s.size());
  cl[0] = std::max(mean - sigma_multiplier * sigma,f.minCoeff());
  cl[1] = std::min(mean + sigma_multiplier * sigma,f.maxCoeff());
}

void Generate_field(GLcanvas & gui, State & gs)
{
  switch(gs.current_field_method) {
    case GAUSS: gs.f = gaussian_curvature(gs.m); break;
    case MEAN: gs.f = Mean_curvature(gs.m); break;
    case L_EIGEN: gs.f = Laplacian_eigenfunction(gs); break;
    case COORDX: gs.f = Coordinate(gs.m,0); break;
    case COORDY: gs.f = Coordinate(gs.m,1); break;
    case COORDZ: gs.f = Coordinate(gs.m,2); break;
    case RANDOM: gs.f = Random_field(gs.m); break;
    default: cout << "Bad case in switch - This shouldn't happen!\n";
  }
  if (gs.normalize_f) normalize_in_01(gs.f);
  Set_clamp_limits(gs.f, 2, gs.f_clamp);
  gs.FIELD_IS_PRESENT = true;

  if (gs.show_cp) {
    remove_points(gs.points,gui);
    gs.show_cp = false;
  }
  gs.SCALE_SPACE_IS_PRESENT = false;
  if (gs.show_field==2) gs.show_field=1;
  if (gs.show_field==1) {
    gs.curr_clamp[0]=gs.f_clamp[0];
    gs.curr_clamp[1]=gs.f_clamp[1];
    ScalarField phi = Clamp_and_rescale_field(vector(gs.f.data(),gs.f.data()+gs.f.size()),gs.curr_clamp);
    phi.copy_to_mesh(gs.m);
    gs.m.show_texture1D(TEXTURE_1D_HSV);
  }
}

//=========================== SCALE-SPACE FUNCTIONS =============================

vector<int> Find_Critical_Points(const DrawableTrimesh<> &m, const vector<vector<uint>> &VV,
                                          const Eigen::VectorXd &f, vector<vector<int>> &ties)
// find critical points of field f on shape m
// return a vector indexed on vertices of m: -1 regular; 0 minimum; 1 maximum; k>1 (k-1)-saddle
// VV is vector of vertex-vertex topological relations of m
// ties reports edges with equal field values at both endpoints, if any, as pairs of vertex indices 
{
  vector<int> buf(f.size());
  uint nv = m.num_verts();
  ties.resize(0);
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

vector<Eigen::VectorXd> Diffusion_flow(const DrawableTrimesh<> &m, const Eigen::VectorXd &f,
                                  int nlevels, float diff_lambda = 0.0001, int stride=10, float mult = 1.05,
                                  bool linear=true, bool normalize=false)
{
  Eigen::SparseMatrix<double> L  = laplacian(m, COTANGENT);
  Eigen::SparseMatrix<double> MM = mass_matrix(m);
  vector<Eigen::VectorXd> buf(nlevels);
  Eigen::VectorXd f1(f);

  cout << "Diffusion: ";
  if (linear) cout << "linear progression "; else cout << "exponential progression  ";
  float step = diff_lambda;
  if (normalize) {normalize_in_01(f1); cout << "with normalization...\n";} 
  else cout << "without normalization..\n";
  buf[0]=f1;
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> LLT(MM - step * L);
  for (auto i=1;i<nlevels;i++) {
    // iterated backward Euler time integration of heat flow equation
    if (linear) {
      for (int j=0;j<stride;j++) f1 = LLT.solve(MM * f1);
      buf[i] = f1;
    } else {
      if (i>0) Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> LLT(MM - step * L);
      buf[i] = LLT.solve(MM * buf[i-1]);
      step *= mult;
    }
    if (normalize) normalize_in_01(buf[i]);
    if (i%10 == 0) cout << "level "<< i << " completed\n";
  }
  return buf;
}

vector<Eigen::VectorXd> Smoothness_energy_opt(const DrawableTrimesh<> &m, const Eigen::VectorXd &f,
                                  int nlevels, float w_param = 1000000, int stride=10, float div = 1.05,
                                  bool linear=true, bool normalize=false)
{
  Eigen::SparseMatrix<double> L  = laplacian(m, COTANGENT);
  Eigen::SparseMatrix<double> M = mass_matrix(m);
  Eigen::SparseMatrix<double> M_inv = Invert_diag_matrix(M);
  Eigen::SparseMatrix<double> LML  = L*M_inv*L;
  vector<Eigen::VectorXd> buf(nlevels);
  Eigen::VectorXd f1(f);

  cout << "Smoothness energy optimization: ";
  if (linear) cout << "linear progression...\n"; else cout << "exponential progression...\n";
  float w = w_param;
  if (normalize) normalize_in_01(f1);
  buf[0]=f1;

  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> LLT(LML + w*M);
  for (auto i=1;i<nlevels;i++) {
    // iterated optimization of energy
    if (linear) {
      for (int j=0;j<stride;j++) f1 = LLT.solve(w*M*f1);
      buf[i] = f1;
    } else {
      if (i>0) Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> LLT(LML + w*M);
      buf[i] = LLT.solve(w * M * buf[i-1]);
      w /= div;
    }
    if (normalize) normalize_in_01(buf[i]);
    if (i%10 == 0) cout << "level "<< i << " completed\n";
  }
  return buf;
}



void Build_disc_ss(GLcanvas &gui, State &gs)
// compute a discrete scale-space with "levels" levels from input field f on shape m
// if linear applies diffusion flow levels * mult times with steady coefficient time_scalar
// else applies diffusion flow "levels" times with exponentially increasing diffusion coefficient:
// initial coefficient time_scalar is multiplied by mult at each iteration
// cumulative smoothing: input field at iteration i is the smoothed field at iteration i-1
// if normalize the result is normalizd in [0,1] at each iteration
{
  vector<Eigen::VectorXd> buf;

  cout << "Building scale-space - ";

  if (gs.method==1)
    buf = Diffusion_flow(gs.m,gs.f,gs.nlevels,gs.diff_lambda,
                          gs.diff_stride,gs.diff_mult,gs.diff_progression==1,gs.normalize);
  else
    buf = Smoothness_energy_opt(gs.m,gs.f,gs.nlevels,gs.opt_w,
                                gs.opt_stride,gs.opt_div,gs.opt_progression==1,gs.normalize);

  gs.fields = vector<vector<double>>(gs.nlevels,vector<double>(gs.nverts));
  for (auto i=0;i<buf.size();i++) // convert from Eigen to std::vector
    gs.fields[i] = vector(buf[i].data(),buf[i].data()+buf[i].size());

  // reset critical points in viewer
  if (gs.show_cp) {
    remove_points(gs.points,gui);
    gs.show_cp = false;
  }

  // set clamp limits for visualization
  gs.clamp_limits = vector<float*>(gs.nlevels);
  for (auto &l : gs.clamp_limits) l = new float[2];
  for (auto i=0;i<gs.nlevels;i++) 
    Set_clamp_limits(buf[i], 2, gs.clamp_limits[i]); 

  // find critical points
  gs.critical = vector<vector<int>>(gs.nlevels,vector<int>(gs.nverts));
  vector<vector<int>> ties; // "flat" edges
  for (auto i=0;i<buf.size();i++) {
    gs.critical[i] = Find_Critical_Points(gs.m,gs.VV,buf[i],ties);
    if (ties.size()>0) {    // report ties, if any
      cout << "Found ties at level " << i << ": ";
      for (auto i=0;i<ties.size();i++) 
        cout << "(" << ties[i][0] << "," << ties[i][1] << "), ";
      cout << endl;
    }
  }

  //log
  print_statistics(gs.critical);
  cout << "done"<< endl;

  // set state
  gs.selected_entry = 0;
  gs.SCALE_SPACE_IS_PRESENT = true;
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
        Load_mesh(gui,gs);
      }
    }
    // Modal popup for loading files
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center,ImGuiCond_Appearing,ImVec2(0.5f,0.5f));
    if (ImGui::BeginPopupModal("Load mesh?", NULL, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse))
    {
      static bool dont_ask_me_next_time = false;
      if (dont_ask_me_next_time) {Load_mesh(gui,gs); ImGui::CloseCurrentPopup();}
      ImGui::Text("All data structures will be reset - Load anyway?\n\n");
      ImGui::Separator();           
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0,0));
      ImGui::Checkbox("Don't ask me next time", &dont_ask_me_next_time);
      ImGui::PopStyleVar();
      if (ImGui::Button("OK", ImVec2(120,0))) {Load_mesh(gui,gs); ImGui::CloseCurrentPopup();}
      ImGui::SameLine();
      if (ImGui::Button("Cancel", ImVec2(120,0))) {ImGui::CloseCurrentPopup();}
      ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Save scale-space")) {
      if (!gs.SCALE_SPACE_IS_PRESENT) 
        ImGui::OpenPopup("No data!");
      else Save_scale_space(gs);
    }
    ImGui::SameLine();
    if (ImGui::Button("Export critical points")) {
      if (!gs.SCALE_SPACE_IS_PRESENT || gs.selected_entry>gs.critical.size()) 
        ImGui::OpenPopup("No data!");
      else Export_critical_points(gs);
    }

    // Alert popup
    if (ImGui::BeginPopupModal("No data!", NULL, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse))
    {
      ImGui::Text("No data available! - Cannot perform this task\n\n");
      if (ImGui::Button("Cancel", ImVec2(120,0))) {ImGui::CloseCurrentPopup();}
      ImGui::EndPopup();
    }

    // Field
    ImGui::SeparatorText("Field");
    ImGui::PushItemWidth(200);
    ImGui::Combo("Method",&gs.current_field_method,FIELD_METHOD_LABELS,IM_ARRAYSIZE(FIELD_METHOD_LABELS)); 
    ImGui::PopItemWidth();
    ImGui::SameLine(280);
    ImGui::Checkbox("Normalize", &gs.normalize_f);
    ImGui::Text("Eigenfunctions:");
    ImGui::PushItemWidth(100);
    ImGui::InputInt("Max",&gs.max_eigenfunctions,1,100); ImGui::SameLine(200);
    ImGui::InputInt("Selected",&gs.selected_eigenfunction,0,99);
    ImGui::PopItemWidth();
      if (ImGui::Button("Generate field")) {
      if (!gs.MESH_IS_LOADED) 
        ImGui::OpenPopup("No data!");
      else if (!gs.FIELD_IS_PRESENT) 
        Generate_field(gui,gs);
      else ImGui::OpenPopup("Generate field?"); 
    }
    // Modal popup for generating field
    if (ImGui::BeginPopupModal("Generate field?", NULL, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse))
    {
      static bool dont_ask_me_next_time = false;
      if (dont_ask_me_next_time) {Generate_field(gui,gs); ImGui::CloseCurrentPopup();}
      ImGui::Text("Field and scale-space will be reset - Generate anyway?\n\n");
      ImGui::Separator();           
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0,0));
      ImGui::Checkbox("Don't ask me next time", &dont_ask_me_next_time);
      ImGui::PopStyleVar();
      if (ImGui::Button("OK", ImVec2(120,0))) {Generate_field(gui,gs); ImGui::CloseCurrentPopup();}
      ImGui::SameLine();
      if (ImGui::Button("Cancel", ImVec2(120,0))) {ImGui::CloseCurrentPopup();}
      ImGui::EndPopup();
    }

    // Processing
    ImGui::SeparatorText("Scale-space");
    ImGui::PushItemWidth(200);
    ImGui::InputInt("Levels",&gs.nlevels,1,10); ImGui::SameLine(280);
    ImGui::PopItemWidth();
    ImGui::Checkbox("Normalize", &gs.normalize);
    ImGui::Text("Method: ");
    {
      ImGui::BeginChild("Diffusion",ImVec2(ImGui::GetContentRegionAvail().x*0.5,120));
      ImGui::RadioButton("Diffusion",&gs.method,1);
      ImGui::InputFloat("Lambda",&gs.diff_lambda,0.0001,0.001,"%.6f");
      ImGui::Text("Progression: ");
      ImGui::RadioButton("Linear",&gs.diff_progression,1); ImGui::SameLine(60);
      ImGui::PushItemWidth(60);
      ImGui::InputInt("Stride",&gs.diff_stride,0,0);
      ImGui::PopItemWidth();
      ImGui::RadioButton("Exp.",&gs.diff_progression,2);  ImGui::SameLine(60);
      ImGui::PushItemWidth(60);
      ImGui::InputFloat("Mult.",&gs.diff_mult,0,0,"%.3f");
      ImGui::PopItemWidth();
      ImGui::EndChild();
    }
    ImGui::SameLine(); 
    // ImGui::Text("|"); ImGui::SameLine(); 
    {
      ImGui::BeginChild("Optimization",ImVec2(0,120));
      ImGui::RadioButton("Smoothness opt.",&gs.method,2);
      ImGui::InputFloat("w",&gs.opt_w,0.0001,0.001,"%.6f");
      ImGui::Text("Progression: ");
      ImGui::RadioButton("Linear",&gs.opt_progression,1); ImGui::SameLine(60);
      ImGui::PushItemWidth(60);
      ImGui::InputInt("Stride",&gs.opt_stride,0,0);
      ImGui::PopItemWidth();
      ImGui::RadioButton("Exp.",&gs.opt_progression,2); ImGui::SameLine(60); 
      ImGui::PushItemWidth(60);
      ImGui::InputFloat("Div.",&gs.opt_div,0,0,"%.3f");
      ImGui::PopItemWidth();
      ImGui::EndChild();
    }
    if (ImGui::Button("Compute scale-space")) {
      if (!gs.FIELD_IS_PRESENT) 
        ImGui::OpenPopup("No data!");
      else if (!gs.SCALE_SPACE_IS_PRESENT) 
        Build_disc_ss(gui,gs);
      else ImGui::OpenPopup("Build scale-space?"); 
    }
    // Modal popup for generating scale-space
    if (ImGui::BeginPopupModal("Build scale-space?", NULL, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse))
    {
      static bool dont_ask_me_next_time = false;
      if (dont_ask_me_next_time) {Build_disc_ss(gui,gs); ImGui::CloseCurrentPopup();}
      ImGui::Text("Scale-space will be reset - Build anyway?\n\n");
      ImGui::Separator();           
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0,0));
      ImGui::Checkbox("Don't ask me next time", &dont_ask_me_next_time);
      ImGui::PopStyleVar();
      if (ImGui::Button("OK", ImVec2(120,0))) {Build_disc_ss(gui,gs); ImGui::CloseCurrentPopup();}
      ImGui::SameLine();
      if (ImGui::Button("Cancel", ImVec2(120,0))) {ImGui::CloseCurrentPopup();}
      ImGui::EndPopup();
    }

    // View
    ImGui::SeparatorText("View");
    if (ImGui::Checkbox("Show mesh", &gs.show_m)) {
      if (gs.show_m) gs.m.show_mesh(true);
      else gs.m.show_mesh(false);
      if (gs.MESH_IS_LOADED) gs.m.updateGL();
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Show wireframe", &gs.show_wf)) {
      if (gs.show_wf) gs.m.show_wireframe(true);
      else gs.m.show_wireframe(false);
      if (gs.MESH_IS_LOADED) gs.m.updateGL();
    } 
    ImGui::SameLine();
    if (ImGui::Checkbox("Show Critical Points", &gs.show_cp)) {
      if (gs.SCALE_SPACE_IS_PRESENT && gs.show_cp) {
        set_points(gs.m,gs.critical[gs.selected_entry],gs.points,gs.point_size*gs.point_multiplier);
        draw_points(gs.points,gui);
      } else if (gs.SCALE_SPACE_IS_PRESENT && !gs.show_cp)
        remove_points(gs.points,gui);
      else gs.show_cp = false;
    } 

    ImGui::Text("Show field: "); ImGui::SameLine(100);
    if (ImGui::RadioButton("None",&gs.show_field,0)) gs.m.show_poly_color();
    ImGui::SameLine();
    if (ImGui::RadioButton("Base",&gs.show_field,1) && gs.FIELD_IS_PRESENT) {
        gs.curr_clamp[0]=gs.f_clamp[0];
        gs.curr_clamp[1]=gs.f_clamp[1];
        ScalarField phi = Clamp_and_rescale_field(vector(gs.f.data(),gs.f.data()+gs.f.size()),gs.curr_clamp);
        phi.copy_to_mesh(gs.m);
        gs.m.show_texture1D(TEXTURE_1D_HSV);
    }  
    ImGui::SameLine();
    if (ImGui::RadioButton("Scale-space",&gs.show_field,2) && gs.SCALE_SPACE_IS_PRESENT) {
        gs.curr_clamp[0]=gs.clamp_limits[gs.selected_entry][0];
        gs.curr_clamp[1]=gs.clamp_limits[gs.selected_entry][1];
        ScalarField phi = Clamp_and_rescale_field(gs.fields[gs.selected_entry],gs.curr_clamp);
        phi.copy_to_mesh(gs.m);
        gs.m.show_texture1D(TEXTURE_1D_HSV);
    }

    if (ImGui::SliderFloat("Point size", &gs.point_multiplier, 0, 10.0, "%.1f")) {
      if (gs.show_cp && gs.SCALE_SPACE_IS_PRESENT) {
        set_points(gs.m,gs.critical[gs.selected_entry],gs.points,gs.point_size*gs.point_multiplier);
        // gs.m.updateGL();
      }
    }
    if (ImGui::SliderFloat2("Clamp values", gs.curr_clamp, gs.f.minCoeff(), gs.f.maxCoeff(),"%.6f",ImGuiSliderFlags_Logarithmic)) {
       if (gs.show_field==1 && gs.FIELD_IS_PRESENT) {
        ScalarField phi = Clamp_and_rescale_field(vector(gs.f.data(),gs.f.data()+gs.f.size()),gs.curr_clamp);
        phi.copy_to_mesh(gs.m);
        gs.m.updateGL();
      } else if (gs.show_field == 2 && gs.SCALE_SPACE_IS_PRESENT) {
        ScalarField phi = Clamp_and_rescale_field(gs.fields[gs.selected_entry],gs.curr_clamp);
        phi.copy_to_mesh(gs.m);
        gs.m.updateGL();
      }
    }
    if (ImGui::InputInt("Level", &gs.selected_entry)) {
      if (gs.selected_entry>=gs.fields.size()) gs.selected_entry = gs.fields.size()-1;
      else {
        if (gs.show_field==2 && gs.SCALE_SPACE_IS_PRESENT) {
          gs.curr_clamp[0]=gs.clamp_limits[gs.selected_entry][0];
          gs.curr_clamp[1]=gs.clamp_limits[gs.selected_entry][1];
          ScalarField phi = Clamp_and_rescale_field(gs.fields[gs.selected_entry],gs.curr_clamp);
          phi.copy_to_mesh(gs.m);
          gs.m.updateGL();
        }
        if (gs.show_cp && gs.SCALE_SPACE_IS_PRESENT) {
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

  //SETUP GLOBAL STATE AND GUI:::::::::::::::::::::::
  State gs;
  GLcanvas gui = Init_GUI();
  Setup_GUI_Callbacks(gui,gs);

  //Load mesh
  if (argc>1) {
    string s = "../data/" + string(argv[1]);
    Load_mesh(s,gui,gs);
  }

  // // // GENERATE FIELD:::::::::::::::::::::::::::::::
  // Generate_field(gui,gs);

  // // COMPUTE DISCRETE SCALE SPACE:::::::::::::::::
  // Build_disc_ss(gui,gs);

  // render the mesh
  return gui.launch();
}
