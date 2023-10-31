#pragma once
#include <realtime/gpu.h>
#include <unordered_set>
#include <utils/drawing_circle.h>
#include <utils/drawing_polygons.h>
#include <utils/mesh_io.h>
#include <utils/utilities.h>
#include <yocto/yocto_bvh.h>
#include <yocto_gui/yocto_imgui.h>
#include <yocto_gui/yocto_shade.h>
#include <yocto_gui/yocto_window.h>
using namespace yocto;

#define PROFILE 0

inline std::unordered_set<int> make_set(size_t n) {
  auto result = std::unordered_set<int>();
  result.reserve(n);
  for (int i = 0; i < n; i++) {
    result.insert(i);
  }
  return result;
}
struct Gui_Input {
  bool translating = false;
  bool rotating = false;
  bool scaling = false;
  bool duplicating = false;
  bool angle_editing = false;
};

struct Added_Path {
  geodesic_path path;
  vector<vec3f> positions;
  vec3f color;
  float radius;
  shade_instance *instance;
};

struct Added_Points {
  vector<mesh_point> points;
  vec3f color;
  float radius;
  shade_instance *instance;
};
enum struct editing_context : uint {
  is_doing_nothing,
  // is_rotating,
  is_editing_existing_curve,
  is_creating_new_curve,
  is_creating_first_curve,
};

static string context_names[4] = {
    "is_doing_nothing",
    "is_editing_existing_curve",
    "is_creating_new_curve",
    "is_creating_first_curve",
};
// enum app_state {
//   GEODESIC,
//   CIRCLE,
//   TRIANGLE,
//   SQUARE,
//   RHOMBUS,
//   RECTANGLE,
//   PARALLELOGRAM,
//   PENTAGON,
//   HEXAGON,
//   OCTAGON,
//   DECAGON
// };
enum primitives {
  none,
  geo,
  polyline,
  sheaf,
  cir,
  polygon,
  tri,
  rect,
  pent,
  hex,
  oct,
  deca
};
struct canvas {
  int mode = none;
  vector<mesh_point> points = vector<mesh_point>();
  vector<geodesic_path> paths = {geodesic_path{}};
  vector<Circle> circles = {Circle()};
  unordered_map<int, vector<int>> circles_to_shape = {};
  vector<Added_Path *> circle_shape = {};
  Added_Path *result_shape = nullptr;
  Added_Path *geodesic_shape = nullptr;
};
struct App {
  // TEMPORARY DATA
  string models_name = "test";
  vec3f prev_color = zero3f;
  Circle *selected_circle = nullptr;
  int result_entry = -1;
  vector<Added_Path *> added_paths = {};
  vector<Added_Points *> added_points = {};
  vector<Added_Path *> polyhedral_radial_geodesic_shape = {};
  vector<Added_Path *> smooth_radial_geodesic_shape = {};
  Added_Path *vector_field_shape = {};
  Added_Path *vector_field_shape2 = {};
  vector<Added_Path *> ground_truth_shapes = {};
  Added_Path *gamma_shape = {};
  vec2i sheaf_range = zero2i;
  Added_Path *gamma0_shape = {};
  Added_Path *gamma1_shape = {};
  Added_Path *average_shape = {};
  Added_Path *bezier_circle_shape = {};
  vector<Added_Path *> circle_shape = {};
  vector<Added_Points *> control_points_shape = {};
  vector<int> selected_circle_entries = {};
  vec3f source_color = zero3f;
  vec3f cl_color = {1, 1, 0};
  geodesic_path gamma = {};
  vector<mesh_point> gamma0 = {};
  vector<mesh_point> gamma1 = {};
  vec3f curr_color = zero3f;
  vec3f agap_color = vec3f{1, 0, 0};
  int curr_path = 0;
  int curr_circle = 0;
  int selected_point = -1;
  int selected_mesh = -1;
  int source_shape_entry = -1;
  int type_of_solver = graph;
  int total_number_of_curves = 5;
  float threshold_for_jumps = 10;
  vector<float> w = {};
  int selected_weights = 0;
  float curr_w = 1.0;
  bool show_gradient = false;
  bool use_exp_map_construction = true;
  bool show_gamma = false;
  bool show_original_polygon = true;
  bool show_agap = false;
  bool show_inscr_cir = false;
  bool show_construction = false;
  vector<vector<mesh_point>> polyline = {};
  vector<vector<vec3f>> polyline_pos = {};
  vector<vector<int>> node_of_the_tree = {};
  vector<float> field = {};
  vector<vector<float>> fields = {};
  vector<vector<vec3f>> grads = {};
  float scaling_factor = 5;
  vector<vec3f> vector_field = {};
  vector<vec3f> glyph_normals{};
  vector<vec3f> vector_field2 = {};
  vector<vec3f> vector_field_pos{};
  vector<vec3f> vector_field_normals{};
  vector<vec3f> vector_field_pos2{};
  vector<vec3f> vector_field_normals2{};
  vector<ogl_shape *> temp_points = {};
  int temp_levels = -1;
  ogl_shape *eval_point_shape = new ogl_shape{};
  Svg svg = {};
  mesh_point xxx_point = {};
  vector<mesh_point> control_points = {};
  vector<bool> point_moved = {};
  unordered_map<int, int> point_to_shape;
  vector<int> verts = {};
  int selected_vert = 0;
  vec2i window_size = {};
  bool started = false;
  int playback_tick = 0;
  int type_of_rectangle = Diagonal;
  int type_of_triangle = Equilateral;
  int type_of_circle = G1;
  bool edit_mode = false;
  bool flip_triangle = false;
  float theta0 = 0.f;
  float theta1 = 0.f;
  float lambda1 = 0.f;
  float lambda2 = 1.f;
  float lambda3 = 0.f;
  float len = 0.f;
  float sigma = 0.5;
  float lambda = 1;
  vec2f v0 = zero2f;
  vec2f v1 = zero2f;
  vec3f v03d = zero3f;
  vec3f v13d = zero3f;
  canvas canv = {};
  shape_data mesh = {};
  shape_geometry topology = {};
  shape_op operators = {};
  geodesic_solver solver = {};
  dual_geodesic_solver dual_solver = {};
  Isoline isoline = Isoline();
  struct {
    mat4f view = identity4x4f;
    mat4f projection = identity4x4f;
    mat4f projection_view = identity4x4f;
  } matrices;

  // bool recording_input = false;
  // vector<gui_input> input_record = {};
  // struct Editing_State {
  //   Gui_Input input = {};
  // };
  // editing_context context = editing_context::is_doing_nothing;
  // Editing_State state = {};

  // int editing_history_count = 0;
  // vector<Editing_State> editing_history = {};
  // int history_index = 0;

  // const Gui_Input &input() const { return state.input; }
  // Gui_Input &input() { return state.input; }

  // void commit_state() {
  //   editing_history.resize(history_index + 1);
  //   editing_history[history_index] = state;
  //   history_index += 1;
  //   printf("%s: %d\n", "commit_state()", history_index);
  // }

  string filename = "data/mesh.obj";
  string testname = "tests/test.json";
  string scene_name = "data/mesh.obj_gamma";
  string exported_scene_name = "scene.ply";
  float line_size = 1;
  float curve_size = 10.0;
  int scale_factor = 5;
  int vector_thickness_scale_factor = 3;
  float vector_thickness = 0.0005 * vector_thickness_scale_factor;
  int type_of_strip = 0;
  float vector_size = 0.005 * scale_factor;
  float lift_factor = 0;
  bool initial_strip = false;
  bool path_on_graph = false;
  bool recompute_fields = false;
  float time_of_last_click = -1;

  float angle = 0;

  string error = "";

  bool show_edges = false;
  bool show_points = true;
  bool envlight = false;
  bool show_control_polygon = false;
  bool AEAP = true;
  ogl_shape edges_shape = {};
  ogl_shape branches_shape = {};
  ogl_shape co_branches_shape = {};

  gui_widget *widget = new gui_widget{};
  shade_scene *scene = nullptr;
  shade_material *spline_material = nullptr;
  shade_material *mesh_material = nullptr;
  shade_shape *mesh_shape = nullptr;
  vector<shade_material *> meshes_material = {};
  vector<shade_shape *> meshes_shape = {};
  shade_camera *camera = {};
  gpu::Camera gpu_camera;
  yocto::shade_params shade_params{};
  float camera_focus;
  shape_bvh bvh = {};
  struct Editing_State {
    Gui_Input input = {};
  };

  Editing_State e_state = {};
  const Gui_Input &input() const { return e_state.input; }
  Gui_Input &input() { return e_state.input; }
  // Data stored on the gpu for rendering.
  std::unordered_map<string, ogl_shape> shapes;
  std::unordered_map<string, ogl_program> shaders;

  std::unordered_map<string, gpu::Shape> gpu_shapes;
  std::unordered_map<string, gpu::Shader> gpu_shaders;
};

#include <thread>
template <typename F> inline void parallel_for(int size, F &&f) {
  auto num_threads = min(size, 16);
  auto threads = vector<std::thread>(num_threads);
  auto batch_size = (size + num_threads - 1) / num_threads;

  auto batch = [&](int k) {
    int from = k * batch_size;
    int to = min(from + batch_size, size);
    for (int i = from; i < to; i++)
      f(i);
  };

  for (int k = 0; k < num_threads; k++) {
    threads[k] = std::thread(batch, k);
  }
  for (int k = 0; k < num_threads; k++) {
    threads[k].join();
  }
}

template <typename F> inline void serial_for(int size, F &&f) {
  for (int i = 0; i < size; i++) {
    f(i);
  }
}

void init_bvh(App &app);
void init_bvhs(App &app);
shade_camera _make_framing_camera(const vector<vec3f> &positions);

void init_camera(App &app, const vec3f &from = vec3f{0, 0.5, 1.5},
                 const vec3f &to = {0, 0, 0});

vector<vec3f> make_normals(const vector<vec3i> &triangles,
                           const vector<vec3f> &positions);

void init_bvh(App &app);

ray3f camera_ray(const App &app, vec2f mouse);

vec2f screenspace_from_worldspace(App &app, const vec3f &position);

mesh_point intersect_mesh(const App &app, vec2f mouse);

void init_gpu(App &app, bool envlight);

void delete_app(App &app);

bool load_program(ogl_program *program, const string &vertex_filename,
                  const string &fragment_filename);

void set_points_shape(ogl_shape *shape, const vector<vec3f> &positions);

void set_mesh_shape(ogl_shape *shape, const vector<vec3i> &triangles,
                    const vector<vec3f> &positions,
                    const vector<vec3f> &normals);
void set_polyline_shape(ogl_shape *shape, const vector<vec3f> &positions);

void save_curve(const App &app);
vector<vec3f> load_curve(App &app);
vector<vector<vec3f>> load_WA_curve(App &app);
void save_control_points(const App &app);
void load_control_points(App &app);

inline vec3f random_color() {
  auto result = zero3f;
  for (auto i = 0; i < 3; ++i) {
    result[i] = (float)rand() / RAND_MAX;
  }
  return result;
}

inline int intersect_control_points(App &app, const vec2f &mouse) {
  // Find index of clicked control point.
  float min_dist = flt_max;
  int selected_point = -1;
  float threshold = 0.1;

  for (int i = 0; i < app.control_points.size(); i++) {
    auto &mesh = app.mesh;
    // Skip handle points of non-selected anchors.
    auto point = app.control_points[i];
    auto pos = eval_position(mesh.triangles, mesh.positions, point);
    auto pos_ss = screenspace_from_worldspace(app, pos);
    float dist = length(pos_ss - mouse);
    if (dist < threshold && dist < min_dist) {
      selected_point = i;
      min_dist = dist;
    }
  }

  return selected_point;
}
inline void update_camera_info(App &app, const gui_input &input) {
  auto &camera = *app.camera;
  auto viewport = input.framebuffer_viewport;
  camera.aspect = (float)viewport.z / (float)viewport.w;

  auto camera_yfov =
      (camera.aspect >= 0)
          ? (2 * yocto::atan(camera.film / (camera.aspect * 2 * camera.lens)))
          : (2 * yocto::atan(camera.film / (2 * camera.lens)));

  app.matrices.view = frame_to_mat(inverse(camera.frame));
  app.matrices.projection = perspective_mat(
      camera_yfov, camera.aspect, app.shade_params.near, app.shade_params.far);
  app.matrices.projection_view = app.matrices.projection * app.matrices.view;
}

Added_Path *add_path_shape(App &app, const geodesic_path &path, float radius,
                           const vec3f &color);
Added_Path *add_path_shape(App &app, const vector<vec3f> &positions,
                           float radius, const vec3f &color,
                           const float &threshold = flt_max);

Added_Path *add_path_shape(App &app, const vector<vec3f> &positions,
                           float radius, const vec3f &color, const int entry,
                           const float &threshold = flt_max);

void show_shape(App &app, const int entry);

Added_Points *add_points_shape(App &app, const vector<mesh_point> &points,
                               float radius, const vec3f &color);

Added_Points *add_points_shape(App &app, const vector<vec3f> &points,
                               float radius, const vec3f &color);

Added_Path *add_vector_field_shape(App &app, const vector<vec3f> &vector_field,
                                   const float &scale, const float &radius,
                                   const vec3f &color);
Added_Path *add_glyph_shape(App &app, const vector<vec3f> &alpha,
                            const vector<vec3f> &instances,
                            const vector<vec3f> &normals, const float &scale,
                            const float &radius, const vec3f &color,
                            const float &offset);
void update_generic_vector_field_shape(shade_shape *shape,
                                       const vector<vec3f> &vector_field,
                                       const vector<vec3f> &instances,
                                       const vector<vec3f> &normals,
                                       const float &scale, const float &radius,
                                       const vec3f &color, const float &offset);
void update_glyph_shape(shade_shape *shape, const vector<vec3f> &alpha,
                        const vector<vec3f> &instances,
                        const vector<vec3f> &normals, const float &scale,
                        const float &radius, const vec3f &color,
                        const float &offset);

Added_Path *add_generic_vector_field_shape(
    App &app, const vector<vec3f> &vector_field, const vector<vec3f> &instances,
    const vector<vec3f> &normals, const float &scale, const float &radius,
    const vec3f &color, const float &offset);
void update_path_shape(const vector<Added_Path *> &paths,
                       const shape_data &mesh, const float &radius,
                       const vector<int> &entries);
void update_path_shape(const vector<Added_Path *> &paths,
                       const shape_data &mesh, const vector<vec3f> &positions,
                       const float &radius, const int entry);
void update_path_shape(const vector<Added_Path *> &paths,
                       const shape_data &mesh, const vec3f &color,
                       const int entry);
void update_path_shape(const vector<Added_Path *> &paths,
                       const shape_data &mesh, const vec3f &color,
                       const vector<int> &entries);
void update_path_shape(shade_shape *shape, const shape_data &mesh,
                       const vector<vec3i> &adjacencies,
                       const geodesic_path &path, float radius);
void update_path_shape(shade_shape *shape, const shape_data &mesh,
                       const vector<vec3f> &positions, float radius);
void update_path_shape(shade_shape *shape, const shape_data &mesh,
                       const vector<vec3f> &positions, float radius,
                       const float &treshold);
void update_points_shape(shade_shape *shape, const vector<vec3f> &positions,
                         float radius);
void update_points_shape(shade_shape *shape, const shape_data &mesh,
                         const vector<mesh_point> &points, float radius);
void update_points_shape(const vector<Added_Points *> &points,
                         const shape_data &data,
                         const vector<mesh_point> &new_points, const int entry);
void update_vector_field_shape(shade_shape *shape, shape_data &mesh,
                               const vector<vec3f> &vector_field,
                               const float &scale, const float &radius,
                               const vec3f &color);

Added_Path *add_polygonal_shape(App &app, const vector<mesh_point> &vertices);

Added_Path *add_polyline_shape(App &app, const vector<mesh_point> &vertices);

void update_polygonal_shape(App &app, const vector<mesh_point> &vertices);

void update_polygonal_shape(App &app, const vector<mesh_point> &vertices,
                            const int entry);

void update_polyline_shape(App &app, const vector<mesh_point> &vertices);

void update_glpoints(App &app, const vector<vec3f> &positions,
                     const string &name);

void update_glpoints(App &app, const vector<mesh_point> &points,
                     const string &name = "selected_points");

void update_glvector_field(App &app, const vector<vec3f> &vector_field,
                           float &scale, const string &name = "vector_field");

void compute_affine_transformation(App &app, mesh_point &point,
                                   const vec2f &mouse);

inline bool points_are_too_close(shape_data &mesh, const mesh_point &a,
                                 const mesh_point &b) {
  auto pos_a = eval_position(mesh.triangles, mesh.positions, a);
  auto pos_b = eval_position(mesh.triangles, mesh.positions, b);
  return (length(pos_b - pos_a) < 1e-6);
}
inline void intersect_circle_center(App &app, const vec2f &mouse) {
  // Find index of clicked control point.
  float min_dist = flt_max;

  for (auto i = 0; i < app.canv.circles.size() - 1; ++i) {
    auto &circ = app.canv.circles[i];
    auto &point = circ.center;
    auto pos = eval_position(app.mesh.triangles, app.mesh.positions, point);
    auto pos_ss = screenspace_from_worldspace(app, pos);
    float dist = length(pos_ss - mouse);
    if (dist <= min_dist) {
      app.selected_circle = &circ;
      min_dist = dist;
      app.selected_circle_entries = app.canv.circles_to_shape.at(i);
      app.curr_color =
          (circ.primitive != primitives::cir && circ.construction == -1)
              ? app.added_paths[app.selected_circle_entries[0] + 1]
                    ->instance->material->color
              : app.added_paths[app.selected_circle_entries[0]]
                    ->instance->material->color;
      // app.max_radius = max_of_field(app.mesh, circ.distances);
    }
  }
}
inline bool is_editing(const App &app) {
  return (app.input().angle_editing || app.input().duplicating ||
          app.input().rotating || app.input().scaling ||
          app.input().translating);
}
