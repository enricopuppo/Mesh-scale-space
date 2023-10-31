#include <stdio.h>
#include <thread>
#include <vector>
#include <yocto/yocto_commonio.h>
#include <yocto/yocto_geometry.h>
#include <yocto/yocto_mesh.h>
using namespace std;
#include "app.h"
#include <yocto_gui/yocto_imgui.h>
#include <yocto_gui/yocto_opengl.h>
#include <yocto_gui/yocto_window.h>
using namespace yocto;

//
#include "editing.h"
#include "playback.h"

inline bool affine_transofrmation(const App &app) {
  if (app.input().rotating || app.input().translating || app.input().scaling)
    return true;

  return false;
}
vector<vec3f> concatenate_curve(const vector<vector<vec3f>> &curve) {
  auto result = vector<vec3f>{};
  for (auto i = 0; i < curve.size(); ++i)
    result.insert(result.end(), curve[i].begin(), curve[i].end());

  return result;
}
std::pair<vector<Added_Path *>, vector<int>>
add_isoline_shape(App &app, Isoline &isoline) {
  auto result = vector<Added_Path *>(isoline.size());
  auto entries = vector<int>(isoline.size());
  for (auto i = 0; i < isoline.size(); ++i) {
    auto curr =
        closed_curve_positions(isoline[i], app.mesh.triangles,
                               app.mesh.positions, app.topology.adjacencies);
    entries[i] = (int)app.added_paths.size();
    result[i] =
        add_path_shape(app, curr, app.curve_size * 0.0002, app.curr_color);
  }
  return {result, entries};
}
void update_isoline_shape(App &app, const Isoline &isoline,
                          vector<int> &entries) {
  if (entries.size() == isoline.size()) {
    for (auto i = 0; i < isoline.size(); ++i) {
      auto curr =
          closed_curve_positions(isoline[i], app.mesh.triangles,
                                 app.mesh.positions, app.topology.adjacencies);
      update_path_shape(app.added_paths, app.mesh, curr,
                        app.curve_size * 0.0002, entries[i]);
    }
  } else if (entries.size() < isoline.size()) {
    auto new_entries = entries;
    for (auto i = 0; i < isoline.size(); ++i) {
      if (i < entries.size()) {
        auto curr = closed_curve_positions(isoline[i], app.mesh.triangles,
                                           app.mesh.positions,
                                           app.topology.adjacencies);
        update_path_shape(app.added_paths, app.mesh, curr,
                          app.curve_size * 0.0002, entries[i]);
      } else {
        auto curr = closed_curve_positions(isoline[i], app.mesh.triangles,
                                           app.mesh.positions,
                                           app.topology.adjacencies);
        new_entries.push_back((int)app.added_paths.size());
        add_path_shape(app, curr, app.curve_size * 0.0002, app.curr_color);
      }
    }
    entries = new_entries;
  } else {
    auto erased_entries = vector<int>((int)(entries.size() - isoline.size()));
    for (auto i = 0; i < entries.size(); ++i) {
      if (i < isoline.size()) {
        auto curr = closed_curve_positions(isoline[i], app.mesh.triangles,
                                           app.mesh.positions,
                                           app.topology.adjacencies);
        update_path_shape(app.added_paths, app.mesh, curr,
                          app.curve_size * 0.0002, entries[i]);
      } else {
        erased_entries[i - isoline.size()] = i;
      }
    }
    entries.erase(entries.begin() + erased_entries[0], entries.end());
    for (auto j = 0; j < erased_entries.size(); ++j) {
      clear_shape(app.added_paths[erased_entries[j]]->instance->shape);
    }
    app.added_paths.erase(app.added_paths.begin() + erased_entries[0],
                          app.added_paths.begin() + erased_entries.back() + 1);
  }
}

void set_common_uniforms(const App &app, const ogl_program *program) {
  auto &view = app.matrices.view;
  auto &projection = app.matrices.projection;
  set_uniform(program, "frame", identity4x4f);
  set_uniform(program, "view", view);
  set_uniform(program, "projection", projection);
  set_uniform(program, "eye", app.camera->frame.o);
  set_uniform(program, "envlight", (int)app.envlight);
  set_uniform(program, "gamma", app.shade_params.gamma);
  set_uniform(program, "exposure", app.shade_params.exposure);
  // set_uniform(program, "size", app.line_size);
  if (app.scene->environments.size()) {
    auto &env = app.scene->environments.front();
    if (env->envlight_diffuse)
      set_uniform(program, "envlight_irradiance", env->envlight_diffuse, 6);
    if (env->envlight_specular)
      set_uniform(program, "envlight_reflection", env->envlight_specular, 7);
    if (env->envlight_brdflut)
      set_uniform(program, "envlight_brdflut", env->envlight_brdflut, 8);
  }
}

void draw_scene(const App &app, const vec4i &viewport) {
  clear_ogl_framebuffer(vec4f{0, 0, 0, 1});

  // Draw mesh and environment.
  draw_scene(app.scene, app.camera, viewport, app.shade_params);

  if (app.show_points) {
    auto program = &app.shaders.at("points");
    bind_program(program);
    set_common_uniforms(app, program);
    set_uniform(program, "size", 3.0f * 0.001f * app.line_size);

    set_uniform(program, "color", vec3f{0.1, 0.1, 0.9});
  }

  if (app.temp_levels > 0)
    draw_shape(app.temp_points[app.temp_levels]);
  auto camera_aspect = (float)viewport.z / (float)viewport.w;
  auto camera_yfov =
      camera_aspect >= 0
          ? (2 * yocto::atan(app.camera->film /
                             (camera_aspect * 2 * app.camera->lens)))
          : (2 * yocto::atan(app.camera->film / (2 * app.camera->lens)));
  auto view = frame_to_mat(inverse(app.camera->frame));
  auto projection = perspective_mat(
      camera_yfov, camera_aspect, app.shade_params.near, app.shade_params.far);

  if (app.gpu_shapes.find("edges") != app.gpu_shapes.end())
    gpu::draw_shape(app.gpu_shapes.at("edges"), app.gpu_shaders.at("points"),
                    gpu::Uniform("color", vec3f{0, 0, 0}));
  gpu::set_point_size(10);
  if (app.gpu_shapes.find("selected_points") != app.gpu_shapes.end()) {

    gpu::draw_shape(
        app.gpu_shapes.at("selected_points"), app.gpu_shaders.at("points"),
        gpu::Uniform("color", vec3f{0, 0, 1}),
        gpu::Uniform("frame", identity4x4f), gpu::Uniform("view", view),
        gpu::Uniform("projection", projection));
  }

  if (app.gpu_shapes.find("vector_field") != app.gpu_shapes.end())
    gpu::draw_shape(
        app.gpu_shapes.at("vector_field"), app.gpu_shaders.at("points"),
        gpu::Uniform("color", vec3f{0, 0, 1}),
        gpu::Uniform("frame", identity4x4f), gpu::Uniform("view", view),
        gpu::Uniform("projection", projection));

  if (app.gpu_shapes.find("vector_field_2") != app.gpu_shapes.end())
    gpu::draw_shape(
        app.gpu_shapes.at("vector_field_2"), app.gpu_shaders.at("points"),
        gpu::Uniform("color", vec3f{1, 0, 0}),
        gpu::Uniform("frame", identity4x4f), gpu::Uniform("view", view),
        gpu::Uniform("projection", projection));

  if (app.show_edges) {
    auto program = &app.shaders.at("lines");
    bind_program(program);
    set_common_uniforms(app, program);
    set_uniform(program, "color", vec3f{0, 0, 0});
    draw_shape(&app.edges_shape);
  }
}

inline void sleep(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

inline bool is_pressing(gui_button button) {
  return button.state == gui_button::state::pressing;
}
inline bool is_releasing(gui_button button) {
  return button.state == gui_button::state::releasing;
}
inline bool is_down(gui_button button) {
  return button.state == gui_button::state::down ||
         button.state == gui_button::state::pressing;
}
inline bool is_pressing(const gui_input &input, gui_key key) {
  return is_pressing(input.key_buttons[(int)key]);
}

bool process_camera_move(App &app, const gui_input &input) {

  auto rotating = input.modifier_shift;
  auto panning = input.modifier_alt;
  auto &camera = *app.camera;

  auto update_camera_frame = [&](frame3f &frame, float &focus, bool rotating,
                                 bool panning, bool zooming) {
    auto last_pos = input.mouse_last;
    auto mouse_pos = input.mouse_pos;
    auto mouse_left = is_down(input.mouse_left);
    auto mouse_right = is_down(input.mouse_right);
    // handle mouse and keyboard for navigation
    if (mouse_left) {
      auto dolly = 0.0f;
      auto pan = zero2f;
      auto rotate = zero2f;
      if (rotating) {
        if (mouse_left)
          rotate = (mouse_pos - last_pos) / 100.0f;
      }
      if (zooming) {
        if (mouse_right)
          dolly = (mouse_pos.y - last_pos.y) / 100.0f;
      }
      if (panning) {
        if (mouse_left)
          pan = (mouse_pos - last_pos) * focus / 200.0f;
      }
      pan.x = -pan.x;
      rotate.y = -rotate.y;
      update_turntable(frame, focus, rotate, dolly, pan);
    }
  };

  if (is_down(input.mouse_left) && (rotating || panning)) {
    update_camera_frame(camera.frame, app.camera_focus, rotating, panning,
                        false);
    return true;
  }

  // Zoom-in/out by scrolling;
  float zoom = input.scroll.y * 0.1;
  if (zoom != 0) {
    update_turntable(camera.frame, app.camera_focus, zero2f, zoom, zero2f);
    return true;
  }

  return false;
}
void process_key_input(App &app, const gui_input &input) {

  for (int i = 0; i < input.key_buttons.size(); i++) {
    auto key = gui_key(i);
    if (!is_pressing(input, key))
      continue;
    if (!app.edit_mode)
      continue;
    printf("%c pressed!\n", (char)key);
    if (key == gui_key('R')) {
      app.input().rotating = true;
    }
    if (key == gui_key('S')) {
      app.input().scaling = true;
    }
    if (key == gui_key('T')) {
      app.input().translating = true;
    }
    if (key == gui_key('D')) {
      app.input().duplicating = true;
    }
    if (key == gui_key('A')) {
      app.input().angle_editing = true;
    }
  }
}
void update_primitives(App &app) {

  if (app.canv.mode == geo) {
    if (app.canv.points.size() > 0 && (int)(app.canv.points.size()) % 2 == 0) {
      app.canv.paths[app.curr_path] = compute_geodesic_path(
          app.mesh, app.topology, app.dual_solver, app.canv.points.rbegin()[1],
          app.canv.points.rbegin()[0]);
    }
  } else if (app.canv.mode == sheaf) {
    if (app.canv.points.size() == 3) {
      app.canv.points.erase(app.canv.points.begin(), app.canv.points.end() - 1);
      app.gamma_shape = nullptr;
      app.sheaf_range = zero2i;
    }
  } else if (app.canv.mode == cir) {
    if (app.canv.points.size() == 1 &&
        app.canv.circles[app.curr_circle].distances.size() == 0) {
      app.canv.circles[app.curr_circle] = create_circle(
          app.mesh, app.topology, app.solver, app.canv.points[0], 0.f);
    } else if (app.canv.points.size() == 2) {
      set_radius(app.mesh.triangles, app.topology.adjacencies,
                 &app.canv.circles[app.curr_circle],
                 get_distance(app.mesh.triangles[app.canv.points.back().face],
                              get_bary(app.canv.points.back().uv),
                              app.canv.circles[app.curr_circle].distances));
    }
  } else if (!app.use_exp_map_construction && app.canv.mode == rect &&
             app.canv.points.size() == 2) {
    app.canv.paths.push_back(geodesic_path{});
    ++app.curr_path;
  }
}
void draw_primitives(App &app) {
  auto &canv = app.canv;
  if (app.input().angle_editing && app.selected_circle != nullptr &&
      app.selected_circle != nullptr &&
      app.selected_circle_entries.size() == 1 &&
      !app.selected_circle->has_been_edited) {
    if (app.selected_circle->primitive == tri ||
        app.selected_circle->primitive == rect) {
      auto vertices = app.selected_circle->vertices;
      auto equiangular =
          equiangular_polygon(app.mesh, app.topology, app.dual_solver,
                              app.selected_circle->vertices, app.lambda1,
                              app.lambda2, app.lambda3, app.AEAP);
      app.selected_circle->has_been_edited = true;

      // update_path_shape(
      //     app.added_paths, app.mesh,
      //     polyline_pos(app.mesh.triangles, app.mesh.positions, equiangular),
      //     app.curve_size * 0.000, entry);
      auto new_entry = app.selected_circle_entries[0] + 2;
      add_path_shape(
          app,
          polyline_pos(app.mesh.triangles, app.mesh.positions, equiangular),
          app.curve_size * 0.0002, app.agap_color, new_entry);
      app.input().angle_editing = false;
      app.show_agap = true;
    }
  } else if (affine_transofrmation(app) && app.selected_circle != nullptr &&
             app.selected_circle_entries.size() == 1) {
    auto vertices = vector<mesh_point>{};
    if (app.selected_circle->construction == -1 &&
        app.selected_circle->primitive != rect)
      vertices = make_n_gon(app.mesh, app.topology, app.selected_circle,
                            (int)app.selected_circle->vertices.size());
    else if (app.selected_circle->construction == -1)
      vertices = parallelogram_tangent_space(app.mesh, app.topology,
                                             app.selected_circle, app.sigma, 1,
                                             app.selected_circle->theta);
    else
      vertices = app.selected_circle->vertices;

    update_polygonal_shape(app, vertices, app.selected_circle_entries[0] + 1);
    app.selected_circle->vertices = vertices;
    if (app.selected_circle->has_been_edited && app.show_agap) {
      auto equiangular =
          equiangular_polygon(app.mesh, app.topology, app.dual_solver,
                              app.selected_circle->vertices, app.lambda1,
                              app.lambda2, app.lambda3, app.AEAP);
      auto entry = app.selected_circle_entries[0] + 2;
      update_path_shape(
          app.added_paths, app.mesh,
          polyline_pos(app.mesh.triangles, app.mesh.positions, equiangular),
          app.curve_size * 0.0002, entry);
    }
  } else if (app.canv.mode == geo &&
             canv.paths[app.curr_path].start.face != -1) {

    if (canv.geodesic_shape == nullptr) {
      canv.geodesic_shape =
          add_path_shape(app, canv.paths[app.curr_path],
                         app.curve_size * 0.0002, app.curr_color);
    } else
      update_path_shape(canv.geodesic_shape->instance->shape, app.mesh,
                        app.topology.adjacencies, canv.paths[app.curr_path],
                        app.curve_size * 0.0002);

    if (canv.points.size() == 2) {
      clear_shape(app.added_paths.back()->instance->shape);
      app.added_paths.pop_back();

      add_path_shape(app, canv.paths[app.curr_path], app.curve_size * 0.0002,
                     app.curr_color);

      canv.geodesic_shape = nullptr;
      canv.paths.push_back(geodesic_path{});
      canv.points.clear();
      app.curr_path += 1;
    }

  } else if ((canv.mode == polyline || canv.mode == polygon) &&
             canv.points.size() > 1) {

    if (canv.result_shape == nullptr)
      canv.result_shape = add_polyline_shape(app, canv.points);
    else
      update_polyline_shape(app, canv.points);

  } else if (canv.mode == sheaf && canv.points.size() == 2 &&
             app.gamma_shape == nullptr) {
    auto p0 = canv.points[0];
    auto p1 = canv.points[1];
    auto sheaf = geodesic_sheaf(app.mesh, app.topology, app.dual_solver, p0, p1,
                                app.total_number_of_curves);
    app.gamma_shape = add_path_shape(
        app, polyline_pos(app.mesh.triangles, app.mesh.positions, sheaf[0]),
        app.curve_size * 0.0002, app.curr_color);
    app.sheaf_range.x = (int)app.added_paths.size();
    for (auto i = 1; i < sheaf.size(); ++i)
      add_path_shape(
          app, polyline_pos(app.mesh.triangles, app.mesh.positions, sheaf[i]),
          app.curve_size * 0.0002, app.curr_color);
    app.sheaf_range.y = (int)app.added_paths.size();
  } else if (app.canv.mode == cir &&
             app.canv.circles[app.curr_circle].radius > 0) {
    auto circ = &canv.circles[app.curr_circle];

    if (canv.circle_shape.size() == 0)
      std::tie(canv.circle_shape, app.selected_circle_entries) =
          add_isoline_shape(app, circ->isoline);
    else
      update_isoline_shape(app, circ->isoline, app.selected_circle_entries);

    if (canv.points.size() == 2) {
      // clear_shape(app.added_points.back()->instance->shape);
      canv.circles_to_shape[app.curr_circle] = app.selected_circle_entries;
      canv.circles[app.curr_circle].primitive = cir;
      canv.circles[app.curr_circle].levels = 1;
      ++app.curr_circle;
      canv.circles.push_back(Circle());
      canv.circle_shape = {};
      app.isoline.clear();
      canv.points = {};
      app.bezier_circle_shape = nullptr;
    }
  } else if (app.canv.mode == tri && canv.points.size() > 0) {

    if (canv.points.size() == 1) {

      if (app.canv.circles[app.curr_circle].radius > 0) {

        app.selected_circle = &app.canv.circles[app.curr_circle];

        if (canv.circle_shape.size() == 0)
          std::tie(canv.circle_shape, app.selected_circle_entries) =
              add_isoline_shape(app, app.selected_circle->isoline);
        else
          update_isoline_shape(app, app.selected_circle->isoline,
                               app.selected_circle_entries);

        auto vertices =
            make_n_gon(app.mesh, app.topology, app.selected_circle, 3);
        canv.circles[app.curr_circle].vertices = vertices;

        if (canv.result_shape == nullptr)
          canv.result_shape = add_polygonal_shape(app, vertices);
        else
          update_polygonal_shape(app, vertices,
                                 app.selected_circle_entries[0] + 1);
      } else if (app.canv.paths[app.curr_path].start.face != -1) {
        if (canv.geodesic_shape == nullptr) {
          app.selected_circle_entries = {(int)app.added_paths.size()};
          canv.geodesic_shape =
              add_path_shape(app, canv.paths[app.curr_path],
                             app.curve_size * 0.0002, app.curr_color);
        } else if (app.canv.paths[app.curr_path].end.face != -1)
          update_path_shape(canv.geodesic_shape->instance->shape, app.mesh,
                            app.topology.adjacencies, canv.paths[app.curr_path],
                            app.curve_size * 0.0002);
      }

    } else if (canv.points.size() == 2 &&
               app.selected_circle_entries.size() == 1 &&
               app.use_exp_map_construction) {
      app.canv.points.clear();
      clear_shape(canv.circle_shape[0]->instance->shape);
      canv.circles_to_shape[app.curr_circle] = app.selected_circle_entries;
      canv.circles[app.curr_circle].primitive = tri;

      ++app.curr_circle;
      canv.circles.push_back(Circle());
      app.canv.circle_shape = {};
      app.canv.result_shape = nullptr;
    } else if (canv.points.size() == 2) {
      if (app.type_of_triangle == Equilateral) {
        auto vertices = equilateral_triangle(app.mesh, app.topology, app.solver,
                                             app.dual_solver, canv.points[0],
                                             canv.points[1], app.flip_triangle);
        canv.circles[app.curr_circle].vertices = vertices;

        if (canv.result_shape == nullptr) {

          canv.result_shape = add_polygonal_shape(app, vertices);
        } else
          update_polygonal_shape(app, vertices,
                                 app.selected_circle_entries[0] + 1);
      } else if (app.type_of_triangle == IsoSameL) {
        auto len = path_length(
            compute_geodesic_path(app.mesh, app.topology, app.dual_solver,
                                  canv.points[0], canv.points.back()),
            app.mesh.triangles, app.mesh.positions, app.topology.adjacencies);
        auto vertices = same_lengths_isoscele_triangle(
            app.mesh, app.topology, app.solver, canv.points[0], canv.points[1],
            1.5 * len, app.flip_triangle);
        canv.circles[app.curr_circle].vertices = vertices;
        if (canv.result_shape == nullptr) {
          app.selected_circle_entries = {(int)app.added_paths.size()};
          canv.result_shape = add_polygonal_shape(app, vertices);
        } else
          update_polygonal_shape(app, vertices,
                                 app.selected_circle_entries[0] + 1);

      } else if (app.type_of_triangle == IsoBisector) {
        auto len = path_length(canv.paths[app.curr_path], app.mesh.triangles,
                               app.mesh.positions, app.topology.adjacencies);
        auto vertices = altitude_isoscele_triangle(
            app.mesh, app.topology, app.solver, app.dual_solver, canv.points[0],
            canv.points[1], 1.5 * len, app.flip_triangle);
        canv.circles[app.curr_circle].vertices = vertices;
        if (canv.result_shape == nullptr) {
          canv.result_shape = add_polygonal_shape(app, vertices);
        } else
          update_polygonal_shape(app, vertices,
                                 app.selected_circle_entries[0] + 1);
      }
      canv.circles_to_shape[app.curr_circle] = app.selected_circle_entries;
      canv.circles[app.curr_circle].primitive = tri;
      canv.circles[app.curr_circle].construction = app.type_of_triangle;
      ++app.curr_circle;
      canv.circles.push_back(Circle());
      canv.points.clear();
      canv.geodesic_shape = nullptr;
      canv.result_shape = nullptr;
    }
  } else if (canv.mode == rect && canv.points.size() > 0) {

    if (canv.points.size() == 1) {
      if (app.use_exp_map_construction && app.canv.points.size() == 1 &&
          app.canv.circles[app.curr_circle].radius > 0) {
        auto &circle = app.canv.circles[app.curr_circle];
        if (canv.circle_shape.size() == 0) {
          std::tie(canv.circle_shape, app.selected_circle_entries) =
              add_isoline_shape(app, circle.isoline);
        } else
          update_isoline_shape(app, circle.isoline,
                               app.selected_circle_entries);

        auto vertices = parallelogram_tangent_space(
            app.mesh, app.topology, &circle, app.sigma, 1, circle.theta);
        canv.circles[app.curr_circle].vertices = vertices;
        if (canv.result_shape == nullptr)
          canv.result_shape = add_polygonal_shape(app, vertices);
        else
          update_polygonal_shape(app, vertices);
      } else if (canv.paths[app.curr_path].end.face != -1) {

        if (canv.geodesic_shape == nullptr) {
          app.selected_circle_entries = {(int)app.added_paths.size()};
          canv.geodesic_shape =
              add_path_shape(app, canv.paths[app.curr_path],
                             app.curve_size * 0.0002, app.curr_color);
        } else if (app.canv.paths[app.curr_path].end.face != -1)
          update_path_shape(canv.geodesic_shape->instance->shape, app.mesh,
                            app.topology.adjacencies, canv.paths[app.curr_path],
                            app.curve_size * 0.0002);
      }

    } else if (canv.points.size() == 2) {
      if (app.use_exp_map_construction &&
          app.canv.circles[app.curr_circle].radius > 0 &&
          app.selected_circle_entries.size() == 1) {
        app.canv.points.clear();
        clear_shape(canv.circle_shape[0]->instance->shape);
        canv.circles_to_shape[app.curr_circle] = app.selected_circle_entries;
        canv.circles[app.curr_circle].primitive = rect;
        canv.circles[app.curr_circle].levels = 1;
        ++app.curr_circle;
        canv.circles.push_back(Circle());
        app.canv.circle_shape = {};
        app.canv.result_shape = nullptr;

      } else if (app.canv.paths.back().end.face != -1) {

        switch (app.type_of_rectangle) {
        case Diagonal: {
          auto vertices =
              diagonal_rectangle(app.mesh, app.topology, canv.paths.rbegin()[1],
                                 canv.paths.rbegin()[0]);
          if (canv.result_shape == nullptr) {

            canv.result_shape = add_polygonal_shape(app, vertices);
          } else
            update_polygonal_shape(app, vertices,
                                   app.selected_circle_entries[0] + 1);
        } break;
        case Euclidean: {
          auto height =
              path_length(canv.paths.back(), app.mesh.triangles,
                          app.mesh.positions, app.topology.adjacencies);

          auto vertices = euclidean_rectangle(
              app.mesh, app.topology, app.solver, app.dual_solver,
              compute_geodesic_path(app.mesh, app.topology, app.dual_solver,
                                    canv.points[0], canv.points[1]),
              height);
          if (canv.result_shape == nullptr) {

            canv.result_shape = add_polygonal_shape(app, vertices);
          } else
            update_polygonal_shape(app, vertices,
                                   app.selected_circle_entries[0] + 1);

          // add_points_shape(app, vertices, 2 * app.curve_size * 0.000,
          // zero3f);
        } break;
        case SameLengths: {
          auto height =
              path_length(canv.paths.back(), app.mesh.triangles,
                          app.mesh.positions, app.topology.adjacencies);
          auto vertices = same_lengths_rectangle(
              app.mesh, app.topology, app.solver,
              compute_geodesic_path(app.mesh, app.topology, app.dual_solver,
                                    canv.points[0], canv.points[1]),
              height);
          if (canv.result_shape == nullptr) {

            canv.result_shape = add_polygonal_shape(app, vertices);
          } else
            update_polygonal_shape(app, vertices,
                                   app.selected_circle_entries[0] + 1);

          // add_points_shape(app, vertices, 2 * app.curve_size * 0.000,
          // zero3f);

        } break;
        }
      }

    } else if (canv.points.size() == 3) {

      switch (app.type_of_rectangle) {
      case Diagonal: {
        auto vertices = diagonal_rectangle(app.mesh, app.topology,
                                           canv.paths[0], canv.paths[1]);
        clear_shape(canv.geodesic_shape->instance->shape);
        canv.circles[app.curr_circle].vertices = vertices;
        // canv.paths = {geodesic_path{}};
        // app.curr_path = 0;
        update_polygonal_shape(app, vertices,
                               app.selected_circle_entries[0] + 1);
        canv.points.clear();
      } break;
      case Euclidean: {
        auto height = path_length(canv.paths.back(), app.mesh.triangles,
                                  app.mesh.positions, app.topology.adjacencies);

        auto vertices = euclidean_rectangle(
            app.mesh, app.topology, app.solver, app.dual_solver,
            compute_geodesic_path(app.mesh, app.topology, app.dual_solver,
                                  canv.points[0], canv.points[1]),
            height);
        canv.circles[app.curr_circle].vertices = vertices;
        // canv.paths = {geodesic_path{}};
        // app.curr_path = 0;
        update_polygonal_shape(app, vertices,
                               app.selected_circle_entries[0] + 1);
        canv.points.clear();
      } break;
      case SameLengths: {
        auto height = path_length(canv.paths.back(), app.mesh.triangles,
                                  app.mesh.positions, app.topology.adjacencies);
        auto vertices = same_lengths_rectangle(
            app.mesh, app.topology, app.solver,
            compute_geodesic_path(app.mesh, app.topology, app.dual_solver,
                                  canv.points[0], canv.points[1]),
            height);
        canv.circles[app.curr_circle].vertices = vertices;
        // canv.paths = {geodesic_path{}};
        // app.curr_path = 0;
        update_polygonal_shape(app, vertices,
                               app.selected_circle_entries[0] + 1);
        canv.points.clear();
      } break;
      }

      canv.circles_to_shape[app.curr_circle] = app.selected_circle_entries;
      canv.circles[app.curr_circle].primitive = rect;
      canv.circles[app.curr_circle].construction = app.type_of_rectangle;
      ++app.curr_circle;
      canv.circles.push_back(Circle());
      clear_shape(app.canv.geodesic_shape->instance->shape);
      canv.geodesic_shape = nullptr;
      canv.result_shape = nullptr;
      // canv.result_shape = nullptr;
      // canv.points.clear();
      // add_path_shape(app, line0, 0.75 * app.curve_size * 0.000, {0, 1,
      // 0}); add_path_shape(app, line1, 0.75 * app.curve_size * 0.000, {0,
      // 1, 0}); add_path_shape(app, line2, 0.75 * app.curve_size * 0.000,
      // {0, 1, 0}); add_path_shape(
      //     app,
      //     compute_geodesic_path(app.mesh, vertices[0],
      //     vertices.rbegin()[1]), 0.75 * app.curve_size * 0.000, zero3f);
    }
    // else if (canv.points.size() == 4) {
    //   if (canv.result_shape == nullptr) {
    //     app.result_entry = (int)app.added_paths.size();
    //     canv.result_shape = add_polygonal_shape(app, canv.points);
    //   } else
    //     update_polygonal_shape(app, canv.points, app.result_entry);
    // }
  } else if (app.canv.mode == pent && canv.points.size() > 0) {

    if (canv.points.size() == 1) {

      if (app.canv.circles[app.curr_circle].radius > 0) {

        app.selected_circle = &app.canv.circles[app.curr_circle];

        if (canv.circle_shape.size() == 0)
          std::tie(canv.circle_shape, app.selected_circle_entries) =
              add_isoline_shape(app, app.selected_circle->isoline);
        else
          update_isoline_shape(app, app.selected_circle->isoline,
                               app.selected_circle_entries);

        auto vertices =
            make_n_gon(app.mesh, app.topology, app.selected_circle, 5);
        canv.circles[app.curr_circle].vertices = vertices;

        if (canv.result_shape == nullptr)
          canv.result_shape = add_polygonal_shape(app, vertices);
        else
          update_polygonal_shape(app, vertices,
                                 app.selected_circle_entries[0] + 1);
      }

    } else if (canv.points.size() == 2 &&
               app.selected_circle_entries.size() == 1 &&
               app.use_exp_map_construction) {
      app.canv.points.clear();
      clear_shape(canv.circle_shape[0]->instance->shape);
      canv.circles_to_shape[app.curr_circle] = app.selected_circle_entries;
      canv.circles[app.curr_circle].primitive = tri;

      ++app.curr_circle;
      canv.circles.push_back(Circle());
      app.canv.circle_shape = {};
      app.canv.result_shape = nullptr;
    }
  } else if (app.canv.mode == primitives::hex && canv.points.size() > 0) {

    if (canv.points.size() == 1) {

      if (app.canv.circles[app.curr_circle].radius > 0) {

        app.selected_circle = &app.canv.circles[app.curr_circle];

        if (canv.circle_shape.size() == 0)
          std::tie(canv.circle_shape, app.selected_circle_entries) =
              add_isoline_shape(app, app.selected_circle->isoline);
        else
          update_isoline_shape(app, app.selected_circle->isoline,
                               app.selected_circle_entries);

        auto vertices =
            make_n_gon(app.mesh, app.topology, app.selected_circle, 6);
        canv.circles[app.curr_circle].vertices = vertices;

        if (canv.result_shape == nullptr)
          canv.result_shape = add_polygonal_shape(app, vertices);
        else
          update_polygonal_shape(app, vertices,
                                 app.selected_circle_entries[0] + 1);
      }

    } else if (canv.points.size() == 2 &&
               app.selected_circle_entries.size() == 1 &&
               app.use_exp_map_construction) {
      app.canv.points.clear();
      clear_shape(canv.circle_shape[0]->instance->shape);
      canv.circles_to_shape[app.curr_circle] = app.selected_circle_entries;
      canv.circles[app.curr_circle].primitive = tri;

      ++app.curr_circle;
      canv.circles.push_back(Circle());
      app.canv.circle_shape = {};
      app.canv.result_shape = nullptr;
    }
  } else if (app.canv.mode == primitives::oct && canv.points.size() > 0) {

    if (canv.points.size() == 1) {

      if (app.canv.circles[app.curr_circle].radius > 0) {

        app.selected_circle = &app.canv.circles[app.curr_circle];

        if (canv.circle_shape.size() == 0)
          std::tie(canv.circle_shape, app.selected_circle_entries) =
              add_isoline_shape(app, app.selected_circle->isoline);
        else
          update_isoline_shape(app, app.selected_circle->isoline,
                               app.selected_circle_entries);

        auto vertices =
            make_n_gon(app.mesh, app.topology, app.selected_circle, 8);
        canv.circles[app.curr_circle].vertices = vertices;

        if (canv.result_shape == nullptr)
          canv.result_shape = add_polygonal_shape(app, vertices);
        else
          update_polygonal_shape(app, vertices,
                                 app.selected_circle_entries[0] + 1);
      }

    } else if (canv.points.size() == 2 &&
               app.selected_circle_entries.size() == 1 &&
               app.use_exp_map_construction) {
      app.canv.points.clear();
      clear_shape(canv.circle_shape[0]->instance->shape);
      canv.circles_to_shape[app.curr_circle] = app.selected_circle_entries;
      canv.circles[app.curr_circle].primitive = tri;

      ++app.curr_circle;
      canv.circles.push_back(Circle());
      app.canv.circle_shape = {};
      app.canv.result_shape = nullptr;
    }
  } else if (app.canv.mode == primitives::deca && canv.points.size() > 0) {

    if (canv.points.size() == 1) {

      if (app.canv.circles[app.curr_circle].radius > 0) {

        app.selected_circle = &app.canv.circles[app.curr_circle];

        if (canv.circle_shape.size() == 0)
          std::tie(canv.circle_shape, app.selected_circle_entries) =
              add_isoline_shape(app, app.selected_circle->isoline);
        else
          update_isoline_shape(app, app.selected_circle->isoline,
                               app.selected_circle_entries);

        auto vertices =
            make_n_gon(app.mesh, app.topology, app.selected_circle, 10);
        canv.circles[app.curr_circle].vertices = vertices;

        if (canv.result_shape == nullptr)
          canv.result_shape = add_polygonal_shape(app, vertices);
        else
          update_polygonal_shape(app, vertices,
                                 app.selected_circle_entries[0] + 1);
      }

    } else if (canv.points.size() == 2 &&
               app.selected_circle_entries.size() == 1 &&
               app.use_exp_map_construction) {
      app.canv.points.clear();
      clear_shape(canv.circle_shape[0]->instance->shape);
      canv.circles_to_shape[app.curr_circle] = app.selected_circle_entries;
      canv.circles[app.curr_circle].primitive = tri;

      ++app.curr_circle;
      canv.circles.push_back(Circle());
      app.canv.circle_shape = {};
      app.canv.result_shape = nullptr;
    }
  }
}
bool process_user_input(App &app, const gui_input &input) {

  if (is_active(app.widget))
    return false;
  if (process_camera_move(app, input)) {
    update_camera_info(app, input);
    return false;
  }
  if (app.canv.mode == none)
    return false;
  auto mouse = input.mouse_pos;
  auto size = vec2f{(float)input.window_size.x, (float)input.window_size.y};
  mouse = vec2f{2 * (mouse.x / size.x) - 1, 1 - 2 * (mouse.y / size.y)};
  auto editing = input.modifier_alt || input.modifier_shift;

  if (app.edit_mode && is_pressing(input.mouse_left)) {
    intersect_circle_center(app, mouse);
  }
  auto point = intersect_mesh(app, mouse);

  if (is_releasing(input.mouse_left) && !editing && !app.edit_mode &&
      point.face != -1) {
    app.canv.points.push_back(point);
    update_primitives(app);
    return true;
  }
  if (is_releasing(input.mouse_right)) {
    if (app.input().rotating || app.input().scaling ||
        app.input().translating) {
      app.input().rotating = false;
      app.input().translating = false;
      app.input().scaling = false;
      app.selected_circle = nullptr;
    } else if (app.canv.mode == polyline) {
      app.canv.points.clear();
      app.canv.result_shape = nullptr;
    } else if (app.canv.mode == polygon) {
      update_polygonal_shape(app, app.canv.points);
      app.canv.points.clear();
      app.canv.result_shape = nullptr;
    }
    return true;
  }

  auto drag = input.mouse_pos - input.mouse_last;

  if (!app.edit_mode) {
    if (app.canv.mode == cir) {
      if (app.canv.points.size() == 1 && point.face != -1) {
        auto circle = &app.canv.circles[app.curr_circle];

        set_radius(app.mesh.triangles, app.topology.adjacencies, circle,
                   get_distance(app.mesh.triangles[point.face],
                                get_bary(point.uv), circle->distances));
      }
    } else if (app.canv.mode == geo && app.canv.points.size() > 0) {
      if (validate_points({point}) &&
          !points_are_too_close(app.mesh, app.canv.points.back(), point))
        app.canv.paths[app.curr_path] =
            compute_geodesic_path(app.mesh, app.topology, app.dual_solver,
                                  app.canv.points.back(), point);

    } else if (app.use_exp_map_construction && app.canv.points.size() == 1) {
      if (validate_points({point}) &&
          !points_are_too_close(app.mesh, app.canv.points.back(), point)) {
        auto &circle = app.canv.circles[app.curr_circle];
        if (circle.distances.size() > 0)
          set_radius(app.mesh.triangles, app.topology.adjacencies, &circle,
                     get_distance(app.mesh.triangles[point.face],
                                  get_bary(point.uv), circle.distances));
        else
          circle = create_circle(app.mesh, app.topology, app.solver,
                                 app.canv.points[0], point);
        return true;
      }
    } else if (app.canv.mode == rect && app.canv.points.size() > 0 &&
               length(drag) > 0.1) {
      if (validate_points({point}) &&
          !points_are_too_close(app.mesh, app.canv.points.back(), point)) {
        if (app.canv.points.size() == 1) {
          app.canv.paths[app.curr_path] =
              compute_geodesic_path(app.mesh, app.topology, app.dual_solver,
                                    app.canv.points.back(), point);
          auto &circle = app.canv.circles[app.curr_circle];
          circle.center = app.canv.points[0];
          circle.distances = vector<float>(4);
        } else if (app.canv.points.size() == 2) {
          if (app.type_of_rectangle == Diagonal)
            app.canv.paths[app.curr_path] =
                compute_geodesic_path(app.mesh, app.topology, app.dual_solver,
                                      app.canv.points[0], point);
          else
            app.canv.paths[app.curr_path] =
                compute_geodesic_path(app.mesh, app.topology, app.dual_solver,
                                      app.canv.points.back(), point);
        }
      }
      return true;
    } else if (app.canv.mode == tri && app.canv.points.size() > 0 &&
               length(drag) > 0.1) {
      if (validate_points({point}) &&
          !points_are_too_close(app.mesh, app.canv.points.back(), point)) {
        if (app.canv.points.size() == 1) {
          auto &circle = app.canv.circles[app.curr_circle];
          circle.center = app.canv.points[0];
          circle.distances = vector<float>(4);
          if (!app.use_exp_map_construction)
            app.canv.paths[app.curr_path] =
                compute_geodesic_path(app.mesh, app.topology, app.dual_solver,
                                      app.canv.points.back(), point);
        }
        return true;
      }
    }
  } else if (length(drag) > 0.05 && app.selected_circle != nullptr)
    compute_affine_transformation(app, point, mouse);

  if (is_releasing(input.mouse_right))
    app.selected_point = -1;
  return false;
}

void update_app(App &app, const gui_input &input) {
  // process_gui_input(app, input); TODO(giacomo)

  if (is_active(app.widget))
    return;

  app.window_size = input.window_size;
  process_key_input(app, input);
  process_user_input(app, input);
  draw_primitives(app);

  auto tasks = vector<vec2i>{};
}

void draw(const gui_input &input, void *data) {
  auto &app = *(App *)data;
  app.started = true;
  auto mesh = &app.mesh;
  auto geometry = &app.topology;
  auto op = &app.operators;
  auto solver = &app.solver;
  auto dual_solver = &app.dual_solver;
  auto canv = &app.canv;
  auto primitives_names = vector<string>{
      "None",     "Geodesic", "Polyline", "Sheaf",   "Circle",  "Polygon",
      "Triangle", "Rect",     "Pentagon", "Hexagon", "Octagon", "Decagon"};
  auto rect_names = vector<string>{"Diagonal", "Euclidean", "Same Lenghts"};
  auto tri_names =
      vector<string>{"Equilateral", "Iso-Same Lenghts", "Iso-Bisect"};
  auto circ_names = vector<string>{"Euclidean", "G1"};

  update_camera_info(app, input);

  // Do everything
  update_app(app, input);

  draw_scene(app, input.framebuffer_viewport);

  auto widget = app.widget;
  begin_widget(widget, "Regular shapes on meshes");

  // draw_bullet_text(widget, "Distance Field");
  // vector<string> solver_names = {"Exact", "Graph Based"};
  // if (draw_combobox(widget, "Solver", app.type_of_solver, solver_names)) {
  //   for (auto i = 0; i < app.point_moved.size(); ++i) {
  //     app.point_moved[i] = true;
  //   }
  // }
  draw_separator(widget);

  draw_separator(widget);
  draw_bullet_text(widget, "Primitives Tracing");
  draw_combobox(widget, "Select Primitive", app.canv.mode, primitives_names);
  draw_checkbox(widget, "TP Construction", app.use_exp_map_construction);
  continue_line(widget);
  if (draw_checkbox(widget, "Edit Mode", app.edit_mode)) {
    if (!app.edit_mode)
      app.selected_circle = nullptr;
  }

  if (draw_slider(widget, "H/B Ratio", app.sigma, 0.01, 1.f)) {
    auto vertices = vector<mesh_point>{};
    if (app.edit_mode && app.selected_circle != nullptr) {
      if (app.selected_circle->primitive == rect)
        vertices = parallelogram_tangent_space(app.mesh, app.topology,
                                               app.selected_circle, app.sigma,
                                               1.f, app.selected_circle->theta);
      app.selected_circle->vertices = vertices;
      update_polygonal_shape(app, vertices, app.selected_circle_entries[0] + 1);
    }
  }
  auto disabled = push_disable_items(app.use_exp_map_construction ||
                                     app.canv.mode != tri || !app.edit_mode);

  if (draw_combobox(widget, "Type of Triangle", app.type_of_triangle,
                    tri_names)) {

    if (app.selected_circle != nullptr &&
        app.selected_circle->vertices.size() == 3 &&
        app.selected_circle_entries.size() == 1) {
      {
        auto &points = app.selected_circle->vertices;
        switch (app.type_of_triangle) {
        case Equilateral: {

          auto vertices = equilateral_triangle(
              app.mesh, app.topology, app.solver, app.dual_solver, points[0],
              points[1], app.flip_triangle);
          points = vertices;

        } break;
        case IsoSameL: {
          auto len = path_length(
              compute_geodesic_path(app.mesh, app.topology, app.dual_solver,
                                    points[0], points.back()),
              app.mesh.triangles, app.mesh.positions, app.topology.adjacencies);
          auto vertices = same_lengths_isoscele_triangle(
              app.mesh, app.topology, app.solver, points[0], points[1],
              1.5 * len, app.flip_triangle);
          points = vertices;

        } break;
        case IsoBisector: {
          auto len = path_length(
              compute_geodesic_path(app.mesh, app.topology, app.dual_solver,
                                    points[0], points[1]),
              app.mesh.triangles, app.mesh.positions, app.topology.adjacencies);
          auto vertices = altitude_isoscele_triangle(
              app.mesh, app.topology, app.solver, app.dual_solver, points[0],
              points[1], 1.5 * len, app.flip_triangle);
          points = vertices;
        } break;
        }
        update_polygonal_shape(app, points, app.selected_circle_entries[0] + 1);
      }
    }
  }
  disabled = pop_disable_items(disabled);
  disabled = push_disable_items(app.use_exp_map_construction ||
                                app.canv.mode != rect || !app.edit_mode);

  if (draw_combobox(widget, "Type of Rectangle", app.type_of_rectangle,
                    rect_names)) {
    if (app.selected_circle != nullptr &&
        app.selected_circle->vertices.size() == 4 &&
        app.selected_circle_entries.size() == 1) {
      {
        auto &points = app.selected_circle->vertices;
        switch (app.type_of_rectangle) {
        case Euclidean: {
          auto height = path_length(
              compute_geodesic_path(*mesh, *geometry, *dual_solver,
                                    points.rbegin()[1], points.back()),
              mesh->triangles, mesh->positions, geometry->adjacencies);
          auto vertices = euclidean_rectangle(
              *mesh, *geometry, *solver, *dual_solver,
              compute_geodesic_path(*mesh, *geometry, *dual_solver, points[0],
                                    points.back()),
              height);
          points = vertices;

        } break;
        case SameLengths: {
          auto height = path_length(
              compute_geodesic_path(*mesh, *geometry, *dual_solver,
                                    points.rbegin()[1], points.back()),
              mesh->triangles, mesh->positions, geometry->adjacencies);

          auto vertices = same_lengths_rectangle(
              *mesh, *geometry, *solver,
              compute_geodesic_path(*mesh, *geometry, *dual_solver, points[0],
                                    points.back()),
              height);
          points = vertices;

        } break;
        case Diagonal: {
          auto base = compute_geodesic_path(*mesh, *geometry, *dual_solver,
                                            points[0], points.back());
          auto diagonal = compute_geodesic_path(*mesh, *geometry, *dual_solver,
                                                points[0], points[2]);
          auto vertices = diagonal_rectangle(*mesh, *geometry, base, diagonal);
          points = vertices;
        } break;
        }
        update_polygonal_shape(app, points, app.selected_circle_entries[0] + 1);
      }
    }
  }
  disabled = pop_disable_items(disabled);
  disabled = push_disable_items(!app.use_exp_map_construction ||
                                app.canv.mode != cir || !app.edit_mode);
  if (draw_combobox(widget, "Type of Circle", app.type_of_circle, circ_names)) {
    if (app.selected_circle != nullptr) {
      // if (app.type_of_circle == Classical) {
      auto cir =
          trace_circle(*mesh, *geometry, *op, *solver, *dual_solver,
                       app.selected_circle->center, app.selected_circle->radius,
                       8, 10, exact, app.type_of_circle);
      auto pos = concatenate_curve(cir);
      if (pos.size() > 0) {
        if (app.bezier_circle_shape == nullptr)
          app.bezier_circle_shape = add_path_shape(
              app, pos, app.curve_size * 0.0002, vec3f{0.1, 0.1, 0.9});
        else
          update_path_shape(app.bezier_circle_shape->instance->shape, *mesh,
                            pos, app.curve_size * 0.0002);
      }
      // } else {
      //   auto [points, sides] = debug_g1_circle_control_points(
      //       *mesh, *geometry, app.selected_circle->center,
      //       app.selected_circle->radius, 5);
      //   add_points_shape(app, points, 0.002, vec3f{1, 0, 0});
      //   for (auto &s : sides)
      //     add_path_shape(app, polyline_pos(mesh->triangles, mesh->positions,
      //     s),
      //                    0.001, vec3f{0.1, 0.1, 0.9});
      // }
    }
  }
  disabled = pop_disable_items(disabled);
  {
    draw_bullet_text(widget, "Test on Geodesic");
    if (draw_slider(widget, "Number of curves", app.total_number_of_curves, 2,
                    8)) {
      if (app.gamma_shape != nullptr) {
        for (auto i = app.sheaf_range.x; i < app.sheaf_range.y; ++i) {
          clear_shape(app.added_paths[i]->instance->shape);
        }
        app.added_paths.erase(app.added_paths.begin() + app.sheaf_range.x,
                              app.added_paths.begin() + app.sheaf_range.y);
        auto p0 = canv->points[0];
        auto p1 = canv->points[1];
        auto sheaf = geodesic_sheaf(app.mesh, app.topology, app.dual_solver, p0,
                                    p1, app.total_number_of_curves);
        app.sheaf_range.x = (int)app.added_paths.size();
        for (auto i = 1; i < sheaf.size(); ++i)
          add_path_shape(
              app,
              polyline_pos(app.mesh.triangles, app.mesh.positions, sheaf[i]),
              app.curve_size * 0.00022, app.curr_color);
        app.sheaf_range.y = (int)app.added_paths.size();
      }
    }
    if (draw_checkbox(widget, "Show Geodesic", app.show_gamma)) {
      if (app.gamma_shape != nullptr) {
        if (app.show_gamma)
          app.gamma_shape->instance->material->color = vec3f{1, 0, 0};
        else
          app.gamma_shape->instance->material->color = app.curr_color;
      }
    }
    // if(draw_checkbox(widget,"Show Original
    // Polygon",app.show_original_polygon))
    // {
    //   if(app.selected_circle!=nullptr && app.show_original_polygon)
    //   {
    //     app.added_paths.insert(app.added_paths.begin()+app.selected_circle_entries[0],add_path_shape())
    //   }
    // }
    if (draw_checkbox(widget, "As-Euclidean-as-possibile", app.AEAP)) {
      if (app.selected_circle != nullptr &&
          app.selected_circle->has_been_edited) {
        auto equiangular = equiangular_polygon(
            *mesh, *geometry, *dual_solver, app.selected_circle->vertices,
            app.lambda1, app.lambda2, app.lambda3, app.AEAP);
        update_path_shape(
            app.added_paths, *mesh,
            polyline_pos(mesh->triangles, mesh->positions, equiangular),
            app.curve_size * 0.0002, app.selected_circle_entries[0] + 2);
      }
    }

    if (draw_checkbox(widget, "Show Circle", app.show_inscr_cir)) {
      if (app.selected_circle != nullptr &&
          app.selected_circle->primitive != cir && app.show_inscr_cir) {
        show_shape(app, app.selected_circle_entries[0]);
      } else if (app.selected_circle != nullptr &&
                 app.selected_circle->primitive != cir && !app.show_agap)
        clear_shape(
            app.added_paths[app.selected_circle_entries[0]]->instance->shape);
    }
    if (draw_checkbox(widget, "Show Original Polygon",
                      app.show_original_polygon)) {
      if (app.selected_circle != nullptr &&
          app.selected_circle->has_been_edited && app.show_original_polygon) {
        show_shape(app, app.selected_circle_entries[0] + 1);
      } else if (app.selected_circle != nullptr &&
                 app.selected_circle->has_been_edited &&
                 !app.show_control_polygon)
        clear_shape(app.added_paths[app.selected_circle_entries[0] + 1]
                        ->instance->shape);
    }
    continue_line(widget);
    if (draw_checkbox(widget, "Show AGAP", app.show_agap)) {
      if (app.selected_circle != nullptr &&
          app.selected_circle->has_been_edited && app.show_agap) {
        show_shape(app, app.selected_circle_entries[0] + 2);
      } else if (app.selected_circle != nullptr &&
                 app.selected_circle->has_been_edited && !app.show_agap)
        clear_shape(app.added_paths[app.selected_circle_entries[0] + 2]
                        ->instance->shape);
    }

    if (draw_checkbox(widget, "Show Construction", app.show_construction)) {
      if (app.canv.paths.size() > 0 && app.canv.paths[0].start.face != -1 &&
          app.show_construction) {
        auto &p = app.canv.paths[0];
        auto len = path_length(p, mesh->triangles, mesh->positions,
                               geometry->adjacencies);
        auto t = tangent_path_direction(*mesh, *geometry, p, true);
        for (auto i = 1; i < 5; ++i) {
          auto v = rot_vect(t, i * 2 * pif / 5);
          auto gamma =
              polthier_straightest_geodesic(*mesh, *geometry, p.start, v, len);
          add_path_shape(app,
                         polyline_pos(mesh->triangles, mesh->positions, gamma),
                         app.curve_size * 0.0002, app.curr_color);
        }
      } else if (app.canv.paths.size() > 0 &&
                 app.canv.paths[0].start.face != -1 && !app.show_construction) {
        auto &p = app.canv.paths[0];
        auto len = path_length(p, mesh->triangles, mesh->positions,
                               geometry->adjacencies);
        auto t = tangent_path_direction(*mesh, *geometry, p, true);
        auto vertices = vector<mesh_point>(5);
        vertices[0] = p.end;
        for (auto i = 1; i < 5; ++i) {
          auto v = rot_vect(t, i * 2 * pif / 5);
          auto gamma =
              polthier_straightest_geodesic(*mesh, *geometry, p.start, v, len);
          vertices[i] = gamma.back();
        }
        app.curr_color = vec3f{1, 0, 0};
        add_polygonal_shape(app, vertices);
      }
    }
    if (draw_button(widget, "Hinge")) {
      if (app.canv.paths.size() > 0 &&
          app.canv.paths.rbegin()[1].start.face != -1) {
        auto &p = app.canv.paths.rbegin()[1];
        auto len = path_length(p, mesh->triangles, mesh->positions,
                               geometry->adjacencies);
        auto t = tangent_path_direction(*mesh, *geometry, p, true);
        auto v = rot_vect(t, pif / 3);

        auto gamma =
            polthier_straightest_geodesic(*mesh, *geometry, p.start, v, len);
        add_path_shape(app,
                       polyline_pos(mesh->triangles, mesh->positions, gamma),
                       app.curve_size * 0.0002, app.curr_color);

        auto gamma0 = compute_geodesic_path(*mesh, *geometry, *dual_solver,
                                            p.end, gamma.back());
        add_path_shape(app, gamma0, app.curve_size * 0.0002, vec3f{1, 0, 0});
      }
    }
    // }
    if (draw_slider(widget, "theta0", app.theta0, 0.0, pif / 2 - 1e-8)) {
      if (app.curr_path > 0 &&
          canv->paths[app.curr_path - 1].start.face != -1) {
        auto path = &canv->paths[app.curr_path - 1];
        auto len = path_length(*path, mesh->triangles, mesh->positions,
                               geometry->adjacencies);
        app.v0 = tangent_path_direction(*mesh, *geometry, *path, true);
        app.v0 = rot_vect(app.v0, app.theta0);
        app.gamma0 = polthier_straightest_geodesic(*mesh, *geometry,
                                                   path->start, app.v0, len);
        // if (app.gamma0_shape != nullptr)
        //   update_path_shape(
        //       app.gamma0_shape->instance->shape, *mesh,
        //       polyline_pos(mesh->triangles, mesh->positions, app.gamma0),
        //       0.001);
        // else
        //   app.gamma0_shape = add_path_shape(
        //       app, polyline_pos(mesh->triangles, mesh->positions,
        //       app.gamma0), 0.001, vec3f{1, 0, 0});

        if (app.average_shape != nullptr) {

          auto avg = geodesic_average(*mesh, *geometry, app.dual_solver,
                                      app.canv.points[0], app.canv.points[1],
                                      normalize(app.v0), normalize(app.v1),
                                      app.theta0, app.theta1, app.lambda1,
                                      app.lambda2, app.lambda3, len);
          update_path_shape(app.average_shape->instance->shape, *mesh,
                            polyline_pos(mesh->triangles, mesh->positions, avg),
                            0.001);
        }
        auto n = tid_normal(mesh->triangles, mesh->positions, path->start.face);
        auto v03d =
            rot_vect(tangent_path_direction3D(*mesh, *geometry, *path, true), n,
                     app.theta0);

        app.vector_field = {normalize(v03d)};
        app.vector_field_pos = {
            eval_position(app.mesh.triangles, app.mesh.positions, path->start)};
        app.vector_field_normals = {n};
        if (app.vector_field_shape != nullptr)
          update_generic_vector_field_shape(
              app.vector_field_shape->instance->shape, app.vector_field,
              app.vector_field_pos, app.vector_field_normals, app.vector_size,
              app.vector_thickness, vec3f{1, 0, 0}, app.lift_factor);
        else {
          app.vector_field_shape = add_generic_vector_field_shape(
              app, app.vector_field, app.vector_field_pos,
              app.vector_field_normals, app.vector_size, app.vector_thickness,
              vec3f{1, 0, 0}, app.lift_factor);
        }
      }
    }
    if (draw_slider(widget, "theta1", app.theta1, 0.0, pif / 2 - 1e-8)) {
      if (app.curr_path > 0 &&
          canv->paths[app.curr_path - 1].start.face != -1) {
        auto path = canv->paths[app.curr_path - 1];
        auto len = path_length(path, mesh->triangles, mesh->positions,
                               geometry->adjacencies);
        app.v1 = tangent_path_direction(*mesh, *geometry, path, false);
        app.v1 = rot_vect(app.v1, -app.theta1);
        app.gamma1 = polthier_straightest_geodesic(*mesh, *geometry, path.end,
                                                   app.v1, len);
        // if (app.gamma1_shape != nullptr)
        //   update_path_shape(
        //       app.gamma1_shape->instance->shape, *mesh,
        //       polyline_pos(mesh->triangles, mesh->positions, app.gamma1),
        //       0.001);
        // else
        //   app.gamma1_shape = add_path_shape(
        //       app, polyline_pos(mesh->triangles, mesh->positions,
        //       app.gamma1), 0.001, vec3f{0, 0, 1});

        if (app.average_shape != nullptr) {

          auto avg = geodesic_average(*mesh, *geometry, app.dual_solver,
                                      app.canv.points[0], app.canv.points[1],
                                      normalize(app.v0), normalize(app.v1),
                                      app.theta0, app.theta1, app.lambda1,
                                      app.lambda2, app.lambda3, len);
          update_path_shape(app.average_shape->instance->shape, *mesh,
                            polyline_pos(mesh->triangles, mesh->positions, avg),
                            0.001);
        }
        auto n = tid_normal(mesh->triangles, mesh->positions, path.end.face);
        auto v23d = tangent_path_direction3D(*mesh, *geometry, path, false);
        v23d = rot_vect(v23d, n, -app.theta1);
        app.vector_field2 = {normalize(v23d)};
        app.vector_field_pos2 = {
            eval_position(app.mesh.triangles, app.mesh.positions, path.end)};
        app.vector_field_normals2 = {n};
        if (app.vector_field_shape2 != nullptr) {
          update_generic_vector_field_shape(
              app.vector_field_shape2->instance->shape, app.vector_field2,
              app.vector_field_pos2, app.vector_field_normals2, app.vector_size,
              app.vector_thickness, vec3f{0, 0, 1}, app.lift_factor);
        } else {
          app.vector_field_shape2 = add_generic_vector_field_shape(
              app, app.vector_field2, app.vector_field_pos2,
              app.vector_field_normals2, app.vector_size, app.vector_thickness,
              vec3f{0, 0, 1}, app.lift_factor);
        }
      }
    }

    if (draw_button(widget, "Geodesic Average")) {
      if (app.curr_path > 0 &&
          canv->paths[app.curr_path - 1].start.face != -1) {
        // auto points =
        //     paired_samples(*mesh, *geometry, app.gamma0, app.gamma1,
        // 50);
        // for (auto i = 0; i < points.size(); ++i) {
        //   // auto gamma = compute_geodesic_path(*mesh, *geometry,
        //   app.dual_solver,
        //   //  points[i][0], points[i][1]);
        //   auto col = random_color();

        //   add_points_shape(app, points[i], 0.0020, col);
        //   //  add_path_shape(app, gamma,
        //   //  0.0005, col);
        // }
        auto path = canv->paths[app.curr_path - 1];
        auto len = path_length(path, mesh->triangles, mesh->positions,
                               geometry->adjacencies);

        auto [avg, points] = geodesic_average_w_control_points(
            *mesh, *geometry, app.dual_solver, app.canv.points[0],
            app.canv.points[1], normalize(app.v0), normalize(app.v1),
            app.theta0, app.theta1, app.lambda1, app.lambda2, app.lambda3, len);
        // geodesic_average(*mesh, *geometry, app.dual_solver, app.gamma0,
        //                  app.gamma1, 500);
        app.average_shape = add_path_shape(
            app, polyline_pos(mesh->triangles, mesh->positions, avg),
            app.curve_size * 0.0002, app.curr_color);
        add_points_shape(app, {points[1], points[2]}, app.curve_size * 0.001,
                         {1, 0, 0});
      }
    }

    if (draw_button(widget, "Jacobi Field")) {
      if (app.curr_path > 0 &&
          canv->paths[app.curr_path - 1].start.face != -1 && app.theta0 > 0) {
        auto gamma = &canv->paths[app.curr_path - 1];
        auto r = path_length(*gamma, mesh->triangles, mesh->positions,
                             geometry->adjacencies);
        auto L = 1;
        app.vector_field = jacobi_vector_field(*mesh, *geometry, *gamma, L);
        std::tie(app.vector_field_pos, app.vector_field_normals) =
            path_positions_and_normals(*gamma, mesh->triangles, mesh->positions,
                                       geometry->adjacencies);
        app.vector_field_shape = add_generic_vector_field_shape(
            app, app.vector_field, app.vector_field_pos,
            app.vector_field_normals, app.vector_size, app.vector_thickness,
            {0, 1, 0}, app.lift_factor);
      }
    }
  }

  draw_separator(widget);
  draw_bullet_text(widget, "Parameter for Energy Minimization");
  if (draw_slider(widget, "Stretch Energy", app.lambda1, 0.0,
                  1.0 - app.lambda2 - app.lambda3)) {
    if (app.selected_circle != nullptr && app.edit_mode &&
        app.selected_circle->has_been_edited) {
      auto equiangular = equiangular_polygon(
          *mesh, *geometry, *dual_solver, app.selected_circle->vertices,
          app.lambda1, app.lambda2, app.lambda3, app.AEAP);
      update_path_shape(
          app.added_paths, *mesh,
          polyline_pos(mesh->triangles, mesh->positions, equiangular),
          app.curve_size * 0.0002, app.selected_circle_entries[0] + 2);
    }
  }
  if (draw_slider(widget, "Strain Energy", app.lambda2, 0.0,
                  1.0 - app.lambda1 - app.lambda3)) {
    if (app.selected_circle != nullptr && app.edit_mode &&
        app.selected_circle->has_been_edited) {
      auto equiangular = equiangular_polygon(
          *mesh, *geometry, *dual_solver, app.selected_circle->vertices,
          app.lambda1, app.lambda2, app.lambda3, app.AEAP);
      update_path_shape(
          app.added_paths, *mesh,
          polyline_pos(mesh->triangles, mesh->positions, equiangular),
          app.curve_size * 0.0002, app.selected_circle_entries[0] + 2);
    }
  }
  if (draw_slider(widget, "Curvature Variation", app.lambda3, 0.0,
                  1.0 - app.lambda1 - app.lambda2)) {
    if (app.selected_circle != nullptr && app.edit_mode &&
        app.selected_circle->has_been_edited) {
      auto equiangular = equiangular_polygon(
          *mesh, *geometry, *dual_solver, app.selected_circle->vertices,
          app.lambda1, app.lambda2, app.lambda3, app.AEAP);
      update_path_shape(
          app.added_paths, *mesh,
          polyline_pos(mesh->triangles, mesh->positions, equiangular),
          app.curve_size * 0.0002, app.selected_circle_entries[0] + 2);
    }
  }
  draw_separator(widget);
  draw_bullet_text(widget, "Edit Lines and Arrows");
  if (draw_slider(widget, "Line size", app.curve_size, 1, 20)) {
    if (app.edit_mode && app.selected_circle != nullptr) {
      auto entries = vector<int>{};
      if (app.selected_circle->primitive != cir) {
        entries = {app.selected_circle_entries[0] + 1};
        if (app.selected_circle->has_been_edited)
          entries.push_back(app.selected_circle_entries[0] + 2);
      } else
        entries = app.selected_circle_entries;
      update_path_shape(app.added_paths, *mesh, app.curve_size * 0.0002,
                        entries);
    }
  }
  if (draw_slider(widget, "vectors size", app.scale_factor, 0, 50)) {
    app.vector_size = 0.005 * app.scale_factor;
    if (app.vector_field_shape != nullptr) {
      if (app.vector_field_pos.size() != 0)
        update_generic_vector_field_shape(
            app.vector_field_shape->instance->shape, app.vector_field,
            app.vector_field_pos, app.vector_field_normals, app.vector_size,
            app.vector_thickness, {1, 0, 0}, app.lift_factor);
      else
        update_vector_field_shape(app.vector_field_shape->instance->shape,
                                  *mesh, app.vector_field, app.vector_size,
                                  app.vector_thickness, {1, 0, 0});
    }
    if (app.vector_field_shape2 != nullptr) {
      if (app.vector_field_pos2.size() != 0)
        update_generic_vector_field_shape(
            app.vector_field_shape2->instance->shape, app.vector_field2,
            app.vector_field_pos2, app.vector_field_normals2, app.vector_size,
            app.vector_thickness, {0, 0, 1}, app.lift_factor);
      else
        update_vector_field_shape(app.vector_field_shape->instance->shape,
                                  *mesh, app.vector_field, app.vector_size,
                                  app.vector_thickness, {0, 0, 1});
    }
  }

  if (draw_slider(widget, "vectors thickness",
                  app.vector_thickness_scale_factor, 1, 25)) {
    app.vector_thickness = 0.0005 * app.vector_thickness_scale_factor;
    if (app.vector_field_shape != nullptr) {
      if (app.vector_field_pos.size() != 0)
        update_generic_vector_field_shape(
            app.vector_field_shape->instance->shape, app.vector_field,
            app.vector_field_pos, app.vector_field_normals, app.vector_size,
            app.vector_thickness, {1, 0, 0}, app.lift_factor);
      else
        update_vector_field_shape(app.vector_field_shape->instance->shape,
                                  *mesh, app.vector_field, app.vector_size,
                                  app.vector_thickness, {1, 0, 0});
    }
    if (app.vector_field_shape2 != nullptr) {
      if (app.vector_field_pos2.size() != 0)
        update_generic_vector_field_shape(
            app.vector_field_shape2->instance->shape, app.vector_field2,
            app.vector_field_pos2, app.vector_field_normals2, app.vector_size,
            app.vector_thickness, {0, 0, 1}, app.lift_factor);
      else
        update_vector_field_shape(app.vector_field_shape->instance->shape,
                                  *mesh, app.vector_field, app.vector_size,
                                  app.vector_thickness, {0, 0, 1});
    }
  }

  if (draw_slider(widget, "lift vector", app.lift_factor, 0, 0.05)) {

    if (app.vector_field_shape != nullptr && app.lift_factor != 0) {

      if (app.vector_field_pos.size() != 0)
        update_generic_vector_field_shape(
            app.vector_field_shape->instance->shape, app.vector_field,
            app.vector_field_pos, app.vector_field_normals, app.vector_size,
            app.vector_thickness, {0, 0, 1}, app.lift_factor);
      else
        update_vector_field_shape(app.vector_field_shape->instance->shape,
                                  *mesh, app.vector_field, app.vector_size,
                                  app.vector_thickness, {0, 0, 1});
    }

    if (app.vector_field_shape2 != nullptr && app.lift_factor != 0) {
      for (auto i = 0; i < app.vector_field.size(); ++i) {
        app.vector_field2[i] += app.lift_factor * mesh->normals[i];
      }
      update_vector_field_shape(app.vector_field_shape2->instance->shape, *mesh,
                                app.vector_field2, app.vector_size,
                                app.vector_thickness, {1, 0, 0});
    }
  }
  draw_textinput(widget, "Scene Name", app.models_name);

  draw_separator(widget);
  draw_checkbox(widget, "show edges", app.show_edges);
  draw_coloredit(widget, "Mesh Color", app.mesh_material->color);
  if (draw_coloredit(widget, "Current Color", app.curr_color)) {
    if (app.edit_mode && app.selected_circle != nullptr) {
      auto entry = 0;
      if (app.selected_circle->primitive != cir) {
        entry = app.selected_circle_entries[0] + 1;
      } else
        entry = app.selected_circle_entries[0];
      update_path_shape(app.added_paths, *mesh, app.curr_color, {entry});
    }
  }
  if (draw_coloredit(widget, "AGAP Color", app.agap_color)) {
    if (app.edit_mode && app.selected_circle != nullptr &&
        app.selected_circle->has_been_edited) {
      auto entry = app.selected_circle_entries[0] + 2;
      update_path_shape(app.added_paths, *mesh, app.agap_color, {entry});
    }
  }

  app.shade_params.faceted = app.show_edges;

  if (draw_button(widget, " Reset")) {
    app.canv.points.clear();
    app.canv.points.clear();

    for (auto path : app.added_paths)
      clear_shape(path->instance->shape);

    for (auto point : app.added_points)
      clear_shape(point->instance->shape);

    if (app.gamma_shape != nullptr)
      clear_shape(app.gamma_shape->instance->shape);
    if (app.gamma0_shape != nullptr)
      clear_shape(app.gamma0_shape->instance->shape);
    if (app.gamma1_shape != nullptr)
      clear_shape(app.gamma1_shape->instance->shape);

    app.gamma_shape = nullptr;
    app.gamma0_shape = nullptr;
    app.gamma1_shape = nullptr;

    app.vector_field.clear();
    app.added_paths.clear();
    app.field.clear();
    app.fields.clear();
    app.grads.clear();
    app.w.clear();
    app.point_moved.clear();
    app.point_to_shape.clear();
  }

  end_widget(widget);
}

int main(int num_args, const char *args[]) {
  auto app = App();

  int msaa = 1;
  string filename = args[1];

  if (!load_mesh(filename, app.mesh, app.topology, app.operators, app.solver,
                 app.dual_solver, app.error))
    print_fatal(app.error);
  init_bvh(app);

  app.filename = path_basename(filename);
  // Init window.
  auto win = new gui_window();
  win->msaa = msaa;
  init_window(win, {1080, 720}, "mesh viewer", true);
  win->user_data = &app;

  init_gpu(app, app.envlight);

  init_widget(app.widget, win);

  if (msaa > 1)
    set_ogl_msaa();

  run_ui(win, draw);

  // TODO(giacomo): delete app
  clear_window(win);
}
