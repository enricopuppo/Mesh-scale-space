#include "utilities.h"
#include <cinolib/drawable_segment_soup.h>
#include <cinolib/drawable_sphere.h>
#include <cinolib/geodesics.h>
#include <cinolib/gl/glcanvas.h>
#include <cinolib/gl/surface_mesh_controls.h>
#include <cinolib/gradient.h>
#include <cinolib/io/write_OBJ.h>
#include <cinolib/mean_curv_flow.h>
#include <cinolib/meshes/drawable_tetmesh.h>
#include <cinolib/vector_serialization.h>
#include <fstream>
using namespace std;
using namespace cinolib;
// NOTES: 71 does not get the correspondence right on one hand
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//                                 GUI utility
vec2i pick_vert(const vector<vec3d> &pos0, const vector<vec3d> &pos1,
                const vec3d &p) {
  auto n0 = pos0.size();
  auto n1 = pos1.size();
  auto n = max(n0, n1);

  auto len = DBL_MAX;
  auto closest = -1;
  auto mesh = -1;
  for (uint vid = 0; vid < n; ++vid) {

    if (vid < n0) {
      auto curr_len = (p - pos0[vid]).norm();
      if (curr_len < len) {
        len = curr_len;
        closest = vid;
        mesh = 0;
      }
    }
    if (vid < n1) {
      auto curr_len = (p - pos1[vid]).norm();
      if (curr_len < len) {
        len = curr_len;
        closest = vid;
        mesh = 1;
      }
    }
  }
  return vec2i{closest, mesh};
}
int pick_vert(const vector<vec3d> &pos, const vec3d &p) {
  auto n = pos.size();

  auto len = DBL_MAX;
  auto closest = -1;
  for (uint vid = 0; vid < n; ++vid) {

    auto curr_len = (p - pos[vid]).norm();
    if (curr_len < len) {
      len = curr_len;
      closest = vid;
    }
  }
  return closest;
}
int pick_vert(const vector<vec3d> &pos, const vector<int> &gt, const vec3d &p) {
  auto n = pos.size();

  auto len = DBL_MAX;
  auto closest = -1;
  auto mesh = -1;
  for (uint vid = 0; vid < n; ++vid) {
    if (find(gt.begin(), gt.end(), vid + 1) == gt.end())
      continue;
    auto curr_len = (p - pos[vid]).norm();
    if (curr_len < len) {
      len = curr_len;
      closest = vid;
    }
  }
  return closest;
}
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

double translate_shape(vector<vec3d> &pos0, vector<vec3d> &pos1) {

  auto rightmost = DBL_MIN;
  auto leftmost = DBL_MAX;
  for (auto &v : pos0) {
    rightmost = std::max(v.x(), rightmost);
  }
  for (auto &v : pos1) {
    leftmost = std::min(v.x(), leftmost);
  }
  auto delta = (rightmost - leftmost) / 2 + 0.005;
  for (auto &v : pos1) {
    v += vec3d{delta, 0, 0};
  }
  for (auto &v : pos0) {
    v -= vec3d{delta, 0, 0};
  }

  return delta;
}
double translate_shape_right(vector<vec3d> &pos, const double &delta) {

  for (auto &v : pos) {
    v += vec3d{delta, 0, 0};
  }

  return delta;
}
double translate_shape_left(vector<vec3d> &pos, const double &delta) {

  for (auto &v : pos) {
    v -= vec3d{delta, 0, 0};
  }

  return delta;
}
void translate_mesh(DrawableTrimesh<> &m0, DrawableTrimesh<> &m1) {

  auto &pos0 = m0.vector_verts();
  auto &pos1 = m1.vector_verts();

  translate_shape(pos0, pos1);
}
void translate_mesh_right(DrawableTrimesh<> &m, const double &delta) {

  auto &pos = m.vector_verts();

  translate_shape_right(pos, delta);
}
void translate_mesh_left(DrawableTrimesh<> &m, const double &delta) {

  auto &pos = m.vector_verts();

  translate_shape_left(pos, delta);
}
int main(int argc, char **argv) {
  auto name0 = std::string(argv[1]);
  auto name1 = std::string(argv[2]);
  DrawableSegmentSoup S;
  //:::::::::::::::::FAUST
  auto s0 = "../../../Models/CMCF" + name0 + "_remeshed.obj";
  auto s1 = "../../../Models/CMCF" + name1 + "_remeshed.obj";
  auto s2 = "../../../Models/CMCF" + name0 + ".obj";
  auto s3 = "../../../Models/CMCF" + name1 + ".obj";

  //:::::::::::::::::TOSCA
  // auto s0 = "../../../Models/CMCF" + name0 + "_remeshed.obj";
  // auto s1 = "../../../Models/CMCF" + name1 + "_remeshed.obj";
  // auto s2 = "../../../Models/CMCF" + name0 + ".obj";
  // auto s3 = "../../../Models/CMCF" + name1 + ".obj";

  //:::::::::::::::::.remeshed
  // auto s0 = "../../../Models/FAUST_remeshed/" + name0 + ".off";
  // auto s1 = "../../../Models/FAUST_remeshed/" + name1 + ".off";
  // auto s0 = "../../../Models/REMCMCF" + name0 + "_remeshed.obj";
  // auto s1 = "../../../Models/REMCMCF" + name1 + "_remeshed.obj";
  // auto s2 = "../../../Models/REMCMCF" + name0 + ".obj";
  // auto s3 = "../../../Models/REMCMCF" + name1 + ".obj";
  // auto gt0 =
  //     load_landmarks("../../../Models/FAUST_remeshed/corres/" + name0 +
  //     ".vts");
  // auto gt1 =
  //     load_landmarks("../../../Models/FAUST_remeshed/corres/" + name1 +
  //     ".vts");

  vector<vec3d> pos0, pos1, pos2, pos3;
  vector<vector<uint>> tris0, tris1, tris2, tris3;
  std::vector<vec3d> tex, nor;
  std::vector<std::vector<uint>> poly_tex, poly_nor;
  std::vector<Color> poly_col;
  std::vector<int> poly_lab;
  bool lndmark_picking = true;
  GLcanvas gui;
  auto n = 10;
  float point_size = 0.002;
  int lndmark_size = 5;
  vector<int> lndmarks;
  bool GH = true;

  unordered_map<int, int> mapping;
  unordered_map<int, int> m20;
  unordered_map<int, int> m31;
  unordered_map<int, int> m02;
  unordered_map<int, int> m13;
  string rawname0;
  string rawname1;

  // read_OFF(s0.c_str(), pos0, tris0);
  // read_OFF(s1.c_str(), pos1, tris1);
  read_OBJ(s0.c_str(), pos0, tex, nor, tris0, poly_tex, poly_nor, poly_col,
           poly_lab);
  read_OBJ(s1.c_str(), pos1, tex, nor, tris1, poly_tex, poly_nor, poly_col,
           poly_lab);

  read_OBJ(s2.c_str(), pos2, tex, nor, tris2, poly_tex, poly_nor, poly_col,
           poly_lab);
  read_OBJ(s3.c_str(), pos3, tex, nor, tris3, poly_tex, poly_nor, poly_col,
           poly_lab);

  m20 = shape_correspondence(pos2, pos0);
  m31 = shape_correspondence(pos3, pos1);

  m02 = shape_correspondence(pos0, pos2);
  m13 = shape_correspondence(pos1, pos3);
  filesystem::path p2(s2);
  filesystem::path p3(s3);
  rawname0 = p2.stem();
  rawname1 = p3.stem();

  clean_filename(rawname0, "CMCF");
  clean_filename(rawname1, "CMCF");
  auto filename = "landamarks" + rawname0 + "-" + rawname1;
  std::cout << filename << std::endl;
  fit_into_cube(pos0);
  fit_into_cube(pos1);
  fit_into_cube(pos2);
  fit_into_cube(pos3);
  auto delta = translate_shape(pos0, pos1);
  translate_shape(pos2, pos3);
  auto pc0 = init_pc(pos0);
  DrawableTrimesh<> m0(pos0, tris0);
  // patch_fitting(pc0, m0, 1e-3);
  vector<double> f_min;
  vector<double> f_max;
  auto pc1 = init_pc(pos1);

  DrawableTrimesh<> m1(pos1, tris1);
  // patch_fitting(pc1, m1, 1e-3);

  auto pc2 = init_pc(pos2);
  DrawableTrimesh<> m2(pos2, tris2);
  // patch_fitting(pc2, m2, 1e-3);

  auto pc3 = init_pc(pos3);
  DrawableTrimesh<> m3(pos3, tris3);
  // patch_fitting(pc3, m3, 1e-3);

  pc0.solver = make_geodesic_solver(m0);
  pc1.solver = make_geodesic_solver(m1);
  // pc2.solver = make_geodesic_solver(m2);
  // pc3.solver = make_geodesic_solver(m3);

  auto V0 = m0.num_verts();
  auto V1 = m1.num_verts();

  auto A0 = std::sqrt(m0.mesh_area());
  auto A1 = std::sqrt(m1.mesh_area());
  auto L0 = laplacian(m0, COTANGENT);
  auto L1 = laplacian(m1, COTANGENT);
  auto Grad0 = gradient_matrix(m0, false);
  auto Grad1 = gradient_matrix(m1, false);

  vector<double> eig0;
  vector<double> eig1;
  vector<double> eig4;
  vector<double> eig5;

  auto nf = 50;
  matrix_eigenfunctions(L0, true, nf, eig0, f_min, f_max);
  matrix_eigenfunctions(L1, true, nf, eig1, f_min, f_max);

  vector<int> paired_centers;
  ScalarField phi0, phi1;
  vector<double> field0(V0);
  vector<double> field1(V1);

  for (auto i = 0; i < V0; ++i) {
    field0[i] = eig0[V0 + i];
  }
  for (auto i = 0; i < V1; ++i) {
    field1[i] = eig1[V1 + i];
  }
  auto gradient0 = compute_grad_cino(Grad0, field0);
  auto gradient1 = compute_grad_cino(Grad1, field1);
  auto centers0 = singular_vertices(m0, field0);
  clean_singularities(m0, gradient0, centers0);

  auto centers1 = singular_vertices(m1, field1);
  clean_singularities(m1, gradient1, centers1);

  auto sing0 = singularities(m0, field0);
  auto sing1 = singularities(m1, field1);
  std::cout << "the first shape has " << centers0.size() << " critical values"
            << std::endl;

  std::cout << "the first shape has " << centers1.size() << " critical values"
            << std::endl;
  // auto centers4 = singular_vertices(m4, field4);
  // auto centers5 = singular_vertices(m5, field5);
  // auto center0_orginal = vector<int>(centers0.size());
  // auto center1_orginal = vector<int>(centers1.size());

  // for (auto i = 0; i < centers0.size(); ++i)
  //   center0_orginal[i] = m02.at(centers0[i]);

  // for (auto j = 0; j < centers0.size(); ++j)
  //   center1_orginal[j] = m13.at(centers1[j]);

  // vector<vector<double>> c0(centers0.size());
  // vector<vector<double>> c1(centers1.size());

  // for (auto i = 0; i < centers0.size(); ++i) {
  //   c0[i] = exact_geodesic_distance(m0, centers0[i]);
  //   normalize_field(c0[i], A0);
  // }

  // for (auto i = 0; i < centers1.size(); ++i) {
  //   c1[i] = exact_geodesic_distance(m1, centers1[i]);
  //   normalize_field(c1[i], A1);
  // }

  std::cout << "Average edge length is" << m0.edge_avg_length() << std::endl;
  auto nbr_lnd = 5;
  vector<vector<double>> d0(nbr_lnd, vector<double>(m0.num_verts()));
  vector<vector<double>> d1(nbr_lnd, vector<double>(m1.num_verts()));

  ///::::::::::::::::::::::::::::::::::::::::FAUST
  // LANDMARKS TEASER
  // auto l0 = vector<int>{3748, 1649, 4342, 906, 4561};
  // auto l1 = l0;
  // LANDMARKS GT PERFECT
  // auto l0 = vector<int>{412, 2445, 3219, 6617, 5907};
  // auto l1 = l0;
  // LANDMARKS GT
  // auto l0 = vector<int>{0, 3359, 6759, 4933, 1795};
  // LANDMARKS GT dog
  // auto l0 = vector<int>{7530, 6850, 5674, 4662, 1576, 5189};
  // auto l1 = l0;
  // LANDMARKS GT hippo-lion
  // auto l0 = vector<int>{30913, 37873, 47124, 51082, 17218};
  // auto l1 = l0;

  // // LANDMARKS GT gorilla
  // auto l0 = vector<int>{11375, 17273, 20828, 24472, 4228};
  // auto l1 = l0;

  // LANDMARKS GT Michael
  // auto l0 = vector<int>{28313, 37873, 47124, 51082, 17217};
  // auto l1 = l0;

  auto seed0 = m20.at(0);
  auto seed1 = m31.at(0);
  auto l0 = farthest_point_sampling(pc0.solver, m0, seed0, nbr_lnd);
  auto l1 = farthest_point_sampling(pc1.solver, m1, seed1, nbr_lnd);

  ///::::::::::::::::::::::::::::::::::::::::FAUST REMESHED
  // LANDMARKS GT REMESHED
  // auto entries = vector<int>{1994, 1621, 967, 4537, 307};
  // LANDMARKS GT PERFECT REMESHED
  // auto entries = vector<int>{0, 4317, 2029, 2779, 783};

  // auto l0 = constrained_farthest_point_sampling(pc0.solver, m0, gt0,
  //                                               m20.at(gt0[0] - 1), nbr_lnd);
  // auto l1 = constrained_farthest_point_sampling(pc1.solver, m1, gt1,
  //                                               m31.at(gt1[0] - 1), nbr_lnd);

  // auto l0 = vector<int>(nbr_lnd);
  // auto l1 = vector<int>(nbr_lnd);

  // for (auto i = 0; i < nbr_lnd; ++i) {
  //   l0[i] = gt0[entries[i]] - 1;
  //   l1[i] = gt1[entries[i]] - 1;
  // }
  std::cout << "On the first shape we have these landmarks:" << std::endl;
  for (auto i = 0; i < l0.size(); ++i) {
    std::cout << m02.at(l0[i]) + 1 << std::endl;
  }

  std::cout << "On the second shape we have these landmarks:" << std::endl;
  for (auto i = 0; i < l1.size(); ++i) {
    std::cout << m13.at(l1[i]) + 1 << std::endl;
  }
  lndmarks.resize(2 * nbr_lnd);

  for (auto i = 0; i < 2 * nbr_lnd; i += 2) {
    // FPS on original shape
    // lndmarks[i] = m20.at(l0[i / 2]);
    // lndmarks[i + 1] = m31.at(l1[i / 2]);
    //  FPS on remeshed shape
    lndmarks[i] = l0[i / 2];
    lndmarks[i + 1] = l1[i / 2];

    d0[i / 2] = exact_geodesic_distance(m0, lndmarks[i]);
    // compute_geodesic_distances(pc0.solver, {lndmarks[i]});
    d1[i / 2] = exact_geodesic_distance(m1, lndmarks[i + 1]);
    // compute_geodesic_distances(pc1.solver, {lndmarks[i + 1]});
  }

  // for (auto i = 0; i < c0.size(); ++i) {
  //   normalize_field(c0[i], A0);
  // }
  // for (auto i = 0; i < c1.size(); ++i) {
  //   normalize_field(c1[i], A1);
  // }
  // for (auto i = 0; i < d0.size(); ++i) {
  //   normalize_field(d0[i], A0);
  //   normalize_field(d1[i], A1);
  // }

  bool show_eig = false;
  bool show_sing = false;
  bool show_vor = false;
  bool show_lnd = false;
  bool show_grad = false;
  bool corr0 = false;
  bool corr1 = false;
  bool show_m0 = true;
  bool show_m1 = true;
  bool show_edges = true;
  DrawableVectorField G0;
  DrawableVectorField G1;

  int k = 0;
  int k0 = 0;
  int k1 = 0;
  int selected_center = centers0[k];
  Eigen::MatrixXd D;
  iVd ivd0;
  iVd ivd1;
  iVd ivd4;
  iVd ivd5;

  ivd0 = intrinsic_voronoi_diagram(pc0.solver, m0, centers0);
  ivd1 = intrinsic_voronoi_diagram(pc1.solver, m1, centers1);

  // auto ivd4 = intrinsic_voronoi_diagram(solver4, m4, center0_orginal);
  // auto ivd5 = intrinsic_voronoi_diagram(solver5, m5, center1_orginal);

  gui.draw_side_bar();
  draw_pc(pc0, gui);
  draw_pc(pc1, gui);
  gui.push(&m0, false);
  gui.push(&m1, false);

  gui.callback_app_controls = [&]() {
    if (ImGui::Checkbox("Show Eigenfunctions", &show_eig)) {
      if (show_eig) {

        phi0 = ScalarField(field0);
        phi1 = ScalarField(field1);

        phi0.normalize_in_01();
        phi0.copy_to_mesh(m0);
        m0.show_texture1D(TEXTURE_1D_HSV);
        phi1.normalize_in_01();
        phi1.copy_to_mesh(m1);
        m1.show_texture1D(TEXTURE_1D_HSV);
      }
    }
    if (ImGui::Button("Export Eigenfunctions and critical valued")) {
      string foldername = "../teaser/";
      export_field(field0, foldername + "Fiedler_" + name0);
      export_field(field1, foldername + "Fiedler_" + name1);
      export_centers(centers0, name0, foldername);
      export_centers(centers1, name1, foldername);
    }
    if (ImGui::Checkbox("Show Gradient", &show_grad)) {
      if (show_grad) {

        auto grad0 = compute_grad_cino(Grad0, field0);
        G0 = DrawableVectorField(grad0, pc0.positions);
        auto grad1 = compute_grad_cino(Grad1, field1);
        G1 = DrawableVectorField(grad1, pc1.positions);
        G0.set_arrow_size(0.003);
        G1.set_arrow_size(0.003);
        gui.push(&G0);
        gui.push(&G1);
      } else {
        gui.pop(&G0);
        gui.pop(&G1);
      }
    }
    if (ImGui::Button("CMCF")) {

      MCF(m0, 12, 1e-5, true);
      MCF(m1, 12, 1e-5, true);

      translate_mesh(m0, m1);
      m0.updateGL();
      m1.updateGL();
    }
    ImGui::SameLine();
    if (ImGui::Button("Export meshes")) {
      auto name0 = "CMFC" + rawname0 + ".obj";
      auto name1 = "CMFC" + rawname1 + ".obj";
      write_OBJ(name0.c_str(), obj_wrapper(m0.vector_verts()),
                m0.vector_polys());
      write_OBJ(name1.c_str(), obj_wrapper(m1.vector_verts()),
                m1.vector_polys());
    }
    if (ImGui::Button("Export geodesic matrix")) {
      auto name0 = "geodesic_matrix" + rawname0 + ".txt";
      auto M = geodesic_matrix(pc0.solver);
      export_matrix(M, name0);
    }
    if (ImGui::Checkbox("Show landmarks", &show_lnd)) {
      if (show_lnd) {
        highlight_lndmarks(pc0, pc1, lndmarks);
      } else {
        reset(pc0, point_size);
        reset(pc1, point_size);
      }
    }
    if (ImGui::Checkbox("Show Singularities", &show_sing)) {
      if (field0.size() > 0 && show_m0) {
        if (show_sing) {
          show_singularities(pc0, sing0, 0.03);
        } else
          reset(pc0, point_size);
      }
      if (field1.size() > 0) {
        if (show_sing) {
          show_singularities(pc1, sing1, 0.03);
        } else
          reset(pc1, point_size);
      }
    }
    if (ImGui::Checkbox("Show Voronoi", &show_vor)) {
      if (show_vor) {
        if (mapping.size() > 0) {
          auto cols = show_paired_voronoi_regions(m0, ivd0, centers0, m1, ivd1,
                                                  centers1, mapping);
          highlight_paired_centers_original(pc0, pc1, centers0, centers1,
                                            mapping, cols);
        } else {
          show_voronoi_regions(m0, ivd0, centers0);
          show_voronoi_regions(m1, ivd1, centers1, 1);
          reset(pc0, point_size);
          highlight_points(pc0, centers0);
          reset(pc1, point_size);
          highlight_points(pc1, centers1);
        }

        // show_paired_voronoi_regions_original(m0, ivd0, centers0, m1, ivd1,
        //                                      centers1);

        // auto acc =
        // accuracy_of_region_mapping(ivd0, centers0, ivd1, centers1, c0,
        // c1,
        //                            mapping, m20, m02, m13, m31);
        // accuracy_of_region_mapping_GT(ivd0, ivd1, ivd4, ivd4, m02);
        // std::cout << "Accuracy of the mapping is " << acc << std::endl;
        //  show_voronoi_regions(m1, ivd1, centers1);

      } else {
        reset(pc0, point_size);
        reset(pc1, point_size);

        m0.show_poly_color();
        m1.show_poly_color();
      }
    }
    if (ImGui::Button("Save Landmarks")) {
      if (lndmarks.size() > 0 && (int)lndmarks.size() % 2 == 0) {
        save_landmarks(lndmarks, filename);
      }
    }
    // ImGui::SameLine();
    // if (ImGui::Button("Load Landmarks")) {
    //   lndmarks = load_landmarks(filename);
    //   reset(pc0, point_size);
    //   reset(pc1, point_size);
    //   highlight_lndmarks(pc0, pc1, lndmarks);
    // }

    if (ImGui::SliderInt("Choose center on source shape", &k, 0,
                         centers0.size() - 1))
      selected_center = centers0[k];
    if (ImGui::SliderInt("Show center on source shape", &k0, 0,
                         centers0.size() - 1)) {
      reset(pc0, point_size);
      higlight_point(pc0, centers0[k0], lndmark_size);
    }
    if (ImGui::SliderInt("Show center on target shape", &k1, 0,
                         centers1.size() - 1)) {
      reset(pc1, point_size);
      higlight_point(pc1, centers1[k1], lndmark_size);
    }

    if (ImGui::Button("Show best matching center")) {
      // reset(pc0, point_size);
      // reset(pc1, point_size);
      // highlight_lndmarks(pc0, pc1, lndmarks);
      // auto n0 = centers0.size();
      // auto n1 = centers1.size();
      // auto n = (int)lndmarks.size();
      // auto Des0 = compute_descriptors(m0, centers0, c0);
      // auto Des1 = compute_descriptors(m1, centers1, c1);

      // auto best = -1;
      // auto res = DBL_MIN;
      // for (auto i = 0; i < centers1.size(); ++i) {
      //   auto curr_res = p_descr(Des0[k], Des1[i], 1);
      //   if (curr_res > res) {
      //     res = curr_res;
      //     best = i;
      //   }
      // }
      // higlight_point(pc0, selected_center, lndmark_size);
      // higlight_point(pc1, centers1[best], lndmark_size);
    }
    if (ImGui::Button("Show Best Pairing")) {
      // auto n0 = centers0.size();
      // auto n1 = centers1.size();
      // auto n = (int)lndmarks.size();
      // auto Des0 = compute_descriptors(m0, centers0, c0);
      // auto Des1 = compute_descriptors(m1, centers1, c1);

      // auto best0 = -1;
      // auto best1 = -1;
      // auto res = DBL_MIN;
      // for (auto j = 0; j < centers0.size(); ++j) {
      //   for (auto i = 0; i < centers1.size(); ++i) {
      //     auto curr_res = p_descr(Des0[j], Des1[i], 1);
      //     if (curr_res > res) {
      //       res = curr_res;
      //       best0 = j;
      //       best1 = i;
      //     }
      //   }
      // }
      // higlight_point(pc0, centers0[best0], lndmark_size);
      // higlight_point(pc1, centers1[best1], lndmark_size);
    }
    if (ImGui::Button("Compute Pairing")) {
      if (n > 0 && (int)n % 2 == 0) {
        if (show_vor) {
          show_vor = false;
          m0.show_poly_color();
          m1.show_poly_color();
        }
        reset(pc0, point_size);
        reset(pc1, point_size);
        if (d0.size() == 0) {

          nbr_lnd = lndmarks.size() / 2;
          d0.resize(nbr_lnd, vector<double>(m0.num_verts()));
          d1.resize(nbr_lnd, vector<double>(m1.num_verts()));
          for (auto i = 0; i < 2 * nbr_lnd; i += 2) {
            d0[i / 2] = exact_geodesic_distance(m0, lndmarks[i]);
            // compute_geodesic_distances(pc0.solver, {lndmarks[i]});
            d1[i / 2] = exact_geodesic_distance(m1, lndmarks[i + 1]);
            // compute_geodesic_distances(pc1.solver, {lndmarks[i + 1]});
          }
        }
        mapping = // voronoi_mapping(centers0, centers1, m0, m1, c0, c1);
            voronoi_mapping(centers0, centers1, lndmarks, m0, m1, A0, A1, d0,
                            d1, GH);

        paired_centers.resize(2 * centers0.size());
        for (auto i = 0; i < centers0.size(); ++i) {
          paired_centers[2 * i] = centers0[i];
          paired_centers[2 * i + 1] = mapping.at(centers0[i]);
        }
        highlight_lndmarks(pc0, pc1, paired_centers);
        ivd0 = intrinsic_voronoi_diagram(pc0.solver, m0, centers0);
        ivd1 = intrinsic_voronoi_diagram(pc1.solver, m1, centers1);
        // show_paired_voronoi_regions(m0, ivd0, centers0, m1, ivd1, centers1,
        //                             mapping);
      }
    }

    if (ImGui::Button("Export regions")) {

      string foldername = "../REGIONS" + rawname0 + "-" + rawname1;
      filesystem::create_directories(foldername);
      if (mapping.size() > 0)
        export_regions(ivd0, centers0, ivd1, centers1, lndmarks, mapping, m20,
                       m02, m13, m31, pos2.size(), pos3.size(), foldername,
                       rawname0, rawname1);
      else
        export_regions_general(ivd0, centers0, ivd1, centers1, m20, m02, m13,
                               m31, m2.num_verts(), m3.num_verts(), foldername,
                               name0, name1);
    }

    if (ImGui::Checkbox("Test correspondence on first shape", &corr0)) {
      if (corr0) {
        reset(pc0, point_size);
        reset(pc1, point_size);
        gui.pop(&m1);
        remove_pc(pc1, gui);
        gui.push(&m2, false);
        draw_pc(pc2, gui);

        auto [vor2, centers2] =
            transfer_voronoi_diagram(ivd0, centers0, m20, m02, m0, m2);
        show_paired_voronoi_regions(pc2, vor2, centers2, pc0, ivd0, centers0,
                                    m20);
      } else {
        gui.pop(&m2);
        remove_pc(pc2, gui);
        draw_pc(pc1, gui);
        gui.push(&m1, false);
      }
    }
    if (ImGui::Checkbox("Test correspondence on second shape", &corr1)) {
      if (corr1) {
        reset(pc0, point_size);
        reset(pc1, point_size);
        gui.pop(&m0);
        remove_pc(pc0, gui);
        gui.push(&m3, false);
        draw_pc(pc3, gui);
        auto [vor3, centers3] =
            transfer_voronoi_diagram(ivd1, centers1, m31, m13, m1, m3);
        show_paired_voronoi_regions(pc3, vor3, centers3, pc1, ivd1, centers1,
                                    m31);
      } else {
        gui.pop(&m3);
        remove_pc(pc3, gui);
        gui.push(&m0, false);
        draw_pc(pc0, gui);
      }
    }
    if (ImGui::Checkbox(" Show M0", &show_m0)) {
      if (show_m0) {
        gui.push(&m0, false);

      } else {
        gui.pop(&m0);
        reset(pc0, point_size);
      }
    }
    if (ImGui::Checkbox(" Show M1", &show_m1)) {
      if (show_m1) {
        gui.push(&m1, false);

      } else {
        gui.pop(&m1);
        reset(pc1, point_size);
      }
    }

    if (ImGui::Checkbox(" Show Wirefram", &show_edges)) {
      if (show_edges) {
        m0.show_wireframe(true);
        m1.show_wireframe(true);
      } else {
        m0.show_wireframe(false);
        m1.show_wireframe(false);
      }
    }
    if (ImGui::SliderFloat("Point Size", &point_size, 0, 0.1)) {
      reset(pc0, point_size, true);
      reset(pc1, point_size, true);
      reset(pc2, point_size, true);
      reset(pc3, point_size, true);
    }

    ImGui::SliderInt("Landmark Size", &lndmark_size, 1, 10);
  };
  gui.callback_mouse_left_click = [&](int modifiers) -> bool {
    if (modifiers & GLFW_MOD_SHIFT) {
      vec3d p;
      vec2d click = gui.cursor_pos();
      auto intersect = gui.unproject(click, p);
      if (intersect && !lndmark_picking) // transform click in a 3d point
      {
        auto inter = pick_vert(pos0, p);
        std::cout << "landmark picked at" << m02.at(inter) << std::endl;
        // auto d = exact_geodesic_distance(m0, inter);
        // auto c = create_circle(m0, inter, 0.1, d);
        // auto pos = circle_positions(m0, c);
        // for (auto i = 0; i < pos.size(); ++i) {
        //   for (auto j = 0; j < pos[i].size() - 1; ++j)
        //     S.push_seg(pos[i][j], pos[i][j + 1]);
        // }
        // gui.push(&S, false);
        // auto inter = pick_vert(pos0, gt0, p);
        highlight_points(pc0, {inter});
        // auto it = find(gt0.begin(), gt0.end(), inter + 1);
        // std::cout << "landmark picked at entry" << distance(gt0.begin(),
        // it)
        //           << std::endl;
        // if (inter.y() == 0)
        //   highlight_points(pc0, {inter.x()});
        // else
        //   highlight_points(pc1, {inter.x()});

      } else if (intersect && lndmark_picking) {
        reset(pc0, point_size);
        reset(pc1, point_size);
        auto inter = pick_vert(pos0, pos1, p);

        if (inter.y() == 0 && lndmarks.size() % 2 == 1)
          lndmarks.back() = inter.x();
        else if (inter.y() == 1 && lndmarks.size() % 2 == 0)
          lndmarks.back() = inter.x();
        else
          lndmarks.push_back(inter.x());

        highlight_lndmarks(pc0, pc1, lndmarks);
      }
    }
    return false;
  };

  return gui.launch();
}
