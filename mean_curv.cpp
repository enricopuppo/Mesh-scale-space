#include "diff_geo.h"
#include "utilities.h"
#include <cinolib/gl/glcanvas.h>
#include <cinolib/io/write_OBJ.h>
#include <cinolib/io/write_OFF.h>
#include <cinolib/mean_curv_flow.h>
#include <cinolib/meshes/drawable_trimesh.h>
#include <fstream>
using namespace std;
using namespace cinolib;

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//                                utility

int main(int argc, char **argv) {

  // auto s0 = "../finals/FAUST_shapes_off/" + std::string(argv[1]) + ".off";
  // auto s1 = "../finals/FAUST_shapes_off/" + std::string(argv[2]) + ".off";
  vector<string> names = {"lion", "hippo"};

  // vector<int> entries = {0, 1, 2, 3, 4, 5}; // centaur
  // vector<int> entries = {0, 7}; // hand
  //  vector<int> entries = {0, 1, 2, 3, 5, 6, 7, 8, 10}; // dog
  // vector<int> entries = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}; // cat
  // vector<int> entries = {0, 5, 6, 7, 10, 14, 15, 17}; // horse

  // auto M = read_binary_matrix("michael10");
  // auto diam = diameter(M);
  // vector<string> methods = {"asa", "dif", "std"};
  // auto entries = vector<int>{0, 7};
  //  for (auto i = 0; i < 10; ++i) {
  //    for (auto j = 0; j < 10; ++j) {
  ///:::::::::::::::::::ERROR HEAT MAP MICHAEL:::::::::::::::::::
  // for (auto name : names) {
  //   for (auto method : methods) {
  //     string filename = name + "_" + method + ".txt";
  //     auto T = load_landmarks(filename);
  //     auto err = error_on_michaels(M, T, diam);
  //     write_file(err, "../err_" + filename);
  //   }
  // }

  ///:::::::::::::::::::GEODESIC MATRIX and CMCF:::::::::::::::::::
  for (auto name : names) {
    // for (auto i : entries) {

    // for (auto i = 0; i < 10; ++i) {
    //   for (auto j = 0; j < 10; ++j) {
    vector<vec3d> pos;
    vector<vector<uint>> tris;
    std::vector<vec3d> tex, nor;
    std::vector<std::vector<uint>> poly_tex, poly_nor;
    std::vector<Color> poly_col;
    std::vector<int> poly_lab;
    auto s = "../../../Models/" + name + "_remeshed.off";
    // auto s = "../../../Models/TOSCA/Meshes/horse" + to_string(i) + ".off";
    // auto s = "../../../Models/hand" + to_string(i) + ".off";
    // auto s = "../../../Models/dog" + to_string(i) + "_remeshed.off";
    // auto s = "../../../Models/FAUST_shapes_off/tr_reg_0" + to_string(i) +
    //          to_string(j) + ".off";
    //  read_OBJ(s.c_str(), pos, tex, nor, tris, poly_tex, poly_nor, poly_col,
    //           poly_lab);
    read_OFF(s.c_str(), pos, tris);
    // filesystem::path p(s);
    // string rawname = p.stem();
    // string str = "_remeshed";
    // rawname.erase(rawname.find(str), str.length());
    DrawableTrimesh<> m(pos, tris);

    MCF(m, 12, 1e-5, true);

    auto filename = "../../../Models/CMCF" + name + ".obj";

    write_OBJ(filename.c_str(), obj_wrapper(m.vector_verts()),
              m.vector_polys());
    // auto solver = make_geodesic_solver(m);
    //  auto pc = init_pc(pos);
    //  patch_fitting(pc, m, 1e-3);
    //  auto M = geodesic_matrix(solver);
    //  auto filename = "../geodesic_matrix" + name;
    //  write_binary_matrix(name.c_str(), M);
  }

  return 0;
}
