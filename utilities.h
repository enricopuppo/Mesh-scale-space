#ifndef UTILITIES_H
#define UTILITIES_H

#include "diff_geo.h"
//#include <cinolib/drawable_curve.h>
#include <cinolib/drawable_segment_soup.h>
#include <cinolib/drawable_sphere.h>
#include <cinolib/drawable_vector_field.h>
#include <cinolib/meshes/drawable_trimesh.h>
#include <cinolib/triangle_wrap.h>
#include <filesystem>
using namespace std;
using namespace cinolib;

void clean_filename(string &filename, const string &substring);

vector<double> obj_wrapper(const vector<vec3d> &pos);

DrawableVectorField show_normals(const point_cloud &pc, const bool &Monge);

int max_curvature_point(const point_cloud &pc);

int min_curvature_point(const point_cloud &pc);

void find_local_maxima(point_cloud &pc, const vector<double> &f);

void weights_heat_map(point_cloud &pc, const int vid);

void weights_heat_map(point_cloud &pc, const int vid, const Eigen::MatrixXd &W,
                      const vector<int> &nbr);

void total_weights_heat_map(point_cloud &pc);

DrawableTrimesh<> draw_patch(const point_cloud &pc, const int vid,
                             const bool Monge = true);

void reset_weights_heat_map(point_cloud &pc, const int vid);

void detect_sharp_features(point_cloud &pc);

void higlight_point(point_cloud &pc, const int vid, const int factor);

void highlight_points(point_cloud &pc, const vector<int> &points,
                      const double &size = -1);

vector<Color>
show_paired_voronoi_regions(DrawableTrimesh<> &m0, const iVd &ivd0,
                            const vector<int> &centers0, DrawableTrimesh<> &m1,
                            const iVd &ivd1, const vector<int> &centers1,
                            const std::unordered_map<int, int> &m02,
                            const std::unordered_map<int, int> &m31);

void highlight_lndmarks(point_cloud &pc0, point_cloud &pc1,
                        const vector<int> &lndmarks, const double &size = -1);

void highlight_paired_centers_original(point_cloud &pc0, point_cloud &pc1,
                                       const iVd &vor1,
                                       const vector<int> &centers0,
                                       const vector<int> &centers1,
                                       const vector<Color> &cols);
void highlight_paired_centers_original(point_cloud &pc0, point_cloud &pc1,
                                       const vector<int> &centers0,
                                       const vector<int> &centers1,
                                       const unordered_map<int, int> &mapping,
                                       const vector<Color> &cols);

void show_candidates(point_cloud &pc, const vector<double> &phi, const int V,
                     const int nf);

void show_voronoi_regions(DrawableTrimesh<> &m, const iVd &intrinsic_voronoi,
                          const vector<int> &voronoi_centers,
                          const int offset = 0);

void show_voronoi_regions_and_centers(DrawableTrimesh<> &m, point_cloud &pc,
                                      const iVd &intrinsic_voronoi,
                                      const vector<int> &voronoi_centers,
                                      const double &rad, const int offset);

void show_voronoi_regions(DrawableTrimesh<> &m, const iVd &intrinsic_voronoi,
                          const vector<int> &voronoi_centers,
                          const std::unordered_map<int, int> &mapping);
vector<Color>
show_paired_voronoi_regions(DrawableTrimesh<> &m0, const iVd &ivd0,
                            const vector<int> &centers0, DrawableTrimesh<> &m1,
                            const iVd &ivd1, const vector<int> &centers1,
                            const std::unordered_map<int, int> &mapping);

void residual_heat_map(point_cloud &pc);

void show_outliers(point_cloud &pc);

void show_centers(point_cloud &pc);

void show_patch_tagging(point_cloud &pc);

DrawableTrimesh<> draw_patches(const point_cloud &pc);

DrawableSegmentSoup draw_gradient_field(const point_cloud &pc,
                                        const vector<vec3d> &gradient,
                                        const double h);

DrawableSegmentSoup draw_local_gradient_field(const point_cloud &pc,
                                              const int vid,
                                              const vector<vec3d> &gradient,
                                              const double &r, const double &h);

DrawableSegmentSoup draw_local_princ_dir_field(const point_cloud &pc,
                                               const int vid,
                                               const vector<vec3d> &dir,
                                               const double &r,
                                               const double &h);

void reset(point_cloud &pc, const double &point_size,
           const bool original_color = false);

DrawableSegmentSoup show_CH(const point_cloud &pc, const int vid);

DrawableSegmentSoup isophotic_segmentation(const point_cloud &pc,
                                           const vector<vec3d> &k1,
                                           const vector<vec3d> &k2);

void color_points_according_to_quadric(point_cloud &pc, const int vid);

void color_points_according_to_quadric(point_cloud &pc, const vector<int> &nbr,
                                       const patch &p);
DrawableTrimesh<> draw_primitive(const point_cloud &pc, const iVd &vor,
                                 const vector<int> &centers, const int center,
                                 const patch &p, const int type);

DrawableTrimesh<> draw_primitive(const point_cloud &pc, const int vid,
                                 const patch &p, const int type);

DrawableTrimesh<> draw_patch(const point_cloud &pc, const patch &p);

std::tuple<vector<DrawableSphere>, DrawableTrimesh<>>
show_tagent_plane_mapping(const point_cloud &pc, const int vid);

void show_secondary_patches(point_cloud &pc);

void show_singularities(point_cloud &pc, const vector<double> &f);

void show_singularities(point_cloud &pc, const DrawableTrimesh<> &m,
                        const vector<double> &f);

void show_singularities(point_cloud &pc, const vector<vec2i> &sing,
                        const double &size = -1);

void show_singularities(point_cloud &pc);

void higlight_point(point_cloud &pc, const int vid, const int factor);

void update_patch_tagging(point_cloud &pc, const int vid);

DrawableTrimesh<> draw_Monge_patch(const patch &p);

DrawableTrimesh<> draw_parabolic_cylinder(const point_cloud &pc,
                                          const patch &p);

DrawableSegmentSoup draw_line_in_tangent_space(const point_cloud &pc,
                                               const int vid,
                                               const double &theta, Color &col);

void save_landmarks(const vector<int> &lndmarks, const string &filename);

vector<int> load_landmarks(const string &filename);

Eigen::SparseMatrix<double> import_laplacian(const string &filename,
                                             const int V);

std::tuple<vector<string>, vector<string>> set_pairing_for_non_iso_matching();

void export_field(const vector<double> &field, const string &name);

void export_lndmarks(const vector<int> &lndmarks, const string &name,
                     const string &foldername);
void export_lndmarks(const vector<int> &lndmarks, const string &name0,
                     const string &name1, const string &foldername);
void export_centers(const vector<int> &centers, const string &name,
                    const string &foldername);
bool export_regions(
    const iVd &vor0, const vector<int> &centers0, const iVd &vor1,
    const vector<int> &centers1, const vector<int> &landmarks,
    const unordered_map<int, int> &mapping, const unordered_map<int, int> &m20,
    const unordered_map<int, int> &m02, const unordered_map<int, int> &m13,
    const unordered_map<int, int> &m31, const int V2, const int V3,
    const string &foldername, const string &name0, const string &name1);

bool export_regions_GT_remeshed(
    const iVd &vor0, const vector<int> &centers0, const iVd &vor1,
    const vector<int> &centers1, const vector<int> &l0, const vector<int> &l1,
    const unordered_map<int, int> &mapping, const unordered_map<int, int> &m20,
    const unordered_map<int, int> &m02, const unordered_map<int, int> &m13,
    const unordered_map<int, int> &m31, const int V2, const int V3,
    const string &foldername, const string &name0, const string &name1);

bool export_regions_GT(
    const iVd &vor0, const vector<int> &centers0, const iVd &vor1,
    const vector<int> &centers1, const vector<int> &landmarks,
    const unordered_map<int, int> &mapping, const unordered_map<int, int> &m20,
    const unordered_map<int, int> &m02, const unordered_map<int, int> &m13,
    const unordered_map<int, int> &m31, const int V2, const int V3,
    const string &foldername, const string &name0, const string &name1);

bool export_regions_GT(const iVd &vor0, const vector<int> &centers0,
                       const iVd &vor1, const vector<int> &centers1,
                       const unordered_map<int, int> &m20,
                       const unordered_map<int, int> &m02,
                       const unordered_map<int, int> &m13,
                       const unordered_map<int, int> &m31, const int V2,
                       const int V3, const string &foldername,
                       const string &name0, const string &name1);

bool export_regions_general(const iVd &vor0, const vector<int> &centers0,
                            const iVd &vor1, const vector<int> &centers1,
                            const unordered_map<int, int> &m20,
                            const unordered_map<int, int> &m02,
                            const unordered_map<int, int> &m13,
                            const unordered_map<int, int> &m31, const int V2,
                            const int V3, const string &foldername,
                            const string &name0, const string &name1);
vec3d accuracy_of_region_mapping(
    const iVd &vor0, const vector<int> &centers0, const iVd &vor1,
    const vector<int> &centers1, const vector<vector<double>> &c0,
    const vector<vector<double>> &c1, const unordered_map<int, int> &mapping,
    const unordered_map<int, int> &m20, const unordered_map<int, int> &m02,
    const unordered_map<int, int> &m13, const unordered_map<int, int> &m31);

double accuracy_of_region_mapping_GT(const iVd &vor0, const iVd &vor1,
                                     const unordered_map<int, int> &m20,
                                     const unordered_map<int, int> &m31,
                                     const unordered_map<int, int> &m13,
                                     const unordered_map<int, int> &m02,
                                     const int V5);

double accuracy_of_region_mapping_GT(const iVd &vor0, const iVd &vor1,
                                     const int V0);

std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> compute_indicator_functions(
    const iVd &vor0, const vector<int> &centers0, const iVd &vor1,
    const vector<int> &centers1, const unordered_map<int, int> &mapping,
    const unordered_map<int, int> &m20, const unordered_map<int, int> &m31,
    const int V2, const int V3);

iVd voronoi_diagram_from_matrix(const Eigen::MatrixXd &F,
                                const vector<int> &centers);

std::tuple<iVd, vector<int>> import_voronoi_diagrams(const string &name,
                                                     const string folder);
void export_matrix(const Eigen::MatrixXd &M, const string &filename);
unordered_map<int, int> shape_correspondence(const vector<vec3d> &pos0,
                                             const vector<vec3d> &pos1);

std::tuple<iVd, vector<int>> transfer_voronoi_diagram(
    const iVd &vor0, const vector<int> &centers0,
    const unordered_map<int, int> &m20, const unordered_map<int, int> &m02,
    const DrawableTrimesh<> &m0, const DrawableTrimesh<> &m2);

void show_paired_voronoi_regions(point_cloud &pc0, const iVd &ivd0,
                                 const vector<int> &centers0, point_cloud &pc1,
                                 const iVd &ivd1, const vector<int> &centers1,
                                 const std::unordered_map<int, int> &mapping);

vector<Color> show_paired_voronoi_regions_original(
    DrawableTrimesh<> &m0, const iVd &ivd0, const vector<int> &centers0,
    DrawableTrimesh<> &m1, const iVd &ivd1, const vector<int> &centers1,
    const int offset = 0);

void write_binary_matrix(const char *filename, const Eigen::MatrixXd &matrix);
void write_file(const vector<double> &v, const string &filename);
Eigen::MatrixXd read_binary_matrix(const char *filename);
std::tuple<DrawableSegmentSoup, DrawableSegmentSoup>
draw_curvature_cross_field(const vector<vec3d> &positions,
                           const vector<vec3d> &k1, const vector<vec3d> &k2,
                           const double h);

vector<double> error_on_michaels(const Eigen::MatrixXd &distances,
                                 const vector<int> &T, const double &diam);
inline vec3f random_color() {
  auto result = vec3f{0, 0, 0};
  for (auto i = 0; i < 3; ++i) {

    result[i] = (float)rand() / RAND_MAX;
  }
  return result;
}
#endif
