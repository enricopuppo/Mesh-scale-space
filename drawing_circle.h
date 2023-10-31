#pragma once
#ifndef DRAWING_CIRCLE_H
#define DRAWING_CIRCLE_H

#include <cinolib/geometry/triangle_utils.h>
#include <cinolib/geometry/vec_mat.h>
#include <cinolib/laplacian.h>
#include <cinolib/linear_solvers.h>
#include <cinolib/matrix_eigenfunctions.h>
#include <cinolib/octree.h>
#include <cinolib/profiler.h>
#include <cinolib/tetgen_wrap.h>

using namespace std;
using namespace cinolib;

struct closed_curve {
  vector<int> edges = {};
  vector<double> lerps = {};
};
struct circle_tids {
  double lerp = -1;
  int offset = -1;
};
typedef vector<closed_curve> Isoline;

struct Circle {
  int center;
  double radius;
  Isoline isoline;
  vector<double> distances = {};
};

Circle create_circle(const DrawableTrimesh<> &m, const int center,
                     const double &radius, const vector<double> &distances);

std::vector<vector<vec3d>> circle_positions(const DrawableTrimesh<> &m,
                                            const Circle &c0);

double normalize_circle_length(const DrawableTrimesh<> &m, const Circle &c);

#endif