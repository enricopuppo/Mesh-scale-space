MP_shape_op init_discrete_diff_op_monge_patch(
    const shape_data &data, const shape_geometry &geometry,
    const dual_geodesic_solver &solver, const bool grad, const bool lap,
    const int k) {
  time_function();
  typedef Eigen::Triplet<double> T;
  vector<T> L_entries;
  vector<T> G_entries;
  int V = (int)data.positions.size();
  MP_shape_op result;
  result.quadrics.resize(V);
  result.N.resize(V);
  int invertible = 0;

  for (int i = 0; i < V; ++i) {

    if (is_boundary(data.triangles, geometry.adjacencies, geometry.v2t, i))
      continue;

    auto [nbr, lens, tetas, n] =
        filtered_ring_stencil(data, geometry, solver, i, k);
    result.N[i] = vec3f{(float)n.x, (float)n.y, (float)n.z};
    vec3d vert =
        vec3d{data.positions[i].x, data.positions[i].y, data.positions[i].z};
    int s = (int)nbr.size();
    Eigen::MatrixXd Q(s, 5);
    Eigen::VectorXd h(s);
    auto pos = zero2d;
    auto d = 0.f;
    auto coords = zero3d;

    for (int j = 0; j < s; ++j) {
      pos =
          vec2d{lens[j]  yocto::cos(tetas[j]), lens[j]  yocto::sin(tetas[j])};
      coords = vec3d{data.positions[nbr[j]].x, data.positions[nbr[j]].y,
                     data.positions[nbr[j]].z} -
               vert;

      Q(j, 0) = pos[0];
      Q(j, 1) = pos[1];
      Q(j, 2) = pow(pos[0], 2) / 2;
      Q(j, 3) = pos[0] * pos[1];
      Q(j, 4) = pow(pos[1], 2) / 2;

      h(j) = dot(coords, n);
    }

    Eigen::MatrixXd Qt = Eigen::Transpose<Eigen::MatrixXd>(Q);

    Eigen::MatrixXd A = Qt * Q;
    Eigen::MatrixXd E = rhs(s);
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> dec(A);
    Eigen::VectorXd c(5);
    Eigen::MatrixXd a(5, s + 1);

    if (dec.isInvertible()) {
      Eigen::MatrixXd inv = A.inverse();
      c = inv  Qt  h;
      a = inv  Qt  E;
      ++invertible;

    } else {
      Eigen::MatrixXd Rhsc = Qt * h;
      Eigen::MatrixXd Rhsa = Qt * E;
      c = dec.solve(Rhsc);
      a = dec.solve(Rhsa);
    }
    // c = Q.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(h);
    // a = Q.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(E);
    result.quadrics[i] = c;
    Eigen::Matrix2d g;
    Eigen::Matrix2d g_inv;

    double c0_squared = pow(c[0], 2);
    double c1_squared = pow(c[1], 2);
    g << 1 + c0_squared, c[0]  c[1], c[0]  c[1], 1 + c1_squared;

    double det = 1 + c0_squared + c1_squared;
    g_inv << 1 + c1_squared, -c[0]  c[1], -c[0]  c[1], 1 + c0_squared;
    g_inv /= det;
    nbr.insert(nbr.begin(), i);
    laplacian_entries(L_entries, g, g_inv, nbr, c, a);
    fill_riemannian_gradient_entries(G_entries, nbr, c, a.row(0), a.row(1), V);
  }
  if (grad) {
    result.Grad.resize(2 * V, V);
    result.Grad.setFromTriplets(G_entries.begin(), G_entries.end());
  }

  if (lap) {
    result.Lap.resize(V, V);
    result.Lap.setFromTriplets(L_entries.begin(), L_entries.end());
  }

  std::cout << invertible / V * 100 << std::endl;
  return result;
}

std::tuple<vector<int>, vector<double>, vector<double>, vec3d>
filtered_ring_stencil(const shape_data &data, const shape_geometry &geometry,
                      const dual_geodesic_solver &solver, const int vid,
                      const int k = 1) {

  auto nbr = k_ring(data.triangles, geometry.v2t, vid, k);
  auto N = average_normal(data.normals, nbr, vid);
  auto pos = data.positions[vid];
  // auto it = remove_if(nbr.begin(), nbr.end(), [&](const int curr) {
  //   return (dot(data.normals[curr], N) < 0);
  // });
  // nbr.erase(it, nbr.end());
  auto count = 1;
  while (nbr.size() < 5) {
    nbr = k_ring(data.triangles, geometry.v2t, vid, k + count);
    // it = remove_if(nbr.begin(), nbr.end(), [&](const int curr) {
    //   return (dot(data.normals[curr], N) < 0);
    // });
    // nbr.erase(it, nbr.end());
    ++count;
  }
  auto lens = vector<double>{};
  auto tetas = vector<double>{};

  lens.resize(nbr.size());
  tetas.resize(nbr.size());

  auto e = polar_basis(data.triangles, data.positions, geometry.v2t, N, vid);
  for (auto i = 0; i < nbr.size(); ++i) {
    auto v = data.positions[nbr[i]] - pos;
    auto proj = project_vec(v, N);
    lens[i] = length(v);
    auto theta = angle(proj, e);
    if (dot(cross(e, proj), N) < 0)
      theta = 2 * pif - theta;

    tetas[i] = theta;
  }
  return {nbr, lens, tetas, vec3d{N.x, N.y, N.z}};
}

Eigen::MatrixXd rhs(int s) {
  Eigen::MatrixXd E(s, s + 1);
  Eigen::MatrixXd X = Eigen::MatrixXd::Constant(s, 1, -1);
  E.topLeftCorner(s, 1) = X;
  Eigen::MatrixXd I(s, s);
  I.setIdentity();
  E.topRightCorner(s, s) = I;
  return E;
}