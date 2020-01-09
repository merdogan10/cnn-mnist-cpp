#include "max_pool.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

using Eigen::MatrixXd;
using namespace std;

void test_max_pool_forward() {
  MatrixXd m(4, 4), o(2, 2);
  m << 1, 1, 2, 4, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4;
  o << 6, 8, 3, 4;
  vector<MatrixXd> input;
  input.push_back(m);
  Max_Pool *mp = new Max_Pool(4, 4, 1, 2, 2);
  mp->feed_forward(input);
  if (!mp->output[0].isApprox(o)) {
    cout << "Test failed: test_max_pool_forward" << endl;
  }
}