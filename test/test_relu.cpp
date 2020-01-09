#include "relu.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

using Eigen::MatrixXd;
using namespace std;

void test_relu_forward() {
  MatrixXd m0(3, 3), m1(3, 3), m2(3, 3), o0(3, 3), o1(3, 3), o2(3, 3);
  m0 << 0.780465, -0.959954, -0.52344, -0.302214, -0.0845965, 0.941268,
      -0.871657, -0.873808, 0.804416;
  o0 << 0.780465, 0, 0, 0, 0, 0.941268, 0, 0, 0.804416;
  m1 << 0.70184, -0.249586, 0.335448, -0.466669, 0.520497, 0.0632129, 0.0795207,
      0.0250707, -0.921439;

  o1 << 0.70184, 0, 0.335448, 0, 0.520497, 0.0632129, 0.0795207, 0.0250707, 0;

  m2 << -0.124725, 0.441905, 0.279958, 0.86367, -0.431413, -0.291903, 0.86162,
      0.477069, 0.375723;

  o2 << 0, 0.441905, 0.279958, 0.86367, 0, 0, 0.86162, 0.477069, 0.375723;

  vector<MatrixXd> input;
  input.push_back(m0);
  input.push_back(m1);
  input.push_back(m2);
  ReLU *rl = new ReLU(3, 3, 3);
  rl->feed_forward(input);
  if (!rl->output[0].isApprox(o0) || !rl->output[1].isApprox(o1) ||
      !rl->output[2].isApprox(o2)) {
    cout << "Test failed: test_relu_forward" << endl;
  }
}