#ifndef _RELU_HPP_
#define _RELU_HPP_
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using Eigen::MatrixXd;

class ReLU {
public:
  ReLU(int height, int width, int depth);
  int height;
  int width;
  int depth;

  vector<MatrixXd> input;
  vector<MatrixXd> output;
  vector<MatrixXd> gradients;

  void set_input(vector<MatrixXd> input);
  void clear_output();
  void feed_forward(vector<MatrixXd> input);
  void back_propagation(vector<MatrixXd> upstream_gradient);
};

void test_relu_forward();

#endif