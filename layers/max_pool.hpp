#ifndef _MAX_POOL_HPP_
#define _MAX_POOL_HPP_
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using Eigen::MatrixXd;

class Max_Pool {
public:
  Max_Pool(int height, int width, int depth, int filter_size, int stride);
  int height;
  int width;
  int depth;
  int filter_size = 2; // Square
  int stride = 2;

  vector<MatrixXd> input;
  vector<MatrixXd> output;
  vector<MatrixXd> gradients;

  void set_input(vector<MatrixXd> input);
  void clear_output();
  void feed_forward(vector<MatrixXd> input);
  void back_propagation(vector<MatrixXd> upstream_gradient);
};

void test_max_pool_forward();

#endif