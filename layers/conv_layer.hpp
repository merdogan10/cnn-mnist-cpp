#ifndef _CONV_LAYER_HPP_
#define _CONV_LAYER_HPP_
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using Eigen::MatrixXd;

class Conv_Layer {
public:
  Conv_Layer(int height, int width, int depth, int filter_size, int stride,
             int num_filters);
  int height;
  int width;
  int depth;
  int filter_size = 3; // Square
  int stride = 1;
  int num_filters = 1;

  vector<MatrixXd> input;
  vector<MatrixXd> output;
  vector<MatrixXd> gradients;
  vector<vector<MatrixXd>> filters;
  vector<vector<MatrixXd>> gradient_filters;
  vector<vector<MatrixXd>> accumulated_gradient_filters;

  void set_input(vector<MatrixXd> input);
  void clear_output();
  void feed_forward(vector<MatrixXd> input);
  void back_propagation(vector<MatrixXd> upstream_gradient);
  void update_weights(int batch_size, double learning_rate);

  void save_filters(string dir);
  void load_filters(string dir);
};

void test_conv_layer_forward();

#endif