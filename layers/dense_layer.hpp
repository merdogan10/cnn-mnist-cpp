#ifndef _DENSE_LAYER_HPP_
#define _DENSE_LAYER_HPP_
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;

class Dense_Layer {
public:
  Dense_Layer(int height, int width, int depth, int num_outputs);
  int height;
  int width;
  int depth;
  int num_outputs;

  VectorXd output;
  vector<MatrixXd> input;
  MatrixXd weights;
  VectorXd biases;

  vector<MatrixXd> gradients;
  MatrixXd gradient_weights;
  VectorXd gradient_biases;

  vector<MatrixXd> accumulated_gradients;
  MatrixXd accumulated_gradient_weights;
  VectorXd accumulated_gradient_biases;

  void set_input(vector<MatrixXd> input);
  void clear_output();
  void feed_forward(vector<MatrixXd> input);
  void back_propagation(VectorXd upstream_gradient);
  void update_weights(int batch_size, double learning_rate);

  void save_weights(string dir);
  void load_weights(string dir);
};

#endif