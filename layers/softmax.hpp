#ifndef _SOFTMAX_HPP_
#define _SOFTMAX_HPP_
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class Softmax {
public:
  Softmax(int value){};

  VectorXd input;
  VectorXd output;
  VectorXd gradients;

  void set_input(VectorXd input);
  void feed_forward(VectorXd input);
  void back_propagation(VectorXd upstream_gradient);
};

#endif