#ifndef _CROSS_ENTROPY_HPP_
#define _CROSS_ENTROPY_HPP_
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class Cross_Entropy {
public:
  Cross_Entropy(int value){};
  double loss;

  VectorXd predicted;
  VectorXd actual;
  VectorXd gradients;

  void feed_forward(VectorXd predicted, VectorXd actual);
  void back_propagation();
};

#endif