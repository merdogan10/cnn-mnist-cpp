#ifndef _MNIST_HPP_
#define _MNIST_HPP_
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class MNIST {
public:
  MNIST(string dir);
  string dir = "data";
  string train_file;
  string test_file;
  double split_ratio = 0.9;

  vector<vector<MatrixXd>> train_data;
  vector<vector<MatrixXd>> validation_data;
  vector<vector<MatrixXd>> test_data;

  vector<VectorXd> train_labels;
  vector<VectorXd> validation_labels;
};

#endif