#include "relu.hpp"

double derivative(double x) { return x > 0 ? 1 : 0; }

ReLU::ReLU(int height, int width, int depth) {
  this->height = height;
  this->width = width;
  this->depth = depth;
}
void ReLU::set_input(vector<MatrixXd> input) { this->input = input; }

void ReLU::clear_output() {
  output.clear();
  for (int i = 0; i < this->depth; i++) {
    output.push_back(MatrixXd::Zero(height, width));
  }
}

void ReLU::feed_forward(vector<MatrixXd> input) {
  this->set_input(input);
  this->clear_output();
  for (int d = 0; d < depth; d++)
    output[d] = input[d].cwiseMax(output[d]);
}

void ReLU::back_propagation(vector<MatrixXd> upstream_gradient) {
  gradients = input;
  for (int d = 0; d < depth; d++) {
    gradients[d] = gradients[d].unaryExpr(ptr_fun(derivative));
    gradients[d] =
        (gradients[d].array() * upstream_gradient[d].array()).matrix();
  }
}
