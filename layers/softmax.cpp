#include "softmax.hpp"
void Softmax::set_input(VectorXd input) { this->input = input; }

void Softmax::feed_forward(VectorXd input) {
  this->set_input(input);
  double sum_exp = (input.array() - input.maxCoeff()).exp().sum();
  output = (input.array() - input.maxCoeff()).exp() / sum_exp;
}
void Softmax::back_propagation(VectorXd upstream_gradient) {
  double sub = upstream_gradient.dot(output);
  gradients =
      ((upstream_gradient.array() - sub).array() * output.array()).matrix();
}
