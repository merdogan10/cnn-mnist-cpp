#include "cross_entropy.hpp"

void Cross_Entropy::feed_forward(VectorXd predicted, VectorXd actual) {
  this->predicted = predicted;
  this->actual = actual;
  this->loss = -actual.dot(predicted.array().log().matrix());
}

void Cross_Entropy::back_propagation() {
  gradients = -(actual.array() * (1 / predicted.array())).matrix();
}