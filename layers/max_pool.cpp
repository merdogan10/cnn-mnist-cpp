#include "max_pool.hpp"

Max_Pool::Max_Pool(int height, int width, int depth, int filter_size,
                   int stride) {
  this->height = height;
  this->width = width;
  this->depth = depth;
  this->filter_size = filter_size;
  this->stride = stride;

  gradients.resize(this->depth);
}

void Max_Pool::set_input(vector<MatrixXd> input) { this->input = input; }

void Max_Pool::clear_output() {
  output.clear();
  for (int i = 0; i < this->depth; i++) {
    output.push_back(MatrixXd::Zero((height - filter_size) / stride + 1,
                                    (width - filter_size) / stride + 1));
  }
}

void Max_Pool::feed_forward(vector<MatrixXd> input) {
  if ((height - filter_size) % stride != 0) {
    cout << "Filter dimension and stride is not valid height" << endl;
    return;
  }
  if ((width - filter_size) % stride != 0) {
    cout << "Filter dimension and stride is not valid for width" << endl;
    return;
  }

  this->set_input(input);
  this->clear_output();

  for (int d = 0; d < depth; d++) {
    for (int i = 0, r = 0; i < height - filter_size + 1; i += stride, r++) {
      for (int j = 0, c = 0; j < width - filter_size + 1; j += stride, c++) {
        MatrixXd sub_matrix = input[d].block(i, j, filter_size, filter_size);
        output[d](r, c) = sub_matrix.maxCoeff();
      }
    }
  }
}

void Max_Pool::back_propagation(vector<MatrixXd> upstream_gradient) {
  for (int d = 0; d < depth; d++) {
    gradients[d] = MatrixXd::Zero(height, width);
    for (int i = 0; i + filter_size <= height; i += stride) {
      for (int j = 0; j + filter_size <= width; j += stride) {
        MatrixXd tmp = MatrixXd::Zero(filter_size, filter_size);
        ptrdiff_t r, c;
        input[d].block(i, j, filter_size, filter_size).maxCoeff(&r, &c);
        tmp(r, c) = upstream_gradient[d](i / stride, j / stride);
        gradients[d].block(i, j, filter_size, filter_size) += tmp;
      }
    }
  }
}