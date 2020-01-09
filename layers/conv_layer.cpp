#include "conv_layer.hpp"

Conv_Layer::Conv_Layer(int height, int width, int depth, int filter_size,
                       int stride, int num_filters) {
  this->height = height;
  this->width = width;
  this->depth = depth;
  this->filter_size = filter_size;
  this->stride = stride;
  this->num_filters = num_filters;

  gradients.resize(this->depth);
  filters.resize(this->num_filters);

  for (int i = 0; i < this->num_filters; i++) {
    for (int j = 0; j < this->depth; j++) {
      filters[i].push_back(MatrixXd::Random(filter_size, filter_size));
    }
  }

  accumulated_gradient_filters.resize(num_filters);
  for (int f = 0; f < num_filters; f++)
    for (int d = 0; d < depth; d++)
      accumulated_gradient_filters[f].push_back(
          MatrixXd::Zero(filter_size, filter_size));
}

void Conv_Layer::set_input(vector<MatrixXd> input) { this->input = input; }

void Conv_Layer::clear_output() {
  output.clear();
  for (int i = 0; i < this->num_filters; i++) {
    output.push_back(MatrixXd::Zero((height - filter_size) / stride + 1,
                                    (width - filter_size) / stride + 1));
  }
}

void Conv_Layer::feed_forward(vector<MatrixXd> input) {
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

  for (int f = 0; f < num_filters; f++) {
    for (int i = 0, r = 0; i < height - filter_size + 1; i += stride, r++) {
      for (int j = 0, c = 0; j < width - filter_size + 1; j += stride, c++) {
        for (int d = 0; d < depth; d++) {
          MatrixXd sub_matrix = input[d].block(i, j, filter_size, filter_size);
          output[f](r, c) += filters[f][d].cwiseProduct(sub_matrix).sum();
        }
        // Add bias here
      }
    }
  }
}

void Conv_Layer::back_propagation(vector<MatrixXd> upstream_gradient) {
  for (int d = 0; d < depth; d++) {
    gradients[d] = MatrixXd::Zero(height, width);
  }

  for (int f = 0; f < num_filters; f++) {
    for (int r = 0; r < output[f].rows(); r++) {
      for (int c = 0; c < output[f].cols(); c++) {
        for (int d = 0; d < depth; d++) {
          MatrixXd tmp = MatrixXd::Zero(height, width);
          tmp.block(r * stride, c * stride, filter_size, filter_size) =
              filters[f][d];
          gradients[d] += upstream_gradient[f](r, c) * tmp;
        }
      }
    }
  }

  gradient_filters.clear();
  gradient_filters.resize(num_filters);
  for (int i = 0; i < num_filters; i++) {
    gradient_filters[i].resize(depth);
    for (int j = 0; j < depth; j++) {
      gradient_filters[i][j] = MatrixXd::Zero(filter_size, filter_size);
    }
  }

  for (int f = 0; f < num_filters; f++) {
    for (int r = 0; r < output[f].rows(); r++) {
      for (int c = 0; c < output[f].cols(); c++) {
        for (int d = 0; d < depth; d++) {
          MatrixXd tmp = MatrixXd::Zero(filter_size, filter_size);
          tmp =
              input[d].block(r * stride, c * stride, filter_size, filter_size);
          gradient_filters[f][d] += upstream_gradient[f](r, c) * tmp;
        }
      }
    }
  }

  for (int f = 0; f < num_filters; f++)
    for (int d = 0; d < depth; d++)
      accumulated_gradient_filters[f][d] += gradient_filters[f][d];
}

void Conv_Layer::update_weights(int batch_size, double learning_rate) {
  for (int f = 0; f < num_filters; f++)
    for (int d = 0; d < depth; d++)
      filters[f][d] -=
          learning_rate * (accumulated_gradient_filters[f][d] / batch_size);

  accumulated_gradient_filters.clear();
  accumulated_gradient_filters.resize(num_filters);
  for (int f = 0; f < num_filters; f++)
    for (int d = 0; d < depth; d++)
      accumulated_gradient_filters[f].push_back(
          MatrixXd::Zero(filter_size, filter_size));
}

void Conv_Layer::save_filters(string filters_file) {
  ofstream fout(filters_file);
  fout << this->filters.size() << endl;
  for (int f = 0; f < this->filters.size(); f++) {
    fout << this->filters[f].size() << endl;
    for (int d = 0; d < this->filters[f].size(); d++) {
      fout << this->filters[f][d].rows() << " " << this->filters[f][d].cols()
           << endl;
      fout << this->filters[f][d] << endl;
    }
  }
  fout.close();
}

void Conv_Layer::load_filters(string filters_file) {
  ifstream fin(filters_file);
  int num_filt, num_depth, num_row, num_col;
  fin >> num_filt;
  this->filters.clear();
  this->filters.resize(num_filt);
  for (int f = 0; f < num_filt; f++) {
    fin >> num_depth;
    this->filters[f].resize(num_depth);
    for (int d = 0; d < num_depth; d++) {
      fin >> num_row >> num_col;
      MatrixXd temp_filter(num_row, num_col);
      for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
          fin >> temp_filter(i, j);
        }
      }
      this->filters[f][d] = temp_filter;
    }
  }
  fin.close();
}