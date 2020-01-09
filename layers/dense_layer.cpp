#include "dense_layer.hpp"

Dense_Layer::Dense_Layer(int height, int width, int depth, int num_outputs) {
  this->height = height;
  this->width = width;
  this->depth = depth;
  this->num_outputs = num_outputs;

  // TODO: Use Xavier
  weights = MatrixXd::Random(num_outputs, height * width * depth);

  biases = VectorXd::Zero(num_outputs);

  gradients.resize(depth);
  accumulated_gradients.resize(depth);
  for (int d = 0; d < depth; d++)
    accumulated_gradients[d] = MatrixXd::Zero(height, width);
  accumulated_gradient_weights =
      MatrixXd::Zero(num_outputs, height * width * depth);
  accumulated_gradient_biases = VectorXd::Zero(num_outputs);
}
void Dense_Layer::set_input(vector<MatrixXd> input) { this->input = input; }

void Dense_Layer::feed_forward(vector<MatrixXd> input) {
  this->set_input(input);
  VectorXd flat_input(height * width * depth);
  for (int d = 0; d < depth; d++) {
    // Matrix to vector
    Map<RowVectorXd> flat(input[d].data(), input[d].size());
    flat_input.segment(d * flat.size(), flat.size()) = flat;
  }
  output = (weights * flat_input) + biases;
}

void Dense_Layer::back_propagation(VectorXd upstream_gradient) {
  VectorXd gradients_vec = VectorXd::Zero(height * width * depth);
  for (int i = 0; i < height * width * depth; i++)
    gradients_vec[i] = weights.col(i).dot(upstream_gradient);

  for (int d = 0; d < depth; d++) {
    gradients[d] = MatrixXd::Zero(height, width);
  }
  for (int d = 0; d < depth; d++) {
    // Vector to matrix
    gradients.push_back(MatrixXd::Zero(height, width));
    Map<MatrixXd> mat(
        gradients_vec.segment(d * height * width, height * width).data(),
        height, width);
    gradients[d] = mat;
    accumulated_gradients[d] += gradients[d];
  }

  gradient_weights = MatrixXd::Zero(weights.rows(), weights.cols());
  VectorXd flat_input(height * width * depth);
  for (int d = 0; d < depth; d++) {
    // Matrix to vector
    Map<RowVectorXd> flat(input[d].data(), input[d].size());
    flat_input.segment(d * flat.size(), flat.size()) = flat;
  }
  for (int r = 0; r < gradient_weights.rows(); r++) {
    gradient_weights.row(r) = flat_input.transpose() * upstream_gradient(r);
  }

  accumulated_gradient_weights += gradient_weights;
  gradient_biases = upstream_gradient;
  accumulated_gradient_biases += gradient_biases;
}

void Dense_Layer::update_weights(int batch_size, double learning_rate) {
  weights =
      weights - learning_rate * (accumulated_gradient_weights / batch_size);
  biases = biases - learning_rate * (accumulated_gradient_biases / batch_size);

  accumulated_gradients.resize(depth);
  for (int d = 0; d < depth; d++)
    accumulated_gradients.push_back(MatrixXd::Zero(height, width));
  accumulated_gradient_weights =
      MatrixXd::Zero(num_outputs, height * width * depth);
  accumulated_gradient_biases = VectorXd::Zero(num_outputs);
}

void Dense_Layer::save_weights(string weights_file) {
  ofstream fout(weights_file);
  fout << this->weights.rows() << " " << this->weights.cols() << endl;
  fout << this->weights << endl;
  fout << this->biases.size() << endl;
  fout << this->biases << endl;
  fout.close();
}

void Dense_Layer::load_weights(string weights_file) {
  ifstream fin(weights_file);
  int num_row, num_col, num_size;
  fin >> num_row >> num_col;
  MatrixXd temp_weights(num_row, num_col);
  for (int i = 0; i < num_row; i++) {
    for (int j = 0; j < num_col; j++) {
      fin >> temp_weights(i, j);
    }
  }
  this->weights = temp_weights;
  fin >> num_size;
  VectorXd temp_biases(num_size);
  for (int i = 0; i < num_size; i++) {
    fin >> temp_biases(i);
  }
  this->biases = temp_biases;
  fin.close();
}