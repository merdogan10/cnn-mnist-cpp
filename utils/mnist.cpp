#include "mnist.hpp"

vector<MatrixXd> read_raw(string file_name) {
  vector<MatrixXd> raw_data;
  ifstream indata;
  indata.open(file_name);
  int num_rows = 0, num_cols = 0;
  string line;
  while (getline(indata, line)) {
    stringstream lineStream(line);
    string cell;
    MatrixXd tmp_row;
    if (num_cols) {
      tmp_row.resize(1, num_cols);
    }
    int col_index = 0;
    while (getline(lineStream, cell, ',')) {
      if (!num_rows) {
        num_cols++;
      } else {
        tmp_row(0, col_index) = atof(cell.c_str()); // string to double
        col_index++;
      }
    }
    if (num_rows) {
      raw_data.push_back(tmp_row);
    }
    num_rows++;
  }
  return raw_data;
}

MNIST::MNIST(string dir) {
  this->dir = dir;
  train_file = dir + "/train.csv";
  test_file = dir + "/test.csv";

  vector<MatrixXd> train_raw = read_raw(train_file);

  int num_examples = train_raw.size();
  vector<vector<MatrixXd>> train_data_all;
  vector<VectorXd> train_label_all;

  for (int i = 0; i < num_examples; i++) {
    int label = train_raw[i](0, 0);
    vector<MatrixXd> img;
    img.push_back(MatrixXd::Zero(28, 28));
    for (int r = 0; r < 28; r++)
      img[0].row(r) = train_raw[i].block(0, 28 * r + 1, 1, 28);
    img[0] /= 255.0;
    train_data_all.push_back(img);
    VectorXd label_vector = VectorXd::Zero(10);
    label_vector(label) += 1.0;
    train_label_all.push_back(label_vector);
  }

  train_data = vector<vector<MatrixXd>>(train_data_all.begin(),
                                        train_data_all.begin() +
                                            num_examples * split_ratio);
  train_labels =
      vector<VectorXd>(train_label_all.begin(),
                       train_label_all.begin() + num_examples * split_ratio);
  validation_data = vector<vector<MatrixXd>>(train_data_all.begin() +
                                                 num_examples * split_ratio,
                                             train_data_all.end());
  validation_labels =
      vector<VectorXd>(train_label_all.begin() + num_examples * split_ratio,
                       train_label_all.end());

  vector<MatrixXd> test_raw = read_raw(test_file);

  int num_test = test_raw.size();
  for (int i = 0; i < num_test; i++) {
    vector<MatrixXd> img;
    img.push_back(MatrixXd::Zero(28, 28));
    for (int r = 0; r < 28; r++)
      img[0].row(r) = test_raw[i].block(0, 28 * r, 1, 28);
    img[0] /= 255.0;
    test_data.push_back(img);
  }
}
