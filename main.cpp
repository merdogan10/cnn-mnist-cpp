#include "conv_layer.hpp"
#include "cross_entropy.hpp"
#include "dense_layer.hpp"
#include "max_pool.hpp"
#include "mnist.hpp"
#include "relu.hpp"
#include "softmax.hpp"
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>

using Eigen::MatrixXd;
using namespace std;
int main() {
  /*
  Training accuracy: 0.970026
  Validation accuracy: 0.96881
  */
  test_max_pool_forward();
  test_conv_layer_forward();
  test_relu_forward();

  MNIST *mn = new MNIST("../data");

  vector<vector<MatrixXd>> train_data = mn->train_data;
  vector<vector<MatrixXd>> validation_data = mn->validation_data;
  vector<vector<MatrixXd>> test_data = mn->test_data;

  vector<VectorXd> train_labels = mn->train_labels;
  vector<VectorXd> validation_labels = mn->validation_labels;

  cout << "Data loaded." << endl;

  int TRAIN_DATA_SIZE = train_data.size();
  int VALIDATION_DATA_SIZE = validation_data.size();
  int TEST_DATA_SIZE = test_data.size();

  int EPOCHS = 1; // 5;
  int BATCH_SIZE = 10;
  int BATCHES = TRAIN_DATA_SIZE / BATCH_SIZE;

  double LEARNING_RATE = 0.05;

  // input: 28x28x1 filter: 5x5x1 stride: 1 output: 24x24x6
  Conv_Layer conv1(28, 28, 1, 5, 1, 6);
  // output: 24x24x6
  ReLU relu1(24, 24, 6);
  // input: 24x24x6 filter: 2x2x6 stride: 2 output: 12x12x6
  Max_Pool pool1(24, 24, 6, 2, 2);
  // input: 12x12x6 filter: 5x5x6 stride: 1 output: 8x8x16
  Conv_Layer conv2(12, 12, 6, 5, 1, 16);
  // output: 8x8x16
  ReLU relu2(8, 8, 16);
  // input: 8x8x16 filter: 2x2x16 stride: 2 output: 4x4x16
  Max_Pool pool2(8, 8, 16, 2, 2);
  // input: 4x4x16 output: 1x10
  Dense_Layer dense(4, 4, 16, 10);
  Softmax soft(0);
  Cross_Entropy entropy(0);

  double cumulative_loss = 0.0;

  cout << "TRAIN DATA SIZE: " << TRAIN_DATA_SIZE << endl;
  cout << "VALIDATION DATA SIZE: " << VALIDATION_DATA_SIZE << endl;
  cout << "TEST DATA SIZE: " << TEST_DATA_SIZE << endl;
  cout << "EPOCHS: " << EPOCHS << endl;
  cout << "BATCHES: " << BATCHES << endl;

  char selection;
  cout << "Press \'y\' to use pretrained weights, other chars to train from "
          "scratch..."
       << endl;
  cin >> selection;

  if (selection == 'y' || selection == 'Y') {
    conv1.load_filters("../data/conv1.out");
    conv2.load_filters("../data/conv2.out");
    dense.load_weights("../data/dense.out");
    double true_positive = 0.0;
    // Training accuracy
    for (int i = 0; i < TRAIN_DATA_SIZE; i++) {
      if (i % 1000 == 0) {
        cout << "Training: " << i << endl;
      }
      // Forward pass
      conv1.feed_forward(train_data[i]);
      relu1.feed_forward(conv1.output);
      pool1.feed_forward(relu1.output);
      conv2.feed_forward(pool1.output);
      relu2.feed_forward(conv2.output);
      pool2.feed_forward(relu2.output);
      dense.feed_forward(pool2.output);
      dense.output /= 100;
      soft.feed_forward(dense.output);

      ptrdiff_t actual_index, pred_index;
      train_labels[i].maxCoeff(&actual_index);
      soft.output.maxCoeff(&pred_index);
      if (actual_index == pred_index)
        true_positive += 1.0;
    }
    cout << "true_positive: " << true_positive << endl;
    cout << "Training accuracy: " << true_positive / TRAIN_DATA_SIZE << endl;
    // Validation accuracy
    cumulative_loss = 0.0;
    true_positive = 0.0;
    for (int i = 0; i < VALIDATION_DATA_SIZE; i++) {
      if (i % 1000 == 0) {
        cout << "Validating: " << i << endl;
      }
      // Forward pass
      conv1.feed_forward(validation_data[i]);
      relu1.feed_forward(conv1.output);
      pool1.feed_forward(relu1.output);
      conv2.feed_forward(pool1.output);
      relu2.feed_forward(conv2.output);
      pool2.feed_forward(relu2.output);
      dense.feed_forward(pool2.output);
      dense.output /= 100;
      soft.feed_forward(dense.output);
      entropy.feed_forward(soft.output, validation_labels[i]);
      cumulative_loss += entropy.loss;

      ptrdiff_t actual_index, pred_index;
      validation_labels[i].maxCoeff(&actual_index);
      soft.output.maxCoeff(&pred_index);
      if (actual_index == pred_index)
        true_positive += 1.0;
    }
    cout << "true_positive: " << true_positive << endl;
    cout << "Validation set accuracy: " << true_positive / VALIDATION_DATA_SIZE
         << endl;
    return 0;
  }

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    for (int b = 0; b < BATCHES; b++) {
      if (b % 100 == 0) {
        cout << "epoch: " << epoch << " batch: " << b << endl;
      }
      // Select uniform random indices
      VectorXd batch = VectorXd::Random(BATCH_SIZE) / 2;
      batch = (batch.array() + 0.5).matrix();
      batch *= (TRAIN_DATA_SIZE - 1);

      for (int i = 0; i < BATCH_SIZE; i++) {
        // Forward pass
        conv1.feed_forward(train_data[batch[i]]);
        relu1.feed_forward(conv1.output);
        pool1.feed_forward(relu1.output);
        conv2.feed_forward(pool1.output);
        relu2.feed_forward(conv2.output);
        pool2.feed_forward(relu2.output);
        dense.feed_forward(pool2.output);
        dense.output /= 100;
        soft.feed_forward(dense.output);
        entropy.feed_forward(soft.output, train_labels[batch[i]]);
        cumulative_loss += entropy.loss;

        // Backward pass
        entropy.back_propagation();
        soft.back_propagation(entropy.gradients);
        dense.back_propagation(soft.gradients);
        pool2.back_propagation(dense.gradients);
        relu2.back_propagation(pool2.gradients);
        conv2.back_propagation(relu2.gradients);
        pool1.back_propagation(conv2.gradients);
        relu1.back_propagation(pool1.gradients);
        conv1.back_propagation(relu1.gradients);
      }
      // Update params
      dense.update_weights(BATCH_SIZE, LEARNING_RATE);
      conv1.update_weights(BATCH_SIZE, LEARNING_RATE);
      conv2.update_weights(BATCH_SIZE, LEARNING_RATE);
    }
    // Training accuracy
    double true_positive = 0.0;
    for (int i = 0; i < TRAIN_DATA_SIZE; i++) {
      if (i % 1000 == 0) {
        cout << "Training: " << i << endl;
      }
      // Forward pass
      conv1.feed_forward(train_data[i]);
      relu1.feed_forward(conv1.output);
      pool1.feed_forward(relu1.output);
      conv2.feed_forward(pool1.output);
      relu2.feed_forward(conv2.output);
      pool2.feed_forward(relu2.output);
      dense.feed_forward(pool2.output);
      dense.output /= 100;
      soft.feed_forward(dense.output);

      ptrdiff_t actual_index, pred_index;
      train_labels[i].maxCoeff(&actual_index);
      soft.output.maxCoeff(&pred_index);
      if (actual_index == pred_index)
        true_positive += 1.0;
    }
    cout << "true_positive: " << true_positive << endl;
    cout << "Training accuracy: " << true_positive / TRAIN_DATA_SIZE << endl;

    // Validation accuracy
    cumulative_loss = 0.0;
    true_positive = 0.0;
    for (int i = 0; i < VALIDATION_DATA_SIZE; i++) {
      if (i % 1000 == 0) {
        cout << "Validating: " << i << endl;
      }
      // Forward pass
      conv1.feed_forward(validation_data[i]);
      relu1.feed_forward(conv1.output);
      pool1.feed_forward(relu1.output);
      conv2.feed_forward(pool1.output);
      relu2.feed_forward(conv2.output);
      pool2.feed_forward(relu2.output);
      dense.feed_forward(pool2.output);
      dense.output /= 100;
      soft.feed_forward(dense.output);
      entropy.feed_forward(soft.output, validation_labels[i]);
      cumulative_loss += entropy.loss;

      ptrdiff_t actual_index, pred_index;
      validation_labels[i].maxCoeff(&actual_index);
      soft.output.maxCoeff(&pred_index);
      if (actual_index == pred_index)
        true_positive += 1.0;
    }
    cout << "true_positive: " << true_positive << endl;
    cout << "Validation accuracy: " << true_positive / VALIDATION_DATA_SIZE
         << endl;
  }

  ofstream fout("results.csv");
  fout << "ImageId,Label" << endl;
  for (int i = 0; i < TEST_DATA_SIZE; i++) {
    if (i % 1000 == 0) {
      cout << "Testing: " << i << endl;
    }
    // Forward pass
    conv1.feed_forward(test_data[i]);
    relu1.feed_forward(conv1.output);
    pool1.feed_forward(relu1.output);
    conv2.feed_forward(pool1.output);
    relu2.feed_forward(conv2.output);
    pool2.feed_forward(relu2.output);
    dense.feed_forward(pool2.output);
    dense.output /= 100;
    soft.feed_forward(dense.output);

    ptrdiff_t pred_index;
    soft.output.maxCoeff(&pred_index);
    fout << to_string(i + 1) << "," << to_string(pred_index) << endl;
  }
  fout.close();

  conv1.save_filters("../data/conv1.out");
  conv2.save_filters("../data/conv2.out");
  dense.save_weights("../data/dense.out");
}