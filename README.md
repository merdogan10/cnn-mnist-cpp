# cnn-mnist-cpp
mnist with deep learning from scratch with C++

# How to run?

Run those commands in the same order.

```
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ./testmain
```

If you don't want to train data all over again and want to use pretrained weights, press `y` and then `enter` when the program asks after the start. Otherwise, press another character `not y` and then `enter` to standard long hours training.

Normal training takes ~32 mins.
By using pretrained weights, running train set and validation set takes ~40 secs.
By using pretrained weights, running only validation set takes ~7 secs.

So I decided to use train set and validation set with pretrained weights for demo purpose.
(I printed the index on every 1000 example to keep track.)

# Data

I get the MNIST data from [kaggle](https://www.kaggle.com/c/digit-recognizer/data) as csv files. I read the data from `data/train.csv` and `data/test.csv`. I splitted the given train data to 90% of it as train data and remaining 10% of it as validation data. I get the output of the test data and send it to kaggle competition. Pretrained weights are also kept under `data` folder as `data/conv1.out`, `data/conv2.out`, `data/dense.out`.

# Result

I ran the data 1 epoch and it took 3 hours to get output. When I increase epoch accuracy increases but it takes almost forever. (Ex: 8 hours for 2 epoch...)
```
  Training accuracy: 0.970026
  Validation accuracy: 0.96881
  Test accuracy: 0.96700
```
![result](/kaggle.png)

# Layers

All layers are implemented in different classes. Sizes of outputs and inputs must match. Here are the classes:

- Convolution Layer
- ReLU Layer
- Max Pool Layer
- Dense Layer
- Softmax Layer
- Cross Entropy Layer

Xor network is deprecated and dense layer is implemented all over again.
