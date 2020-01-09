cmake_minimum_required(VERSION 3.9)
project(test VERSION 0.1
        DESCRIPTION "Neural Network"
        LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-O3")

include_directories(./libs/include)
include_directories(./layers)
include_directories(./utils)

set(TEST_TARGET testmain)

add_executable(${TEST_TARGET}
                main.cpp
                layers/conv_layer.cpp
                layers/max_pool.cpp
                layers/relu.cpp
                layers/dense_layer.cpp
                layers/softmax.cpp
                layers/cross_entropy.cpp
                utils/mnist.cpp
                test/test_max_pool.cpp
                test/test_conv_layer.cpp
                test/test_relu.cpp)
target_compile_features(${TEST_TARGET} PUBLIC cxx_std_17)

