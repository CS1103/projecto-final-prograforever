#include <iostream>
#include "neural_network.h"
#include "nn_dense.h"
#include "tensor.h"

struct serializer {
    void save(const utec::neural_network::NeuralNetwork<float>& network, const std::string& path);
    void load(utec::neural_network::NeuralNetwork<float>& network, const std::string& path);
};
