#include "model_serializer.h"
#include "nn_dense.h"
#include "tensor.h"
#include <fstream>
#include <stdexcept>
#include <vector>

void serializer::save(
  const utec::neural_network::NeuralNetwork<float>& network, const std::string& path) {
    std::ofstream output_file(path);
    if (!output_file.is_open())
        throw std::runtime_error("Error opening model file");

    int dense_count = 0;
    for (auto& layer_ptr: network.get_layers()) {
        if (dynamic_cast<utec::neural_network::Dense<float>*>(layer_ptr.get()))
            ++dense_count;
    }
    output_file << dense_count << "\n\n";

    for (auto& layer_ptr: network.get_layers()) {
        //Prueba si el puntero apunta a un Dense<float*>
        if (auto d = dynamic_cast<utec::neural_network::Dense<float>*>(layer_ptr.get())) {
            const auto& W = d->get_weights();
            const auto& B = d->get_bias();
            output_file << W << "\n" << B << "\n\n";
        }
    }
}

void serializer::load(utec::neural_network::NeuralNetwork<float>& network, const std::string& path) {
    std::ifstream input_file(path);
    if (!input_file.is_open()) throw std::runtime_error("Error opening data file");
    int dense_count; input_file >> dense_count;
    for (int layer_index = 0; layer_index < dense_count; ++layer_index) {
        //Se quita la llave de inicio del cout del tensor de 2 dimensiones
        char temp_char; input_file >> temp_char;
        std::vector<float> weight_data;
        input_file >> std::ws; //recorre el espacio en blanco del archivo
        while (input_file.peek() != '}') {
            float v; input_file >> v;
            weight_data.push_back(v);
            input_file >> std::ws;
        }
        input_file >> temp_char; //se quita la llave del final
        input_file >> temp_char; //se quita la llave del principio
        std::vector<float> bias_data;
        input_file >> std::ws;
        while (input_file.peek() != '}') {
            float v; input_file >> v;
            bias_data.push_back(v);
            input_file >> std::ws;
        }
        input_file >> temp_char; //se quita la llave del final
        size_t cols = bias_data.size();
        size_t rows = weight_data.size() / cols;
        utec::algebra::Tensor<float,2> W(rows, cols);
        utec::algebra::Tensor<float,2> B(1, cols);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                W(i,j) = weight_data[i*cols + j];
        for (size_t j = 0; j < cols; ++j)
            B(0,j) = bias_data[j];
        network.add_layer(std::make_unique<utec::neural_network::Dense<float>>(W, B));
        if (layer_index+1 < dense_count)
            network.add_layer(std::make_unique<utec::neural_network::ReLU<float>>());
    }
    network.add_layer(std::make_unique<utec::neural_network::Sigmoid<float>>());
}