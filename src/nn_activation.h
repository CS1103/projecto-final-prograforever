//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#include "tensor.h"
#include <cmath>
#include "nn_interfaces.h"
namespace utec::neural_network {
    template<typename T>
      class ReLU final : public ILayer<T> {
        utec::algebra::Tensor<T, 2> mask;
    public:
        ReLU(): mask(0, 0) {}
        utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& z) override {
            mask = utec::algebra::Tensor<T, 2>(z.shape());
            auto mask_it = mask.begin();
            for (auto it = z.cbegin(); it != z.cend(); ++it) {*mask_it = static_cast<T>(*it > 0); ++mask_it;}
            return z * mask;
        }
        utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& g) override {
            return g * mask;
        }
    };

    template<typename T>
     class Sigmoid final : public ILayer<T> {
        utec::algebra::Tensor<T, 2> sigma;
    public:
        Sigmoid(): sigma(0, 0) {}
        utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& z) override {
            sigma = utec::algebra::Tensor<T, 2>(z.shape());
            auto sigma_it = sigma.begin();
            for (auto it = z.cbegin(); it != z.cend(); ++it) {*sigma_it = 1/(1+std::exp(-(*it))); ++sigma_it;}
            return sigma;
        }
        utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& g) override {
            utec::algebra::Tensor<T, 2> output = g;
            auto sigma_it = sigma.begin();
            for (auto it = output.begin(); it != output.end(); ++it) {*it *= (*sigma_it * (1 - *sigma_it)); ++sigma_it;}
            return output;
        }
    };










}
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
