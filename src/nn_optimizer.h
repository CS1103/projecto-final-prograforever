//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
//Estructura de Adams(uso de unordered_map y de t) sugerida por IA a pesar de que yo no la pedí en la prompt
#include "nn_interfaces.h"
#include "tensor.h"
#include <unordered_map>
namespace utec::neural_network {
    template<typename T>
    class SGD final : public utec::neural_network::IOptimizer<T> {
        T learning_rate;
    public:
        explicit SGD(T learning_rate = 0.01): learning_rate(learning_rate) {}
        void update(utec::algebra::Tensor<T, 2>& params, const utec::algebra::Tensor<T, 2>& grads) override {
            params = params - grads*learning_rate;
        }
    };

    template<typename T>
      class Adam final : public utec::neural_network::IOptimizer<T> {
        std::unordered_map<utec::algebra::Tensor<T, 2>*, utec::algebra::Tensor<T, 2>> m1;
        std::unordered_map<utec::algebra::Tensor<T, 2>*, utec::algebra::Tensor<T, 2>> m2;
        T learning_rate, beta1, beta2, epsilon; size_t t = 1;
    public:
        explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8):
            learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon) {}
        void update(utec::algebra::Tensor<T, 2>& params, const utec::algebra::Tensor<T, 2>& grads) override {
            if (!m1.contains(&params)) {
                Tensor<T, 2> m_input(params.shape()); m_input.fill(0); m1.emplace(&params, std::move(m_input));}
            if (!m2.contains(&params)) {
                Tensor<T, 2> m_input(params.shape()); m_input.fill(0); m2.emplace(&params, std::move(m_input));}
            utec::algebra::Tensor<T, 2>& m1_actual = m1[&params];
            utec::algebra::Tensor<T, 2>& m2_actual = m2[&params];
            m1_actual = m1_actual * beta1 + (1-beta1) * grads;
            m2_actual = m2_actual * beta2 + (1-beta2) * (grads * grads);
            utec::algebra::Tensor<T, 2> m1_corr = m1_actual / T(1 - std::pow(beta1, t));
            utec::algebra::Tensor<T, 2> m2_corr = m2_actual / T(1 - std::pow(beta2, t));
            params = params - learning_rate * m1_corr / (sqrt(m2_corr) + epsilon);
        }
        void step() override {++t;}
    };
}
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
