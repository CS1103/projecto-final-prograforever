//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#include "tensor.h"
#include "nn_interfaces.h"
namespace utec::neural_network {
    template<typename T>
      class MSELoss final: public ILoss<T, 2> {
        utec::algebra::Tensor<T, 2> y_prediction;
        utec::algebra::Tensor<T, 2> y_true;
    public:
        MSELoss(const utec::algebra::Tensor<T,2>& y_prediction, const utec::algebra::Tensor<T,2>& y_true):
        y_prediction(y_prediction), y_true(y_true) {}
        T loss() const override {
            T sum = 0; auto pred_it = y_prediction.cbegin();
            for (auto it = y_true.cbegin(); it != y_true.cend(); ++it) {sum += (*pred_it-*it)*(*pred_it-*it); ++pred_it;}
            return sum / T(y_true.size());
        }
        utec::algebra::Tensor<T,2> loss_gradient() const override {
            return (y_prediction - y_true) * (T(2)/y_true.size());
        }
    };

    template<typename T>
      class BCELoss final: public ILoss<T, 2> {
        utec::algebra::Tensor<T, 2> y_prediction;
        utec::algebra::Tensor<T, 2> y_true;
    public:
        BCELoss(const utec::algebra::Tensor<T,2>& y_prediction, const utec::algebra::Tensor<T,2>& y_true):
        y_prediction(y_prediction), y_true(y_true) {}
        T loss() const override {
            T sum = 0; auto pred_it = y_prediction.cbegin();
            for (auto it = y_true.cbegin(); it != y_true.cend(); ++it) {
                sum -= *it * std::log(*pred_it) + (1- *it)*std::log(1- *pred_it); ++pred_it;
            }
            return sum / T(y_true.size());
        }
        utec::algebra::Tensor<T,2> loss_gradient() const override {
            return (y_true/y_prediction - (1-y_true)/(1-y_prediction)) / -T(y_true.size());
        }
    };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
