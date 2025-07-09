#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "nn_activation.h"
#include "nn_dense.h"
#include "nn_interfaces.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include "tensor.h"
#include "training_observer.h"
#include <memory>



namespace utec::neural_network {
    template<typename T>
    class NeuralNetwork {
        std::vector<std::unique_ptr<ILayer<T>>> layers;
        std::vector<std::unique_ptr<ITrainingObserver<T>>> observers;
    public:
        void add_observer(std::unique_ptr<ITrainingObserver<T>> observer) {
            observers.emplace_back(std::move(observer));}
        const std::vector<std::unique_ptr<ILayer<T>>>& get_layers() const {return layers;}
        void add_layer(std::unique_ptr<ILayer<T>> layer) {layers.push_back(std::move(layer));}
        template <template <typename ...> class LossType,
            template <typename ...> class OptimizerType = SGD>
        void train( const Tensor<T,2>& X,const Tensor<T,2>& Y,
            const size_t epochs, const size_t batch_size, T learning_rate) {
                OptimizerType<T> optimizer(learning_rate);
                for (size_t epoch=0; epoch<epochs; ++epoch) {
                    T epoch_loss = 0; int batch_counter = 0;
                    for (size_t i=0; i<X.shape()[0]; i += batch_size) {
                        Tensor<T, 2> x_slice = X.slice_row(i, batch_size);
                        Tensor<T, 2> y_slice = Y.slice_row(i, batch_size);
                        Tensor<T, 2> A = x_slice;
                        for (auto& layer: layers) A = layer->forward(A);
                        LossType<T> loss_function(A, y_slice);
                        T batch_loss = loss_function.loss();
                        epoch_loss += batch_loss; ++batch_counter;

                        auto loss_gradient = loss_function.loss_gradient();
                        for (auto layerit = layers.rbegin(); layerit != layers.rend(); ++layerit)
                            loss_gradient = (*layerit)->backward(loss_gradient);
                        for (auto& layer: layers) layer->update_params(optimizer);
                        if constexpr(std::is_same_v<OptimizerType<T>, Adam<T>>) optimizer.step();
                    }
                    T average_loss = epoch_loss / T(batch_counter);
                    for (auto& obs: observers) obs->onEpochEnd(epoch, epochs, average_loss);
                }
        }
        Tensor<T,2> predict(const Tensor<T,2>& X) {
            Tensor<T, 2> result = X;
            for (auto& layer: layers) result = layer->forward(result);
            return result;
        }
    };
}




#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
