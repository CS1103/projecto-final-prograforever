#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "nn_interfaces.h"
#include "tensor.h"
namespace utec::neural_network {
    template<typename T>
     class Dense final : public utec::neural_network::ILayer<T> {
        utec::algebra::Tensor<T, 2> w, b, x, dw, db;
    public:
        template<typename InitWFun, typename InitBFun>
        Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun) {
            w = utec::algebra::Tensor<T, 2>(in_f, out_f);
            b = utec::algebra::Tensor<T, 2>(1, out_f);
            init_w_fun(w); init_b_fun(b);
        }
        Dense(const utec::algebra::Tensor<T, 2>& w_init, const utec::algebra::Tensor<T, 2>& b_init):
            w(w_init), b(b_init) {}

        utec::algebra::Tensor<T, 2> get_weights() const {return w;}
        utec::algebra::Tensor<T, 2> get_bias() const {return b;}
        void set_weights(const utec::algebra::Tensor<T, 2>& w_new) {w = w_new;}
        void set_bias(const utec::algebra::Tensor<T, 2>& b_new) {b = b_new;}

        utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& x) override {
            this->x = x;
            return utec::algebra::matrix_product(x, w) + b;
        }
        utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& dZ) override {
            dw = utec::algebra::matrix_product(transpose_2d(x), dZ);
            /*
            db.fill(0);
            for (int i=0; i<dZ.size(); ++i) {
                db(0, i%dZ.elements_per_layer[0]) += dZ.elements[i];
            }*/
            db = dZ.row_sums();
            return utec::algebra::matrix_product(dZ, transpose_2d(w));
        }
        void update_params(utec::neural_network::IOptimizer<T>& optimizer) override {
            optimizer.update(w, dw);
            optimizer.update(b, db);
        }
    };
}





#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
