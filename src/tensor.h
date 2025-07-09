//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#include <vector>
#include <cmath>
#include <array>
#include <string>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <limits>
namespace utec::algebra {

    template<typename T>
    static constexpr T eps = std::numeric_limits<T>::epsilon();


    template<typename T, int Rank>
    class Tensor {
    private:
        std::array<size_t, Rank> dimensions;
        std::array<size_t, Rank> elements_per_layer;
        std::vector<T> elements;
    public:
        template<typename T_, int Rank_>
        friend std::ostream& operator<<(std::ostream& out, const Tensor<T_, Rank_>& t);
        template<typename T_, int Rank_>
        friend Tensor<T_, Rank_> matrix_product(const Tensor<T_, Rank_>& t1, const Tensor<T_, Rank_>& t2);

        void update_elements_per_layer() {
            int layernum = 1;
            for (int i=0; i<Rank; ++i) {
                for (int j=i+1; j<Rank; ++j) layernum *= dimensions[j];
                elements_per_layer[i] = layernum;
                layernum = 1;
            }
        }

        static std::array<size_t, Rank> return_elements_per_layer(const std::array<size_t, Rank>& dim) {
            std::array<size_t, Rank> epl;
            int layernum = 1;
            for (int i=0; i<Rank; ++i) {
                for (int j=i+1; j<Rank; ++j) layernum *= dim[j];
                epl[i] = layernum;
                layernum = 1;
            }
            return epl;
        }


        Tensor(const std::array<size_t, Rank>& dimensions, const std::array<size_t, Rank>& elements_per_layer,
               std::vector<T> elements): dimensions(dimensions), elements_per_layer(elements_per_layer),
                                         elements(elements) {}

        Tensor() = default;

        Tensor(const std::array<size_t, Rank>& shape) {
            dimensions = shape;
            update_elements_per_layer();
            int linear_size = 1;
            for (int i=0; i<Rank; ++i) linear_size *= dimensions[i];
            elements.resize(linear_size);
        }

        template<typename... Dims>
        requires (sizeof...(Dims)==Rank)
        Tensor(Dims ...dims) {
            if (sizeof...(dims) != Rank) throw std::runtime_error("Number of dimensions do not match with " + std::to_string(Rank));
            std::vector<size_t> dim_vector;
            (dim_vector.push_back(dims), ...);
            for (int i=0; i<Rank; ++i) dimensions[i] = dim_vector[i];
            update_elements_per_layer();
            elements.clear();
            int linear_size = 1;
            for (int i=0; i<Rank; ++i) linear_size *= dimensions[i];
            elements.resize(linear_size);
        }

        Tensor(const Tensor<T, Rank>& other): dimensions(other.dimensions), elements_per_layer(other.elements_per_layer),
                                              elements(other.elements) {}

        Tensor& operator=(const Tensor<T, Rank>& other) {
            dimensions = other.dimensions;
            elements_per_layer = other.elements_per_layer;
            elements = other.elements;
            return *this;
        }

        Tensor& operator=(std::initializer_list<T> lst) {
            if (lst.size() != elements.size()) throw std::runtime_error("Data size does not match tensor size");
            elements.assign(lst);
            return *this;
        }

        template<typename... Idxs> T& operator()(Idxs... idxs) {
            if (sizeof...(idxs) != Rank) throw std::runtime_error("Number of dimensions do not match with " + std::to_string(Rank));
            std::vector<int> requested = {idxs...};
            auto it = elements.begin();
            for (int i=0; i<Rank; ++i) std::advance(it, elements_per_layer[i] * requested[i]);
            return *it;
        }

        template<typename... Idxs> const T& operator()(Idxs... idxs) const {
            if (sizeof...(idxs) != Rank) throw std::runtime_error("Number of dimensions do not match with " + std::to_string(Rank));
            std::vector<int> requested = {idxs...};
            auto it = elements.begin();
            for (int i=0; i<Rank; ++i) std::advance(it, elements_per_layer[i] * requested[i]);
            return *it;
        }

        Tensor row_sums() const {
            auto new_dimensions = dimensions; new_dimensions[0] = 1;
            auto new_epl = return_elements_per_layer(new_dimensions);
            std::vector<T> new_elements; new_elements.resize(elements_per_layer[0]);
            for (int i=0; i<elements.size(); ++i) {
                new_elements[i%elements_per_layer[0]] += elements[i];
            }
            return Tensor(new_dimensions, new_epl, new_elements);
        }


        const std::array<size_t, Rank>& shape() const noexcept {return dimensions;}

        void reshape(const std::array<size_t, Rank>& new_shape) {
            int linear_size = 1;
            for (int i=0; i<Rank; ++i) linear_size *= new_shape[i];
            dimensions = new_shape;
            elements.resize(linear_size);
            update_elements_per_layer();
        }

        template<typename... Dims> void reshape(Dims ...dims) {
            if (sizeof...(dims) != Rank) throw std::runtime_error("Number of dimensions do not match with " + std::to_string(Rank));
            std::vector<size_t> dim_vector;
            (dim_vector.push_back(dims), ...);
            int linear_size = 1;
            for (int i=0; i<Rank; ++i) linear_size *= dim_vector[i];
            for (int i=0; i<Rank; ++i) dimensions[i] = dim_vector[i];
            elements.resize(linear_size);
            update_elements_per_layer();
        }

        void fill(const T& value) noexcept {
            for (auto it = elements.begin(); it != elements.end(); ++it) *it = value;
        }

        auto begin() {return elements.begin();}
        auto end() {return elements.end();}
        auto cbegin() const {return elements.cbegin();}
        auto cend() const {return elements.cend();}

        auto size() const {return elements.size();}

        template<typename Function> friend Tensor apply(const Tensor& t, Function op) {
            Tensor output(t.shape());
            std::transform(t.cbegin(), t.cend(), output.begin(), [&](const T& v){ return op(v);});
            return output;
        }

        Tensor slice(const size_t first, const size_t amount) const {
            if (first >= size()) throw std::runtime_error("Índice fuera del rango del tensor");
            Tensor<T, 2> output(1, std::min(amount, size()-first));
            auto own = cbegin(); std::advance(own, first); auto out = output.begin();
            for (size_t i=1; i<=std::min(amount, size()-first); ++i) {
                *out = *own; ++out; ++own;
            }
            return output;
        }

        Tensor slice_row(const size_t first, const size_t amount) const {
            if (first*elements_per_layer[0] >= size()) throw std::runtime_error("Índice fuera del rango del tensor");
            Tensor<T, 2> output(std::min(amount, dimensions[0]-first), elements_per_layer[0]);
            auto own = cbegin(); std::advance(own, first*elements_per_layer[0]); auto out = output.begin();
            for (size_t i=1; i<=std::min(amount, dimensions[0]-first)*elements_per_layer[0]; ++i) {
                *out = *own; ++out; ++own;
            }
            return output;
        }

        friend std::vector<T> copy_to_resized(const Tensor& t, const std::array<size_t, Rank>& new_epl, const int new_size,
                                              const std::array<bool, Rank>& indexes_to_copy) {
            std::vector<T> output(new_size);
            for (int index = 0; index != new_size; ++index) {
                int old_index = 0;
                for (int i=0; i<Rank; ++i) {
                    int first_pos_i;
                    if (indexes_to_copy[i]) first_pos_i = 0;
                    else if (i == 0) first_pos_i = int(index) / int(new_epl[i]);
                    else first_pos_i = int(index % new_epl[i-1]) / int(new_epl[i]);
                    old_index += first_pos_i * t.elements_per_layer[i];
                }
                output[index] = t.elements[old_index];
            }
            return output;
        }


        friend Tensor broadcast(const Tensor& t, const std::array<size_t, Rank>& new_dimensions) {
            if (t.dimensions.size() != new_dimensions.size()) throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");
            std::array<bool, Rank> indexes_to_copy;
            for (int i=0; i<t.dimensions.size(); ++i) {
                if (t.dimensions[i]==0 || new_dimensions[i]==0 || (
                        t.dimensions[i] != 1 && t.dimensions[i] != new_dimensions[i]
                    )) throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");
                if (t.dimensions[i] != new_dimensions[i] && t.dimensions[i] == 1) indexes_to_copy[i] = true;
                else indexes_to_copy[i] = false;
            }
            auto new_epl = return_elements_per_layer(new_dimensions);
            int linear_size = 1;
            for (int i=0; i<Rank; ++i) linear_size *= new_dimensions[i];
            std::vector<T> new_elements = copy_to_resized(t, new_epl, linear_size, indexes_to_copy);
            return Tensor(new_dimensions, new_epl, new_elements);
        }

        Tensor operator+(const Tensor& other) const {
            if (dimensions.size() != other.dimensions.size()) throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");
            std::vector<T> element_sum;
            if (dimensions != other.dimensions) {
                std::array<size_t, Rank> new_dimensions;
                for (int i=0; i<dimensions.size(); ++i) new_dimensions[i] = std::max(dimensions[i], other.dimensions[i]);
                Tensor tensor1 = broadcast(*this, new_dimensions);
                Tensor tensor2 = broadcast(other, new_dimensions);
                for (int i=0; i<tensor1.elements.size(); ++i)
                    element_sum.push_back(tensor1.elements[i]+tensor2.elements[i]);
                return Tensor(tensor1.dimensions, tensor1.elements_per_layer, element_sum);
            }
            for (int i=0; i<elements.size(); ++i)
                element_sum.push_back(elements[i]+other.elements[i]);
            return Tensor(dimensions, elements_per_layer, element_sum);
        }

        Tensor operator+(const T& scalar) const {
            std::vector<T> element_sum;
            for (int i=0; i<elements.size(); ++i) {
                element_sum.push_back(elements[i]+scalar);
            }
            return Tensor(dimensions, elements_per_layer, element_sum);
        }

        friend Tensor operator+(const T& scalar, const Tensor& t) {
            std::vector<T> element_sum;
            for (int i=0; i<t.elements.size(); ++i) {
                element_sum.push_back(scalar + t.elements[i]);
            }
            return Tensor(t.dimensions, t.elements_per_layer, element_sum);
        }


        Tensor operator-(const Tensor& other) const {
            if (dimensions.size() != other.dimensions.size()) throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");
            std::vector<T> element_subtraction;
            if (dimensions != other.dimensions) {
                std::array<size_t, Rank> new_dimensions;
                for (int i=0; i<dimensions.size(); ++i) new_dimensions[i] = std::max(dimensions[i], other.dimensions[i]);
                Tensor tensor1 = broadcast(*this, new_dimensions);
                Tensor tensor2 = broadcast(other, new_dimensions);
                for (int i=0; i<tensor1.elements.size(); ++i)
                    element_subtraction.push_back(tensor1.elements[i]-tensor2.elements[i]);
                return Tensor(tensor1.dimensions, tensor1.elements_per_layer, element_subtraction);
            }
            for (int i=0; i<elements.size(); ++i)
                element_subtraction.push_back(elements[i]-other.elements[i]);
            return Tensor(dimensions, elements_per_layer, element_subtraction);
        }

        Tensor operator-(const T& scalar) const {
            std::vector<T> element_subtraction;
            for (int i=0; i<elements.size(); ++i) {
                element_subtraction.push_back(elements[i]-scalar);
            }
            return Tensor(dimensions, elements_per_layer, element_subtraction);
        }

        friend Tensor operator-(const T& scalar, const Tensor& t) {
            std::vector<T> element_subtraction;
            for (int i=0; i<t.elements.size(); ++i) {
                element_subtraction.push_back(scalar - t.elements[i]);
            }
            return Tensor(t.dimensions, t.elements_per_layer, element_subtraction);
        }

        Tensor operator*(const Tensor& other) const {
            if (dimensions.size() != other.dimensions.size()) throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");
            std::vector<T> element_product;
            if (dimensions != other.dimensions) {
                std::array<size_t, Rank> new_dimensions;
                for (int i=0; i<dimensions.size(); ++i) new_dimensions[i] = std::max(dimensions[i], other.dimensions[i]);
                Tensor tensor1 = broadcast(*this, new_dimensions);
                Tensor tensor2 = broadcast(other, new_dimensions);
                for (int i=0; i<tensor1.elements.size(); ++i)
                    element_product.push_back(tensor1.elements[i]*tensor2.elements[i]);
                return Tensor(tensor1.dimensions, tensor1.elements_per_layer, element_product);
            }
            for (int i=0; i<elements.size(); ++i)
                element_product.push_back(elements[i]*other.elements[i]);
            return Tensor(dimensions, elements_per_layer, element_product);
        }

        Tensor operator*(const T& scalar) const {
            std::vector<T> element_product;
            for (int i=0; i<elements.size(); ++i) {
                element_product.push_back(elements[i] * scalar);
            }
            return Tensor(dimensions, elements_per_layer, element_product);
        }

        friend Tensor operator*(const T& scalar, const Tensor& t) {
            std::vector<T> element_product;
            for (int i=0; i<t.elements.size(); ++i) {
                element_product.push_back(scalar * t.elements[i]);
            }
            return Tensor(t.dimensions, t.elements_per_layer, element_product);
        }

        Tensor operator/(const T& scalar) const {
            std::vector<T> element_division;
            for (int i=0; i<elements.size(); ++i) {
                element_division.push_back(elements[i] / scalar);
            }
            return Tensor(dimensions, elements_per_layer, element_division);
        }

        friend Tensor operator/(const T& scalar, const Tensor& t) {
            std::vector<T> element_division;
            for (int i=0; i<t.elements.size(); ++i) {
                element_division.push_back(scalar / t.elements[i]);
            }
            return Tensor(t.dimensions, t.elements_per_layer, element_division);
        }

        Tensor operator/(const Tensor& other) const {
            if (dimensions.size() != other.dimensions.size()) throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");
            std::vector<T> element_quotient;
            if (dimensions != other.dimensions) {
                std::array<size_t, Rank> new_dimensions;
                for (int i=0; i<dimensions.size(); ++i) new_dimensions[i] = std::max(dimensions[i], other.dimensions[i]);
                Tensor tensor1 = broadcast(*this, new_dimensions);
                Tensor tensor2 = broadcast(other, new_dimensions);
                for (int i=0; i<tensor1.elements.size(); ++i) {
                    if (std::abs(tensor2.elements[i]) < eps<T>) {
                        if (tensor2.elements[i] < 0) element_quotient.push_back(tensor1.elements[i]/-eps<T>);
                        else element_quotient.push_back(tensor1.elements[i]/eps<T>);
                    } else element_quotient.push_back(tensor1.elements[i]/tensor2.elements[i]);
                }
                return Tensor(tensor1.dimensions, tensor1.elements_per_layer, element_quotient);
            }
            for (int i=0; i<elements.size(); ++i) {
                if (std::abs(other.elements[i]) < eps<T>) {
                    if (other.elements[i] < 0) element_quotient.push_back(elements[i]/-eps<T>);
                    else element_quotient.push_back(elements[i]/eps<T>);
                } else element_quotient.push_back(elements[i]/other.elements[i]);
            }
            return Tensor(dimensions, elements_per_layer, element_quotient);
        }

        friend Tensor<T, Rank> sqrt(const Tensor& t) {
            std::vector<T> element_root;
            for (int i=0; i<t.elements.size(); ++i) {
                element_root.push_back(sqrt(t.elements[i]));
            }
            return Tensor(t.dimensions, t.elements_per_layer, element_root);
        }



        static std::vector<T> transpose_2d_vector(const std::vector<T>& v, const int dim1, const int dim2) {
            std::vector<T> output(v.size());
            for (int old_index = 0; old_index < v.size(); ++old_index) {
                int new_index = (old_index/dim2) + (old_index % dim2) * dim1;
                output[new_index] = v[old_index];
            }
            return output;
        }


        friend Tensor transpose_2d(const Tensor& t) {
            if (t.dimensions.size() == 1) throw std::runtime_error("Cannot transpose 1D tensor: need at least 2 dimensions");

            std::vector<T> new_elements(t.elements.size());
            std::array<size_t, Rank> new_dimensions;
            for (int i=0; i<Rank; ++i) new_dimensions[i] = t.dimensions[i];
            size_t m = t.dimensions[Rank-2]; size_t n = t.dimensions[Rank-1];
            new_dimensions[Rank-2] = n; new_dimensions[Rank-1] = m;
            std::array<size_t, Rank> new_epl = return_elements_per_layer(new_dimensions);
            int amount_of_matrices = 1;
            for (int i=0; i<Rank-2; ++i) amount_of_matrices *= new_dimensions[i];

            for (int i=0; i<amount_of_matrices; ++i) {
                std::vector<T> v(t.elements.begin()+i*m*n, t.elements.begin()+(i+1)*m*n);
                std::vector<T> tv = transpose_2d_vector(v, m, n);
                std::copy(tv.begin(), tv.end(), new_elements.begin()+i*m*n);
            }
            return Tensor(new_dimensions, new_epl, new_elements);
        }

    };


    template<typename T_>
    std::vector<T_> matrix_multiplication_2d(const std::vector<T_>& v1, const std::vector<T_>& v2,
                                             const int m, const int n, const int p) {
        std::vector<T_> output(m*p);
        for (int i=0; i<m*p; ++i) {
            int v1_pos = (i/p) *n; int v2_pos = i % p;
            for (int j=0; j<n; ++j) {
                output[i] += v1[v1_pos]*v2[v2_pos];
                v1_pos++; v2_pos += p;
            }
        }
        return output;
    }

    template<typename T_, int Rank_>
    Tensor<T_, Rank_> matrix_product(const Tensor<T_, Rank_>& t1, const Tensor<T_, Rank_>& t2) {
        if (Rank_ == 1) return t1 * t2;
        if (t1.dimensions[Rank_-1] != t2.dimensions[Rank_-2]) throw std::runtime_error(
            "Matrix dimensions are incompatible for multiplication");
        int m = t1.dimensions[Rank_-2]; int n = t1.dimensions[Rank_-1]; int p = t2.dimensions[Rank_-1];
        int linear_size = m*p;
        for (int i=0; i<Rank_-2; ++i) {
            if (t1.dimensions[i] != t2.dimensions[i]) throw std::runtime_error(
                "Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
            linear_size *= t1.dimensions[i];
        }

        std::vector<T_> new_elements(linear_size);
        std::array<size_t, Rank_> new_dimensions = t1.dimensions; new_dimensions[Rank_-1] = p;
        std::array<size_t, Rank_> new_epl = Tensor<T_, Rank_>::return_elements_per_layer(new_dimensions);

        for (int i=0; i<linear_size/(m*p); ++i) {
            std::vector<T_> v1(t1.elements.begin()+i*m*n, t1.elements.begin()+(i+1)*m*n);
            std::vector<T_> v2(t2.elements.begin()+i*n*p, t2.elements.begin()+(i+1)*n*p);
            std::vector<T_> submatrix = matrix_multiplication_2d<T_>(v1, v2, m, n, p);
            std::copy(submatrix.begin(), submatrix.end(), new_elements.begin()+i*m*p);
        }
        return Tensor<T_, Rank_>(new_dimensions, new_epl, new_elements);
    }


    template<typename T, int Rank>
    std::ostream& operator<<(std::ostream& out, const Tensor<T, Rank> & t) {
        if (Rank == 1) {
            for (int i=0; i<t.elements.size(); ++i) out << t.elements[i] << " ";
            return out;
        }
        int count = 0;
        out << "{\n";
        while (count < t.elements.size()) {
            for (int i=0; i<Rank-2; ++i) {
                if (count % t.elements_per_layer[i] == 0) {
                    out << "{\n";
                }
            }
            out << t.elements[count] << " ";
            ++count;
            if (count % t.elements_per_layer[Rank-2] == 0) out << "\n";
            for (int i=Rank-3; i>=0; --i) {
                if (count % t.elements_per_layer[i] == 0) {
                    out << "}\n";
                }
            }
        }
        out << "}\n";
        return out;
    }



}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
