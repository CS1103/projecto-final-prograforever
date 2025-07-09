
#include "neural_network.h"
#include "model_serializer.h"
#include "Paddle.h"
#include "Ball.h"
#include "Game.h"
#include "tensor.h"
#include <vector>
#include <memory>
#include <filesystem>

struct Sample {
    State state;  // [ball.x, ball.y, vx, vy, rightPaddle.y]
    int action; // 0=no se mueve, 1=sube, 2=baja
    Sample(State state, int action): state(state), action(action) {}
};

int expertPolicy(const State& pong_state) {
    float paddle_center = pong_state.right_paddle_y + 50.0/600.0;
    if (pong_state.bally > paddle_center) return 2;
    if (pong_state.bally < paddle_center) return 1;
    return 0;
}

void generateDataset(std::vector<Sample>& data, size_t N) {
    Game pong(true);
    data.reserve(N);
    while (data.size() < N) {
        pong.resetRandom();
        while (!pong.is_awaiting_restart() && data.size() < N) {
            State pong_state = pong.getState();
            pong_state.normalize(800.0f, 600.0f, 300.0f, 300.0f, 600.0f);
            Sample samp(pong_state, expertPolicy(pong_state));
            data.push_back(samp);
            pong.auto_step(samp.action);
        }
    }
}

utec::algebra::Tensor<float,2> make_state_tensor(const std::vector<Sample>& sample_vector) {
    size_t samples_amount = sample_vector.size();
    utec::algebra::Tensor<float,2> X(samples_amount, 5);
    for (size_t i = 0; i < samples_amount; ++i)
        for (int j = 0; j < 5; ++j) {
            if (j == 0) X(i, j) = sample_vector[i].state.ballx;
            if (j == 1) X(i, j) = sample_vector[i].state.bally;
            if (j == 2) X(i, j) = sample_vector[i].state.vx;
            if (j == 3) X(i, j) = sample_vector[i].state.vy;
            if (j == 4) X(i, j) = sample_vector[i].state.right_paddle_y;
        }
    return X;
}

utec::algebra::Tensor<float,2> make_action_tensor(const std::vector<Sample>& sample_vector) {
    size_t samples_amount = sample_vector.size();
    utec::algebra::Tensor<float,2> Y(samples_amount, 3);
    Y.fill(0);
    for (size_t i = 0; i < samples_amount; ++i)
        Y(i, sample_vector[i].action) = 1.0f;
    return Y;
}

int main() {
    const int training_data_size = 30000;
    const int epochs = 500;
    const int batch_size = 64;
    const float learning_rate = 0.01;

    std::random_device rd;
    std::mt19937_64 rng(rd());
    auto init_he = [&](utec::algebra::Tensor<float,2> &M){
        int fan_in = int(M.shape()[0]);
        float stddev = std::sqrt(2.0f / fan_in);
        std::normal_distribution<float> dist(0.0f, stddev);
        for (size_t i = 0; i < M.shape()[0]; ++i)
            for (size_t j = 0; j < M.shape()[1]; ++j)
                M(i,j) = dist(rng);
    };


    std::vector<Sample> dataset;
    std::cout << "Creating dataset from data size " << training_data_size << "...\n";
    generateDataset(dataset, training_data_size);

    std::array<size_t,3> counts = {0,0,0};
    for (auto& samp : dataset) {
        counts[samp.action]++;
    }

    size_t total = dataset.size();
    std::cout << "Action summary from data generation process:\n";
    for (int a = 0; a < 3; ++a) {
        float pct = 100.0f * counts[a] / total;
        const char* name = (a==0 ? "idle" : (a==1 ? "up" : "down"));
        std::cout
          << " Action " << a << " (" << name << "): "
          << counts[a]
          << " cases registered (" << pct << "%)\n";
    }


    utec::algebra::Tensor<float,2> X = make_state_tensor(dataset);
    utec::algebra::Tensor<float,2> Y = make_action_tensor(dataset);

    utec::neural_network::NeuralNetwork<float> network;
    network.add_observer(std::make_unique<TrainingProgressDisplay<float>>());


    network.add_layer(std::make_unique<utec::neural_network::Dense<float>>(5, 32, init_he, init_he));
    network.add_layer(std::make_unique<utec::neural_network::ReLU<float>>());
    network.add_layer(std::make_unique<utec::neural_network::Dense<float>>(32, 3, init_he, init_he));
    network.add_layer(std::make_unique<utec::neural_network::Sigmoid<float>>());

    network.train<utec::neural_network::BCELoss>(X, Y, epochs, batch_size, learning_rate);
    std::cout << "Training process concluded with data size " << training_data_size << ".\n";
    serializer training_serializer;
    training_serializer.save(network, "data/model.dat");
    std::cout << "Model data stored successfully.\n";
    return 0;
}
