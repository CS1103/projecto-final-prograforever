#include <iostream>

#include "Game.h"
#include "Paddle.h"
#include "Ball.h"
#include "neural_network.h"
#include "model_serializer.h"
#include <SDL3/SDL.h>

int main() {
    utec::neural_network::NeuralNetwork<float> network;
    serializer loader;
    loader.load(network, "data/model.dat");
    auto& layers = network.get_layers();
    float padleSpeed = 300.0f;
    Paddle leftPaddle(20, 250, padleSpeed);
    Paddle rightPaddle(780, 250, padleSpeed);
    Ball ball(395.0f, 295.0f, 300.0f);

    std::cout << "  ____   ___   _   _   ____        _    ___       \n";
    std::cout << " |  _ \\ / _ \\ | \\ | | / ___|      /_\\  /_ _\\      \n";
    std::cout << " | |_) | | | ||  \\| || |  _      //_\\\\  | |       \n";
    std::cout << " |  __/| |_| || |\\  || |_| |    / ___ \\_| |_      \n";
    std::cout << " |_|    \\___/ |_| \\_| \\____|   /_/   \\_\\___/      \n\n";

    Game game(false, leftPaddle, rightPaddle, ball);
    game.set_network(&network);
    std::cout << "Starting new game of Pong AI...\n";
    std::cout << "Press W to move up and S to move down.\nPress ESC to exit.\n\n";
    game.run();
    return 0;
}
