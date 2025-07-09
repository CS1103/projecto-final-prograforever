#ifndef GAME_H
#define GAME_H
#include <SDL3/SDL.h>
#include <SDL3_ttf/SDL_ttf.h>

#include "Ball.h"
#include "Paddle.h"
#include "neural_network.h"

struct State {
    float ballx, bally, vx, vy, right_paddle_y;
    void normalize(float max_ballx, float max_bally, float max_vx, float max_vy, float max_rpy) {
        ballx /= max_ballx;
        bally /= max_bally;
        vx /= max_vx;
        vy /= max_vy;
        right_paddle_y /= max_rpy;
    }
};

class Game {

private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    Paddle leftPaddle;
    Paddle rightPaddle;
    Ball ball;
    bool running, awaitingRestart;
    int leftScore, rightScore;
    double restartTimer;
    float successes = 0, fails = 0;
    TTF_Font* font = nullptr;
    utec::neural_network::NeuralNetwork<float>* network;

    void processInput();
    void processModelResponse();
    void update();
    void render();
    void renderScore();
public:
    Game(bool headless=false,
        const Paddle& leftPaddle = Paddle{50.0f, 250.0f},
        const Paddle& rightPaddle = Paddle{740.0f, 250.0f},
        const Ball& ball = Ball{395.0f, 295.0f, 300.0f});
    ~Game();
    [[nodiscard]] State getState() const;
    [[nodiscard]] bool is_awaiting_restart() const;
    void resetRandom();
    void auto_step(const int right_paddle_input);
    void set_network(utec::neural_network::NeuralNetwork<float>* network);
    void run();

};

#endif //GAME_H
