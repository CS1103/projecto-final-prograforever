#include "Game.h"

#include <cmath>
#include <iomanip>
#include <SDL3_ttf/SDL_ttf.h>
#include <SDL3/SDL.h>
#include <iostream>
#include "Paddle.h"
#include "Ball.h"

Game::Game(bool headless, const Paddle& leftPaddle, const Paddle& rightPaddle, const Ball& ball):
    leftPaddle(leftPaddle), rightPaddle(rightPaddle), ball(ball), leftScore(0), rightScore(0) {

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "Error while opening SDL_Init: " << SDL_GetError() << std::endl;
        running = false;
        return;
    }

    if (!TTF_Init()) {running = false; return;}
    font = TTF_OpenFont("external/fonts/OpenSans-Regular.ttf", 40);
    if (!font) {
        std::cerr << "Error loading fonts: " << SDL_GetError() << std::endl;
        running = false; return;}

    Uint32 flags = 0; if (headless) flags = SDL_WINDOW_HIDDEN;

    window = SDL_CreateWindow("Pong AI", 800, 600, flags);
    if (!window) {
        std::cerr << "Error loading window: " << SDL_GetError() << std::endl;
        running = false; return; }

    renderer = SDL_CreateRenderer(window, nullptr);
    if (!renderer) {
        std::cerr << "Error creating renderer: " << SDL_GetError() << std::endl;
        running = false; return;
    }

    TTF_Init();
    running = true;
}

Game::~Game() {
    TTF_CloseFont(font); font = nullptr;
    TTF_Quit();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void Game::run() {
    restartTimer = 3; awaitingRestart = true;
    while (running) {
        processInput();
        processModelResponse();
        update();
        render();
        SDL_Delay(16); // Para que el juego corra a alrededor de 60 FPS
    }
    if (successes+fails != 0)
        std::cout << "Movement accuracy from match: " <<
            std::setprecision(4) << successes*100/(successes+fails) << "%\n";
    std::cout << "Right decisions: " << static_cast<int>(successes) << " Wrong decisions: " << static_cast<int>(fails) << "\n";
}

void Game::set_network(utec::neural_network::NeuralNetwork<float>* network) {this->network = network;}

void Game::processModelResponse() {
    if (!awaitingRestart) {
        State current_state = getState();
        current_state.normalize(800.0, 600.0, 300.0, 300.0, 600.0);
        utec::algebra::Tensor<float, 2> state_tensor(1, 5);
        state_tensor(0, 0) = current_state.ballx;
        state_tensor(0, 1) = current_state.bally;
        state_tensor(0, 2) = current_state.vx;
        state_tensor(0, 3) = current_state.vy;
        state_tensor(0, 4) = current_state.right_paddle_y;
        auto model_output = network->predict(state_tensor);
        rightPaddle.stop();
        if (model_output(0,1) > model_output(0,0) && model_output(0, 1) >= model_output(0, 2)) {
            rightPaddle.moveUp();
            if (ball.y < rightPaddle.y + rightPaddle.get_height()/2) ++successes; else ++fails;
        }
        else if (model_output(0,2) > model_output(0,0) && model_output(0, 2) > model_output(0, 1)) {
            rightPaddle.moveDown();
            if (ball.y > rightPaddle.y + rightPaddle.get_height()/2) ++successes; else ++fails;
        }
        else if (ball.y == rightPaddle.y + rightPaddle.get_height()/2) ++successes; else ++fails;
    }
}


void Game::auto_step(const int right_paddle_input) {
    leftPaddle.stop();
    if (ball.y < leftPaddle.y + leftPaddle.get_height()/2) leftPaddle.moveUp();
    else if (ball.y > leftPaddle.y + leftPaddle.get_height()/2) leftPaddle.moveDown();
    rightPaddle.stop();
    if (right_paddle_input == 1) rightPaddle.moveUp();
    else if (right_paddle_input == 2) rightPaddle.moveDown();
    update();
}

void Game::resetRandom() {
    awaitingRestart = false; restartTimer   = 0.0;
    //int w, h; SDL_GetWindowSize(window, &w, &h);
    int w=800, h=600;
    leftPaddle.y  = (h - leftPaddle.get_height())  / 2.0f;
    rightPaddle.y = (h - rightPaddle.get_height()) / 2.0f;
    ball.x = w / 2.0f; ball.y = h / 2.0f;

    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_real_distribution<float> angle_dist(-45.0f, +45.0f);
    std::uniform_int_distribution<int> side(0,1);
    float angle = angle_dist(gen) * (M_PI/180.0f);
    float speed = 300.0f;
    int sign = side(gen) ? +1 : -1;
    ball.dx = std::cos(angle) * speed * sign;
    ball.dy = std::sin(angle) * speed;
}

State Game::getState() const {
    return { ball.x, ball.y, ball.dx, ball.dy,
         rightPaddle.y };
}

bool Game::is_awaiting_restart() const {return awaitingRestart;}

void Game::processInput() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_EVENT_QUIT) {
            running = false;
        }
        if (event.type == SDL_EVENT_KEY_DOWN || event.type == SDL_EVENT_KEY_UP) {
            bool pressed = (event.type == SDL_EVENT_KEY_DOWN);

            switch (event.key.key) {
                case SDLK_W:
                    leftPaddle.stop();
                if (pressed) leftPaddle.moveUp();
                break;
                case SDLK_S:
                    leftPaddle.stop();
                if (pressed) leftPaddle.moveDown();
                break;
                case SDLK_ESCAPE:
                    running = false;
                break;
            }
        }
    }
}

void Game::update() {
    float deltaTime = 0.016f;
    if (awaitingRestart) {
        restartTimer -= deltaTime;
        if (restartTimer <= 0) awaitingRestart = false;
        return;
    }
    leftPaddle.update(deltaTime, 600);
    rightPaddle.update(deltaTime, 600);
    ball.update(deltaTime);
    ball.checkCollision(leftPaddle);
    ball.checkCollision(rightPaddle);
    ball.checkWallCollision();
    if (ball.left_goal_score) {
        ++leftScore; restartTimer = 3; awaitingRestart = true;
        ball.reset(); leftPaddle.reset(); rightPaddle.reset();
    }
    if (ball.right_goal_score) {
        ++rightScore; restartTimer = 3; awaitingRestart = true;
        ball.reset(); leftPaddle.reset(); rightPaddle.reset();
    }
}



void Game::render() {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    leftPaddle.render(renderer);
    rightPaddle.render(renderer);
    if (!awaitingRestart) ball.render(renderer);
    renderScore();
    if (awaitingRestart) {
        int countdown = static_cast<int>(ceil(restartTimer));
        std::string countdownText = std::to_string(countdown);
        SDL_Color color = {255, 255, 255, 255};

        SDL_Surface* surface = TTF_RenderText_Blended(font, countdownText.c_str(),
            countdownText.size(), color);
        SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
        SDL_DestroySurface(surface);

        float countdown_width = 0, countdown_height = 0;
        SDL_GetTextureSize(texture, &countdown_width, &countdown_height);

        SDL_FRect dst = {(800 - countdown_width) / 2.f, (600 - countdown_height) / 2.f,
            static_cast<float>(countdown_width), static_cast<float>(countdown_height)};
        SDL_RenderTexture(renderer, texture, nullptr, &dst);
        SDL_DestroyTexture(texture);
    }


    SDL_RenderPresent(renderer);
}

void Game::renderScore() {
    std::string score_text = std::to_string(leftScore) + " - " + std::to_string(rightScore);
    SDL_Color white{255, 255, 255, 255};
    SDL_Surface* surface = TTF_RenderText_Blended(font, score_text.c_str(),
        score_text.size(), white);

    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    SDL_DestroySurface(surface);

    float text_width = 0, text_height = 0;
    SDL_GetTextureSize(texture, &text_width, &text_height);
    SDL_FRect dst{ (800 - text_width) / 2.f, 20.f, float(text_width), float(text_height) };
    SDL_RenderTexture(renderer, texture, nullptr, &dst);
    SDL_DestroyTexture(texture);

}
