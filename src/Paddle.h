//
// Created by User on 03/07/2025.
//

#ifndef PADDLE_H
#define PADDLE_H

#include <SDL3/SDL.h>

class Paddle {
    float width, height;
    float speed;
    float dy;
public:
    float x, y;
    Paddle(float x, float y, float speed = 300.0f , float width = 10.0f, float height = 100.0f);
    [[nodiscard]] float get_height() const;
    [[nodiscard]] float get_width() const;
    [[nodiscard]] float get_speed() const;
    void moveUp();
    void moveDown();
    void reset();
    void stop(); // para detener el movimiento
    void update(float deltaTime, int windowHeight);
    void render(SDL_Renderer* renderer);
    void setSpeed(float newSpeed);
    float getSpeed() const;
    SDL_FRect getRect() const;
};

#endif //PADDLE_H
