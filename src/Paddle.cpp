//
// Created by User on 03/07/2025.
//

#include "Paddle.h"

Paddle::Paddle(float x, float y, float speed, float width, float height)
    : x(x), y(y), width(width), height(height), speed(speed), dy(0.0f) {}

float Paddle::get_height() const {return height;}
float Paddle::get_width() const {return width;}
float Paddle::get_speed() const {return speed;}

void Paddle::moveUp() {dy = -1.0f;}
void Paddle::moveDown() {dy = 1.0f;}
void Paddle::reset() {stop(); y = 300 - height/2;}
void Paddle::stop() {dy = 0.0f;}

void Paddle::update(float deltaTime, int windowHeight) {
    y += dy * speed * deltaTime;
    if (y < 0.0f) y = 0.0f;
    if (y + height > static_cast<float>(windowHeight))
        y = static_cast<float>(windowHeight) - height;
}

void Paddle::render(SDL_Renderer* renderer) {
    SDL_FRect rect = getRect();
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderFillRect(renderer, &rect);
}

SDL_FRect Paddle::getRect() const {
    return SDL_FRect{ x, y, width, height };
}

void Paddle::setSpeed(float newSpeed) {
    speed = newSpeed;
}

float Paddle::getSpeed() const {
    return speed;
}