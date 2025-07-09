//
// Created by User on 03/07/2025.
//

#include "Ball.h"
#include <iostream>
#include <random>

Ball::Ball(float x, float y, float speed)
    : x(x), y(y), w(10), h(10), speed(speed), right_goal_score(false), left_goal_score(false) {
    std::random_device rd;
    rng = std::mt19937(rd());
    angle_distribution = std::uniform_real_distribution<float>(30.0f, 60.0f);
    setStartDirection();
}


void Ball::update(float deltaTime) {
    xprev = x; yprev = y;
    x += dx * speed * deltaTime;
    y += dy * speed * deltaTime;
}


void Ball::reset() {
    x = 395.0f; y = 295.0f;
    left_goal_score = false; right_goal_score = false;
    setStartDirection();
}


void Ball::checkCollision(const Paddle& paddle) {
    SDL_FRect ballRect = { x, y, w, h};
    SDL_FRect paddleRect = paddle.getRect();
    if (lineIntersectsRect(xprev, yprev, x, y, paddleRect)) {
        dx *= -1;
        if (dx > 0) x = paddleRect.x + paddleRect.w + 1;
        else x = paddleRect.x - w - 1;
    }

}

void Ball::checkWallCollision() {
    if (y <= 0) {y = 0; dy *= -1;}
    else if (y + h >= 600) {y = 600 - h; dy *= -1;}
    if (x + w < 0) right_goal_score = true;
    else if (x > 800) left_goal_score = true;
}

void Ball::render(SDL_Renderer* renderer) {
    SDL_FRect rect = getRect();
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderFillRect(renderer, &rect);
}

void Ball::setStartDirection() {
    std::uniform_int_distribution<int> x_distribution(0, 1);
    std::uniform_int_distribution<int> y_distribution(0, 1);
    float angle_radians = angle_distribution(rng) * M_PI/180.0f;
    dx = std::cos(angle_radians);
    dy = std::sin(angle_radians);
    if (x_distribution(rng) == 1) dx *= -1;
    if (y_distribution(rng) == 1) dy *= -1;
}



SDL_FRect Ball::getRect() const {return SDL_FRect{ x, y, w, h };}

bool Ball::linesIntersect(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4) {
    auto ccw = [](float ax, float ay, float bx, float by, float cx, float cy) {
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax);
    };

    return ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) &&
           ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4);

}

bool Ball::lineIntersectsRect(float x1, float y1, float x2, float y2, const SDL_FRect &rect) {
    float rx = rect.x;
    float ry = rect.y;
    float rw = rect.w;
    float rh = rect.h;

    return linesIntersect(x1, y1, x2, y2, rx,     ry,     rx + rw, ry)     ||  // Borde superior
           linesIntersect(x1, y1, x2, y2, rx,     ry,     rx,      ry + rh) ||  // Borde izquierdo
           linesIntersect(x1, y1, x2, y2, rx + rw, ry,     rx + rw, ry + rh) ||  // Borde derecho
           linesIntersect(x1, y1, x2, y2, rx,     ry + rh, rx + rw, ry + rh);   // Borde inferior

}

