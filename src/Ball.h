//
// Created by User on 03/07/2025.
//

#ifndef BALL_H
#define BALL_H
#include "Paddle.h"
#include <random>

class Ball {
float w, h, speed, xprev, yprev;
    std::mt19937 rng;
    std::uniform_real_distribution<float> angle_distribution;
public:
    float x, y, dx, dy;
    bool right_goal_score, left_goal_score;
    Ball(float x, float y, float speed = 2.0f);
    void update(float deltaTime);
    void reset();
    void checkCollision(const Paddle& paddle);
    void checkWallCollision();
    void setStartDirection();
    void render(SDL_Renderer* renderer);
    [[nodiscard]] SDL_FRect getRect() const;
    bool linesIntersect(float x1, float y1, float x2, float y2,
                    float x3, float y3, float x4, float y4);
    bool lineIntersectsRect(float x1, float y1, float x2, float y2, const SDL_FRect& rect);
};



#endif //BALL_H
