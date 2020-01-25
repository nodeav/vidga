#include "Grid.hpp"
#include <random>
#include <vector>
#include <chrono>

int main(int argc, char **argv) {
    Circle circle{
            .color = {0, 0, 0},
            .position = {10, 10},
            .radius = 9
    };

    constexpr int numCircles = 5e7;
    constexpr int width = 500;
    constexpr int height = 500;
    constexpr int blockSize = 50;
    constexpr float maxRadiusPercent = 0.25;

    Grid grid(width, height, blockSize);

    std::vector<Circle> circles(numCircles);

    auto getColor = []() {
        return static_cast<uint8_t>(rand() % 256);
    };

    auto getPosX = [&width]() {
        return static_cast<uint16_t>(rand() % width);
    };

    auto getPosY = [&height]() {
        return static_cast<uint16_t>(rand() % height);
    };

    auto genRadius = [&]() {
        return static_cast<uint8_t>(rand() % static_cast<int>(std::min(width, height) * maxRadiusPercent));
    };

    for (auto i = 0; i < numCircles; i++) {
        circles[i] = Circle{
                .color = {getColor(), getColor(), getColor()},
                .position = {getPosX(), getPosY()},
                .radius = genRadius()
        };
    }

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (auto i = 0; i < numCircles; i++) {
        grid.addCircle(circles[i]);
    }
    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
//    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "[ms]" << std::endl;
    std::cout << "Per circle = " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / numCircles << "[ns]" << std::endl;


    return 0;
}