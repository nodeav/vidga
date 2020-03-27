#include <vector>
#include <cstdint>
#include <iostream>

struct Color {
    uint8_t r, g, b, a;
};

struct Position {
    uint16_t x, y;

    friend std::ostream &operator<<(std::ostream &os, const Position &p) {
        return os << "{ x: " << p.x << ", y: " << p.y << " }";
    }
};

struct BBox {
    Position tl, br;
};

struct Circle {
    Color color;
    Position position;
    uint16_t radius;
    int id = 0;

    friend std::ostream &operator<<(std::ostream &os, const Circle &c) {
        return os << "{ id: " << c.id << ", radius: " << c.radius << ", position: " << c.position << " }";
    }

    BBox toBBox() const {
        uint16_t left = std::max(position.x, radius) - radius;
        uint16_t top = std::max(position.y, radius) - radius;
        uint16_t right = position.x + radius;
        uint16_t bottom = position.y + radius;
        return {
                .tl = {.x = left, .y = top},
                .br = {.x = right, .y = bottom}
        };
    }
};


class Grid {
    struct Block {
        std::vector<Circle> circles;
    };

    int blockSize;
    std::vector<Block> blocks;
    uint8_t nBlocksX, nBlocksY;

public:
    Grid(int width, int height, int blockSize);

    void addCircle(const Circle &c);

    void getCircles(const Position &pos);

    friend std::ostream &operator<<(std::ostream &os, const Grid &grid) {
        os << "[";
        for (auto y = 0; y < grid.nBlocksY; y++) {
            os << "\t";
            for (auto x = 0; x < grid.nBlocksX; x++) {
                os << "[" << grid.blocks[y * grid.nBlocksY + x].circles.size() << "],";
//                os << "[";
//                for (const auto &circle : grid.blocks[y * grid.nBlocksY + x].circles) {
//                    os << circle.id << ", ";
//                }
//                os << "],";
            }
            os << "\n";
        }
        return os << "]";
    };

};

void Grid::addCircle(const Circle &c) {
    auto bbox = c.toBBox();
    auto fromX = bbox.tl.x / blockSize, fromY = bbox.tl.y / blockSize;
    auto toX = std::min(static_cast<int>(nBlocksX), bbox.br.x / blockSize + 1);
    auto toY = std::min(static_cast<int>(nBlocksY), bbox.br.y / blockSize + 1);

    // TODO: Improve cache efficiency (reduce random access)
    for (auto y = fromY; y < toY; y++) {
        for (auto x = fromX; x < toX; x++) {
            blocks[y * nBlocksY + x].circles.push_back(c);
        }
    }
}

void Grid::getCircles(const Position &pos) {

}

Grid::Grid(int width, int height, int blockSize_) : blockSize(blockSize_) {
    nBlocksX = (width + blockSize) / blockSize - 1; // ceil int division
    nBlocksY = (height + blockSize) / blockSize - 1; // ceil int division
    int numBlocks = nBlocksX * nBlocksY;
    blocks.reserve(numBlocks);
    for (auto i = 0; i < numBlocks; i++) {
        blocks.emplace_back();
    }
}