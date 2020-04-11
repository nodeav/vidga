#include "cudaCircles.cuh"

__global__ void genSmoothCircleMap(float *buffer, unsigned short radius) {
    const unsigned int strideX = blockDim.x * gridDim.x;
    const unsigned int strideY = blockDim.y * gridDim.y;
    const unsigned int initialX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int initialY = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned short x = radius, y = radius;
    unsigned short sideLength = radius * 2;

    for (unsigned row = initialX; row < sideLength; row += strideX) {
        int yValueSquared = (row - y) * (row - y);
        for (unsigned col = initialY; col < sideLength; col += strideY) {
            float distance = sqrtf(yValueSquared + (col - x) * (col - x));
            unsigned idx = row * sideLength + col;
            float diff = distance - radius + 1;
            if (diff < 0) {
                buffer[idx] = 1;
            } else if (diff < 1) {
                buffer[idx] = 1 - diff;
            }
        }
    }
}

void initCircleMaps(unsigned minRadius, unsigned maxRadius, float **gpuBuffers) {
    unsigned numCircles = maxRadius - minRadius + 1;
    cudaMalloc(&gpuBuffers, numCircles * sizeof(float*));
    for (auto i = minRadius; i <= maxRadius; i++) {
        auto idx = i - minRadius;
        unsigned memToAlloc = 4 * i * i * sizeof(float);
        float* circleBuf = gpuBuffers[idx];
        cudaMalloc(&circleBuf, memToAlloc);
        genSmoothCircleMap<<<32, 32>>>(circleBuf, i);
    }

    cudaDeviceSynchronize();
}