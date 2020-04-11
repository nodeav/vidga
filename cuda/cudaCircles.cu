#include "cudaCircles.cuh"
#include "../classes/shapes/circle.h"
//#include "opencv2/highgui.hpp"

//#include "opencv2/core/mat.hpp"
namespace vidga {
    namespace cuda {
        constexpr unsigned colRow2idx(unsigned col, unsigned row, unsigned sideLength) {
            return row * sideLength + col;
        }

        __device__ __always_inline void blendColors(float3 *pixel, float4 color, float modifier) {
            float finalModifier = color.w * modifier;
            pixel->x = max(color.x * finalModifier + pixel->x, 1.f);
            pixel->y = max(color.y * finalModifier + pixel->y, 1.f);
            pixel->z = max(color.z * finalModifier + pixel->z, 1.f);
        }

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
                    unsigned idx = colRow2idx(col, row, sideLength);
                    float diff = distance - radius + 1;
                    if (diff < 0) {
                        buffer[idx] = 1;
                    } else if (diff < 1) {
                        buffer[idx] = 1 - diff;
                    }
                }
            }
        }

        __global__ void
        drawUsingMap(float3 *buffer, unsigned width, unsigned height, const float *map, circle c) {
            const unsigned int strideX = blockDim.x * gridDim.x;
            const unsigned int strideY = blockDim.y * gridDim.y;
            const unsigned int initialX = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int initialY = blockIdx.y * blockDim.y + threadIdx.y;

            unsigned short sideLength = c.radius * 2;
            unsigned maxX = min(width, c.center.x + c.radius);
            unsigned maxY = min(height, c.center.y + c.radius);
            for (unsigned row = initialX; row < maxX; row += strideX) {
                for (unsigned col = initialY; col < maxY; col += strideY) {
                    unsigned idx = colRow2idx(col, row, sideLength);
                    float3 *pixel = &buffer[idx];
                    float modifier = map[idx];
                    blendColors(pixel, c.color, modifier);
                }
            }
        }

        void
        drawUsingMapHostFn(float3 *buffer, unsigned width, unsigned height, const float *map, circle c) {
            dim3 threads(8, 8, 1);
            dim3 blocks(1, 1, 1);
            drawUsingMap<<<blocks, threads>>>(buffer, width, height, map, c);
        }

        void initCircleMaps(unsigned minRadius, unsigned maxRadius, float **gpuBuffers) {
            unsigned numCircles = maxRadius - minRadius + 1;
            gpuBuffers = static_cast<float **>(malloc(numCircles * sizeof(float *)));
            for (auto i = minRadius; i <= maxRadius; i++) {
                auto idx = i - minRadius;
                unsigned memToAlloc = 4 * i * i * sizeof(float *);
                float *circleBuf = gpuBuffers[idx];
                cudaMalloc(&circleBuf, memToAlloc);
                genSmoothCircleMap<<<32, 32>>>(circleBuf, i);
            }
            cudaDeviceSynchronize();

        }
    }
}