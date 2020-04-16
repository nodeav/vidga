#include "cudaCircles.cuh"
#include "../classes/shapes/circle.h"
//#include "opencv2/highgui.hpp"

//#include "opencv2/core/mat.hpp"
namespace vidga {
    namespace cuda {
        __device__ constexpr unsigned colRow2idx(unsigned col, unsigned row, unsigned sideLength) {
            return row * sideLength + col;
        }

        __device__ __always_inline void blendColors(float3 *pixel, float4 color, float modifier) {
            float circleModifier = color.w * modifier;
            float canvasModifier = 1 - circleModifier;
            pixel->x = min(color.x * circleModifier + pixel->x * canvasModifier, 1.f);
            pixel->y = min(color.y * circleModifier + pixel->y * canvasModifier, 1.f);
            pixel->z = min(color.z * circleModifier + pixel->z * canvasModifier, 1.f);
        }

        __global__  void genSmoothCircleMap(float *buffer, unsigned radius) {
            const unsigned int strideX = blockDim.x * gridDim.x;
            const unsigned int strideY = blockDim.y * gridDim.y;
            const unsigned int initialX = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int initialY = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned short x = radius - 1, y = radius - 1;
            unsigned short sideLength = radius * 2 - 1;
            for (unsigned row = initialX; row < sideLength; row += strideX) {
                int yValueSquared = (row - y) * (row - y);
                for (unsigned col = initialY; col < sideLength; col += strideY) {
                    float distance = sqrtf(yValueSquared + (col - x) * (col - x));
                    unsigned idx = colRow2idx(col, row, sideLength);
                    float diff = distance - radius + 1;
                    if (diff < 0) {
                        buffer[idx] = 1.f;
                    } else if (diff < 1) {
                        buffer[idx] = 1.f - diff;
                    }

                }
            }
        }

        __global__ void
        drawUsingMap(float3 *buffer, unsigned width, unsigned height, const float *map, circle c) {
            const unsigned int strideX = blockDim.x * gridDim.x;
            const unsigned int strideY = blockDim.y * gridDim.y;
            const unsigned int posX = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int posY = blockIdx.y * blockDim.y + threadIdx.y;

            unsigned left = max(0, c.center.x - c.radius + 1);
            unsigned top = max(0, c.center.y - c.radius + 1);
            unsigned right = min(width - 1, c.center.x + c.radius);
            unsigned bottom = min(height - 1, c.center.y + c.radius);

            for (unsigned col = top + posY; col <= bottom; col += strideY) {
                for (unsigned row = left + posX; row <= right; row += strideX) {
                    unsigned bufferIdx = colRow2idx(col, row, width);
                    float3 *pixel = &buffer[bufferIdx];

                    unsigned mapIdx = colRow2idx(col - top, row - left, c.radius * 2 - 1);
                    float modifier = map[mapIdx];

//            printf("using col = %u, posY = %u; row = %u, posX = %u; bufIdx = %u, mapIdx = %u\n",
//                   col, col - top, row, row - left, bufferIdx, mapIdx);
                    blendColors(pixel, c.color, modifier);
                }
            }
        }

        void
        drawUsingMapHostFn(float3 *buffer, unsigned width, unsigned height, const float *map, circle c) {
            dim3 threads(16, 16, 1);
            dim3 blocks(4, 4, 1);
            drawUsingMap<<<blocks, threads>>>(buffer, width, height, map, c);
        }

        void initCircleMaps(unsigned minRadius, unsigned maxRadius, float ***gpuBuffers) {
            unsigned numCircles = maxRadius - minRadius + 1;
            *gpuBuffers = static_cast<float **>(malloc(numCircles * sizeof(float *)));
            for (auto i = minRadius; i <= maxRadius; i++) {
                auto idx = i - minRadius;
                auto sideLength = 2 * i - 1;
                auto winPixels = sideLength * sideLength;
                unsigned memToAlloc = winPixels * sizeof(float);
                float **circleBuf = &(*gpuBuffers)[idx];
                cudaMalloc(circleBuf, memToAlloc);
                genSmoothCircleMap<<<32, 32>>>(*circleBuf, i);
            }
            cudaDeviceSynchronize();
        }

        void setGpuMatTo(float3 *mat, unsigned width, unsigned height, float val) {
            auto size = width * height * sizeof(float) * 3;
            cudaMemset(mat, val, size);
        }

        float3 *getWhiteGpuMat(unsigned width, unsigned height) {
            auto size = width * height * sizeof(float) * 3;
            float3 *ret;
            cudaMalloc(&ret, size);
            setGpuMatTo(ret, width, height, 1.f);
            return ret;
        }
    }
}