
#include "cudaCircles.cuh"
#include "../classes/shapes/circle.h"
//#include "opencv2/highgui.hpp"

//#include "opencv2/core/mat.hpp"

#define gpu_check(e)    \
if (e != cudaSuccess) { \
    printf("cuda error - %d on %s:%d\n", e, __FILE__, __LINE__); \
    }

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
            int x = radius, y = radius;
            unsigned sideLength = radius * 2 + 1;
            for (unsigned row = initialX; row < sideLength; row += strideX) {
                int xValSq = (row - x) * (row - x);
                for (unsigned col = initialY; col < sideLength; col += strideY) {
                    int yValSq = (col - y) * (col - y);
                    float distance = sqrtf(xValSq + yValSq);
                    unsigned idx = colRow2idx(col, row, sideLength);
                    float diff = distance - radius;
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


            __shared__ unsigned mapShiftX, mapShiftY, left, top;
            if (c.center.x < c.radius) {
                left = 0;
                mapShiftX = c.radius - c.center.x;
            } else {
                left = c.center.x - c.radius;
                mapShiftX = 0;
            }

            if (c.center.y < c.radius) {
                top = 0;
                mapShiftY = c.radius - c.center.y;
            } else {
                top = c.center.y - c.radius;
                mapShiftY = 0;
            }

            unsigned right = min(width - 1, c.center.x + c.radius);
            unsigned bottom = min(height - 1, c.center.y + c.radius);

            for (unsigned col = top + posY; col <= bottom; col += strideY) {
                for (unsigned row = left + posX; row <= right; row += strideX) {
                    unsigned bufferIdx = colRow2idx(col, row, width);
                    float3 *pixel = &buffer[bufferIdx];

                    unsigned mapIdx = colRow2idx(col - top + mapShiftY, row - left + mapShiftX, c.radius * 2 + 1);
                    float modifier = map[mapIdx];
                    blendColors(pixel, c.color, modifier);
                }
            }
        }

/*
        __global__ void
        drawManyUsingMap(float3 *buffer, unsigned width, unsigned height, float **maps, unsigned mapsOffset,
                         circle *circles, unsigned nCircles) {
            const unsigned int strideX = blockDim.x * gridDim.x;
            const unsigned int strideY = blockDim.y * gridDim.y;
            const unsigned int posX = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int posY = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned int posZ = blockIdx.z * blockDim.z + threadIdx.z;

            unsigned mapShiftX, mapShiftY, left, top;
            circle *c;
            float *map;
            for (unsigned i = posZ; i < nCircles; i++) {
                c = &circles[i];
                map = maps[c->radius - mapsOffset];

                if (c->center.x < c->radius) {
                    left = 0;
                    mapShiftX = c->radius - c->center.x;
                } else {
                    left = c->center.x - c->radius;
                    mapShiftX = 0;
                }

                if (c->center.y < c->radius) {
                    top = 0;
                    mapShiftY = c->radius - c->center.y;
                } else {
                    top = c->center.y - c->radius;
                    mapShiftY = 0;
                }

                unsigned right = min(width - 1, c->center.x + c->radius);
                unsigned bottom = min(height - 1, c->center.y + c->radius);

                for (unsigned col = top + posY; col <= bottom; col += strideY) {
                    for (unsigned row = left + posX; row <= right; row += strideX) {
                        unsigned bufferIdx = colRow2idx(col, row, width);
                        float3 *pixel = &buffer[bufferIdx];

                        unsigned mapIdx = colRow2idx(col - top + mapShiftY, row - left + mapShiftX, c->radius * 2 + 1);
                        float modifier = map[mapIdx];
                        blendColors(pixel, c->color, modifier);
                    }
                }
            }
        }
*/

        void
        drawUsingMapHostFn(float3 *buffer, unsigned width, unsigned height, const float *map, circle c) {
            dim3 threads(16, 16, 1);
            dim3 blocks(4, 4, 1);
            drawUsingMap<<<blocks, threads>>>(buffer, width, height, map, std::move(c));
        }

 /*       void
        drawManyUsingMapHostFn(float3 *buffer, unsigned width, unsigned height, float **maps, unsigned mapsOffset,
                               const circle *circles, unsigned nCircles) {
            dim3 threads(16, 16, 1);
            dim3 blocks(1, 1, 1);
            circle* circlesGpu;
            size_t byteSize = sizeof(circle) * nCircles;
            auto e = cudaMalloc(&circlesGpu, byteSize);
            auto e2 = cudaMemcpy((void *) circlesGpu, circles, byteSize, cudaMemcpyHostToDevice);
            printf("e %d, e2 %d\n", e, e2);
            drawManyUsingMap<<<blocks, threads>>>(buffer, width, height, maps, mapsOffset, circlesGpu, nCircles);
            gpu_check(cudaGetLastError());
        }*/

        void initCircleMaps(unsigned minRadius, unsigned maxRadius, float ***gpuBuffers) {
            unsigned numCircles = maxRadius - minRadius + 1;
            *gpuBuffers = static_cast<float **>(malloc(numCircles * sizeof(float *)));
            for (auto i = minRadius; i <= maxRadius; i++) {
                auto idx = i - minRadius;
                auto winSideLength = 2 * i + 1;
                auto winPixels = winSideLength * winSideLength;
                unsigned memToAlloc = winPixels * sizeof(float);
                float **circleBuf = &(*gpuBuffers)[idx];
                cudaMalloc(circleBuf, memToAlloc);
                genSmoothCircleMap<<<32, 32>>>(*circleBuf, i);
            }
            gpu_check(cudaDeviceSynchronize());
        }

        void setGpuMatTo(float3 *mat, unsigned width, unsigned height, float val) {
            auto size = width * height * sizeof(float) * 3;
            cudaMemset(mat, val, size);
        }

        float3 *getWhiteGpuMat(unsigned width, unsigned height) {
            auto size = width * height * sizeof(float) * 3;
            float3 *ret;
            gpu_check(cudaMalloc(&ret, size));
            setGpuMatTo(ret, width, height, 1.f);
            return ret;
        }
    }
}