
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

        __device__ void blendColorsNoCutoff(float3 *pixel, float4 color, float modifier) {
            float circleModifier = color.w * modifier;
            float canvasModifier = 1 - circleModifier;
//            printf("pixel is %p and color is %f, %f, %f, %f\n", pixel, color.x, color.y, color.z, color.w);
            pixel->x = color.x * circleModifier + pixel->x * canvasModifier;
            pixel->y = color.y * circleModifier + pixel->y * canvasModifier;
            pixel->z = color.z * circleModifier + pixel->z * canvasModifier;
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

            for (unsigned row = top + posY; row <= bottom; row += strideY) {
                for (unsigned col = left + posX; col <= right; col += strideX) {
                    unsigned bufferIdx = colRow2idx(col, row, width);
                    float3 *pixel = &buffer[bufferIdx];

                    unsigned mapIdx = colRow2idx(col - left + mapShiftX, row - top + mapShiftY, c.radius * 2 + 1);
                    float modifier = map[mapIdx];
                    blendColorsNoCutoff(pixel, c.color, modifier);
                }
            }
        }

        __always_inline __device__ bool isInCircleBBox(const circle &c, int x, int y) {
            int centerX = static_cast<int>(c.center.x);
            int centerY = static_cast<int>(c.center.y);
            int radSigned = static_cast<int>(c.radius);
            auto inX = centerX + radSigned >= x && centerX - radSigned <= x;
            auto inY = centerY + radSigned >= y && centerY - radSigned <= y;
            return inX && inY;
        }

        __always_inline __device__ void
        drawCirclePixelUsingMap(float3 *pixel, const float *map, const circle &c, unsigned x, unsigned y) {
            auto bboxStartX = c.center.x - c.radius;
            auto bboxStartY = c.center.y - c.radius;
            auto offsetX = x - bboxStartX;
            auto offsetY = y - bboxStartY;
            unsigned mapIdx = colRow2idx(offsetX, offsetY, c.radius * 2 + 1);
//            printf("\tusing mapIdx %u - offset{%d, %d}\n", mapIdx, offsetX, offsetY);
            float modifier = map[mapIdx];
//            printf("\tmodifier is %f\n", modifier);
            blendColorsNoCutoff(pixel, c.color, modifier);
        }

        __device__ unsigned getOffsetByRadius(unsigned radius) {
            return (((radius + 1) * (2 * radius + 1) * (2 * radius + 3) / 3) | 0);
        }

        __global__ void calcDiffUsingMap(float3 *buffer, float3 *orig, unsigned width, unsigned height, float *map,
                                         const circle *circles, int nCircles, unsigned mapOffset) {
            const unsigned int strideX = blockDim.x * gridDim.x;
            const unsigned int strideY = blockDim.y * gridDim.y;
            const unsigned int posX = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int posY = blockIdx.y * blockDim.y + threadIdx.y;
            extern __shared__ circle sh_circles[];

            if (threadIdx.y == 0) {
                for (auto i = threadIdx.x; i < nCircles; i += blockDim.x) {
                    sh_circles[i] = circles[i];
                }
            }

            __syncthreads();

            auto from = getOffsetByRadius(mapOffset - 1);
            for (unsigned row = posY; row < height; row += strideY) {
                for (unsigned col = posX; col < width; col += strideX) {
                    auto idx = colRow2idx(col, row, width);
                    float3 pixel = {0.f, 0.f, 0.f};
                    for (unsigned i = 0; i < nCircles; i++) {
                        const auto &circle = sh_circles[i];
                        if (isInCircleBBox(circle, static_cast<int>(col), static_cast<int>(row))) {
                            auto to = getOffsetByRadius(circle.radius - 1);
                            /*printf("pixel {%d, %d} is in circle{r:%d, c{%d, %d}}. using map #%d with offset %d, to %d, from %d\n",
                                   col, row, circle.radius, circle.center.x, circle.center.y,
                                   circle.radius - mapOffset, to - from, to, from);*/
                            drawCirclePixelUsingMap(&pixel, map + to - from, circle, col, row);
                        }
                    }
                    auto target = orig[idx];
                    pixel.x = abs(pixel.x - target.x);
                    pixel.y = abs(pixel.y - target.y);
                    pixel.z = abs(pixel.z - target.z);
                    buffer[idx] = pixel;
                }
            }

        };

        void
        drawUsingMapHostFn(float3 *buffer, unsigned width, unsigned height, const float *map, circle c) {
            dim3 threads(16, 16, 1);
            dim3 blocks(4, 4, 1);
            drawUsingMap<<<blocks, threads>>>(buffer, width, height, map, std::move(c));
        }

        void initCircleMaps1D(unsigned minRadius, unsigned maxRadius, float **gpuBuffers);

        void
        calcDiffUsingMapHostFn(float3 *buffer, float3 *orig, unsigned width, unsigned height, float *map,
                               const std::vector<circle> &circles, unsigned mapOffset, cudaStream_t cudaStream) {
            dim3 threads(8, 8, 1);
            dim3 blocks(16, 16, 1);

            circle *d_circles;
            auto circSize = circles.size() * sizeof(circles[0]);
//            printf("using %zu circles, each of size %lu\n", circles.size(), sizeof(circles[0]));
            gpu_check(cudaMalloc(&d_circles, circSize));
            gpu_check(cudaMemcpyAsync(d_circles, circles.data(), circSize, cudaMemcpyHostToDevice, cudaStream));

            calcDiffUsingMap<<<blocks, threads, circSize, cudaStream>>>(buffer, orig, width, height, map,
                                                                        d_circles,
                                                                        circles.size(),
                                                                        mapOffset);

            cudaFree(d_circles);
        }

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

        void initCircleMaps1D(unsigned minRadius, unsigned maxRadius, float **gpuBuffers) {
            size_t nElems = 0;
            for (auto i = minRadius; i <= maxRadius; i++) {
                auto d = 2 * i + 1;
//                printf("initCircleMaps1D: allocating %d more elements for i %d\n", d * d, i);
                nElems += d * d;
            }
            cudaMalloc(gpuBuffers, nElems * sizeof(float));

            auto offset = 0;
            for (auto i = minRadius; i <= maxRadius; i++) {
                auto winSideLength = 2 * i + 1;
                auto winPixels = winSideLength * winSideLength;
//                printf("using offset: %d\n", offset);
                genSmoothCircleMap<<<32, 32>>>((*gpuBuffers) + offset, i);
                offset += winPixels;
            }
            gpu_check(cudaDeviceSynchronize());
        }


        void setGpuMatTo(float3 *mat, unsigned width, unsigned height, int val) {
            auto size = width * height * sizeof(float) * 3;
            cudaMemset(mat, val, size);
        }

        float3 *getWhiteGpuMat(unsigned width, unsigned height) {
            auto size = width * height * sizeof(float) * 3;
            float3 *ret;
            gpu_check(cudaMalloc(&ret, size));
            setGpuMatTo(ret, width, height, 0);
            return ret;
        }
    }
}