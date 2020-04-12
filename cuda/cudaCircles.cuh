#include "../classes/shapes/circle.h"

namespace vidga {
    namespace cuda {
        void initCircleMaps(unsigned minRadius, unsigned maxRadius, float ***gpuBuffers);
        void drawUsingMapHostFn(float3 *buffer, unsigned width, unsigned height, const float *map, circle c);
        void setGpuMatTo(float3* mat, unsigned width, unsigned height, float val);
        float3 *getZeroedGpuMat(unsigned width, unsigned height);

        class cudaCircles {
        private:
        };
    }
}