#include "../classes/shapes/circle.h"

namespace vidga {
    namespace cuda {
        void initCircleMaps(unsigned minRadius, unsigned maxRadius, float ***gpuBuffers);
        void initCircleMaps1D(unsigned minRadius, unsigned maxRadius, float **gpuBuffers);
        void drawUsingMapHostFn(float3 *buffer, unsigned width, unsigned height, const float *map, circle c);

        /*void
        drawManyUsingMapHostFn(float3 *buffer, unsigned width, unsigned height, float **maps, unsigned mapsOffset,
                               const circle *circles, unsigned nCircles);
*/
        void setGpuMatTo(float3 *mat, unsigned width, unsigned height, int val);

        float3 *getWhiteGpuMat(unsigned width, unsigned height);

        void calcDiffUsingMapHostFn(float3 *buffer, float3 *orig, unsigned width, unsigned height, float *map,
                                    const std::vector<circle> &circles, unsigned mapOffset);

        class cudaCircles {
        private:
        };
    }
}