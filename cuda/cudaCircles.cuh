namespace vidga {
    namespace cuda {
        void initCircleMaps(unsigned minRadius, unsigned maxRadius, float **gpuBuffers);
        void drawUsingMapHostFn(float3 *buffer, unsigned width, unsigned height, const float *map, circle c);
        class cudaCircles {
        private:
        };
    }
}