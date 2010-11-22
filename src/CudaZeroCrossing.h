///CudaZeroCrossing.h
/**
 * \author Luis Lourenco (2010)
 * \version 0.0.1
 * \since 26/10/10
 */

extern "C"
float* cudaZeroCrossing(dim3 DimGrid, dim3 DimBlock, float *d_input, int width, int height);

