///CudaDiscreteGaussian.h
/**
 * \author Luis Lourenço (2010)
 * \version 3.0.0
 * \since 20/09/10
 */



extern "C"
float* cudaDiscreteGaussian2D(const float *d_img, int width, int height, float gaussianVariance, unsigned int maxKernelWidth);
