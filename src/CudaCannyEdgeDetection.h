///canny.h
/**
 * \author Luis Lourenço (2010)
 * \version 2.2.1
 * \since 20/05/10
 */


extern "C"
float* cudaCanny(const float *image, int width, int height, const float gaussianVariance, const unsigned int maxKernelWidth, const unsigned int t1, const unsigned int t2);

