///Cuda2DSeparableConvolution.h
/**
 * \author Luis Lourenço (2010)
 * \version 4.0.0
 * \since 20/09/10
 */



extern "C"
float* cuda2DSeparableConvolution(const float *d_img, int width, int height, const float *d_kernelH, int sizeH, const float *d_kernelV, int sizeV);
