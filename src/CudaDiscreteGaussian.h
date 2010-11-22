///CudaDiscreteGaussian.h
/**
 * \author Luis Lourenço (2010)
 * \version 3.2.0
 * \since 20/09/10
 */


extern "C"
float* cuda1DGaussianOperator(dim3 DimGrid, dim3 DimBlock, unsigned int width, float gaussianVariance);
