///CudaDiscreteGaussian.cu
/**
 * \author Luis Lourenço (2010)
 * \version 3.1.0
 * \since 15/09/10
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cutil.h>

#include "CudaDiscreteGaussian.h"

/// allocate texture variables
texture<float, 1, cudaReadModeElementType> texRef;
texture<float, 1, cudaReadModeElementType> gaussTexRef;



__global__ void calculateGaussianKernel(float *gaussKernel, const float sigma, int halfKernelWidth){

  /// pixel index of this thread
  /// this makes the normal curve
  int i = threadIdx.x - halfKernelWidth;
  extern __shared__ float s_gaussKernel[];
  __shared__ float sum;
  
  /// this kernel must allocate 'kernelWidth' threads
  s_gaussKernel[threadIdx.x] = (__fdividef(1,(sqrtf(2*M_PI*sigma))))*expf((-1)*(__fdividef((i*i),(2*sigma*sigma))));

  __syncthreads();

  /// Thread 0 sum all the gassian kernel array
  // This is not so bad because the array is always short
  if (!threadIdx.x) {
    int th;
    sum = 0;
    for(th = 0; th<blockDim.x; th++) sum += s_gaussKernel[th];
  }

  __syncthreads();

  gaussKernel[threadIdx.x] = s_gaussKernel[threadIdx.x]/sum;

}

float* cuda1DGaussianOperator(dim3 DimGrid, dim3 DimBlock, unsigned int width, float gaussianVariance){

  /// The Gaussian Kernel Width must be odd
  if (width < 1) width = 1;
  if (width%2 == 0) width--;
  short halfWidth = width >> 1;

  int kernelSize = width*sizeof(float);

  float *cudaGaussKernel;
  cudaMalloc((void**)&cudaGaussKernel,kernelSize);

  /// Calculate gaussian kernel
  calculateGaussianKernel<<<DimGrid,DimBlock,kernelSize>>>(cudaGaussKernel, gaussianVariance, halfWidth);
  
  return(cudaGaussKernel);
}

