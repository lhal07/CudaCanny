///CudaDiscreteGaussian.cu
/**
 * \author Luis Lourenço (2010)
 * \version 3.0.0
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



__global__ void calculateGaussianKernel(float *gaussKernel, const float sigma, int kernelWidth){

  ///pixel index of this thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int i = pixIdx - (kernelWidth>>1);
  float smaller;
  extern __shared__ float s_gaussKernel[];
  
  ///this kernel must allocate 'kernelWidth' threads
  s_gaussKernel[threadIdx.x] = (__fdividef(1,(sqrtf(2*M_PI*sigma))))*expf((-1)*(__fdividef((i*i),(2*sigma*sigma))));

  __syncthreads();

  smaller = s_gaussKernel[0];

  gaussKernel[pixIdx] = s_gaussKernel[threadIdx.x]/smaller;

}

__global__ void kernel_1DConvolutionH_texture(float *output, int3 size, short halfkernelsize){
///this version uses the texture memory to store the gaussian kernel and the
///image data

  float2 sum;
  int2 pos;

  extern __shared__ float s_gauss[];

  ///pixel index of this thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;

  ///output pixel index
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  if(threadIdx.x<((halfkernelsize<<1)+1)) s_gauss[threadIdx.x] = tex1Dfetch(gaussTexRef,threadIdx.x);
  
  sum.x = sum.y = 0;

  for(int k=-halfkernelsize;k<(halfkernelsize+1);k++){
    sum.x += (tex1Dfetch(texRef, pixIdx + k * (((pos.x+k)>=0)*((pos.x+k)<size.x))) * s_gauss[k+halfkernelsize]);
    sum.y += s_gauss[k+halfkernelsize];
  }

  output[pixIdx] = __fdividef(sum.x,sum.y);
}

__global__ void kernel_1DConvolutionV_texture(float *output, int3 size, short halfkernelsize){
///this version uses the texture memory to store the gaussian kernel and the
///image data

  float2 sum;
  int2 pos;

  extern __shared__ float s_gauss[];

  ///pixel index of this thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;

  ///output pixel index
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  if(threadIdx.x<((halfkernelsize<<1)+1)) s_gauss[threadIdx.x] = tex1Dfetch(gaussTexRef,threadIdx.x);
  
  sum.x = sum.y = 0;

  for(int k=-halfkernelsize;k<(halfkernelsize+1);k++){
    sum.x += (tex1Dfetch(texRef, pixIdx + (size.x*k) * (((pos.y+k)>=0)*((pos.y+k<size.y)))) * s_gauss[k+halfkernelsize]);
    sum.y += s_gauss[k+halfkernelsize];
  }

  output[pixIdx] = __fdividef(sum.x,sum.y);
}

extern "C"
float* cudaDiscreteGaussian2D(const float *d_img, int width, int height, float gaussianVariance, unsigned int maxKernelWidth){

  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;

  int threadsPerBlock = 256;
  int blocksPerGrid = ((size.z) + threadsPerBlock -1) >> 8;
  dim3 DimBlock(threadsPerBlock,1,1);
  dim3 DimGrid(blocksPerGrid,1,1);

  int kernelSize = maxKernelWidth*sizeof(float);

  unsigned int timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  /// Start timer

  /// The Gaussian Kernel Width must be odd
  if (maxKernelWidth < 1) maxKernelWidth = 1;
  if (maxKernelWidth%2 == 0) maxKernelWidth--;
  short halfkernelsize = maxKernelWidth >> 1;

  float *cudaGaussKernel;
  cudaMalloc((void**)&cudaGaussKernel,kernelSize);

  /// Calculate gaussian kernel
  calculateGaussianKernel<<<1,maxKernelWidth,kernelSize>>>(cudaGaussKernel, gaussianVariance, maxKernelWidth);

  /// Allocate output memory to image data
  float *d_output;
  cudaMalloc((void**) &d_output, size.z*sizeof(float));

  /// Allocate temporary memory to image data
  float *d_tmpbuffer;
  cudaMalloc((void**) &d_tmpbuffer, size.z*sizeof(float));

  /// Bind a texture to the CUDA array
  cudaBindTexture (NULL, gaussTexRef, cudaGaussKernel);
  CUT_CHECK_ERROR("Texture bind failed");

  /// Host side settable texture attributes
  gaussTexRef.normalized = false;
  gaussTexRef.filterMode = cudaFilterModePoint;

  /// Bind a texture to the CUDA array
  cudaBindTexture (NULL, texRef, d_img);
  CUT_CHECK_ERROR("Texture bind failed");

  /// Host side settable texture attributes
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;

  kernel_1DConvolutionH_texture<<<DimGrid,DimBlock,kernelSize>>>(d_tmpbuffer,size,halfkernelsize);

  /// Bind temporary data texture
  cudaUnbindTexture(texRef);
  cudaBindTexture (NULL ,texRef, d_tmpbuffer);

  kernel_1DConvolutionV_texture<<<DimGrid,DimBlock,kernelSize>>>(d_output,size,halfkernelsize);

  /// Free allocated memory
  cudaFree(d_tmpbuffer);
  cudaUnbindTexture(texRef);
  cudaFree(cudaGaussKernel);
  cudaUnbindTexture(gaussTexRef);
  CUT_CHECK_ERROR("Memory image free failed");

  cudaThreadSynchronize();
  cutStopTimer( timer );  /// Stop timer
  printf("Gaussian time = %f ms\n",cutGetTimerValue( timer ));

  return(d_output);
}

