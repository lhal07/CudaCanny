///Cuda2DSeparableConvolution.cu
/**
 * \author Luis Lourenço (2010)
 * \version 4.0.0
 * \since 15/09/10
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cutil.h>

#include "Cuda2DSeparableConvolution.h"


/// allocate texture variables
texture<float, 1, cudaReadModeElementType> texRef;
texture<float, 1, cudaReadModeElementType> convTexRef;


__global__ void kernel_1DConvolutionH_texture(float *output, int3 size, short halfkernelsize){
///this version uses the texture memory to store the convolution kernel and the
///image data

  float sum = 0;
  int2 pos;

  extern __shared__ float s_mask[];

  ///pixel index of this thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;

  ///output pixel index
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  if(threadIdx.x<((halfkernelsize<<1)+1)) s_mask[threadIdx.x] = tex1Dfetch(convTexRef,threadIdx.x);
  
  __syncthreads();

  for(int k=-halfkernelsize;k<(halfkernelsize+1);k++){
    sum += (tex1Dfetch(texRef, pixIdx + k * (((pos.x+k)>=0)*((pos.x+k)<size.x))) * s_mask[k+halfkernelsize]);
  }

  output[pixIdx] = sum;
}

__global__ void kernel_1DConvolutionV_texture(float *output, int3 size, short halfkernelsize){
///this version uses the texture memory to store the convolution kernel and the
///image data

  float sum = 0;
  int2 pos;

  extern __shared__ float s_mask[];

  ///pixel index of this thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;

  ///output pixel index
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  if(threadIdx.x<((halfkernelsize<<1)+1)) s_mask[threadIdx.x] = tex1Dfetch(convTexRef,threadIdx.x);
  
  __syncthreads();

  for(int k=-halfkernelsize;k<(halfkernelsize+1);k++){
    sum += (tex1Dfetch(texRef, pixIdx + (size.x*k) * (((pos.y+k)>=0)*((pos.y+k<size.y)))) * s_mask[k+halfkernelsize]);
  }

  output[pixIdx] = sum;
}

float* cuda2DSeparableConvolution(dim3 DimGrid, dim3 DimBlock, const float *d_img, int width, int height, const float *d_kernelH, int sizeH, const float *d_kernelV, int sizeV){

  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;

  short halfKernelWidthH = sizeH >> 1;
  int kernelSizeH = sizeH*sizeof(float);
  short halfKernelWidthV = sizeV >> 1;
  int kernelSizeV = sizeV*sizeof(float);

  float *d_output;
  cudaMalloc((void**) &d_output, size.z*sizeof(float));

  /// Allocate temporary memory to image data
  float *d_tmpbuffer;
  cudaMalloc((void**) &d_tmpbuffer, size.z*sizeof(float));

  /// Bind a texture to the CUDA array
  cudaBindTexture (NULL, convTexRef, d_kernelH);
  CUT_CHECK_ERROR("Texture bind failed");

  /// Host side settable texture attributes
  convTexRef.normalized = false;
  convTexRef.filterMode = cudaFilterModePoint;

  /// Bind a texture to the CUDA array
  cudaBindTexture (NULL, texRef, d_img);
  CUT_CHECK_ERROR("Texture bind failed");

  /// Host side settable texture attributes
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;

  kernel_1DConvolutionH_texture<<<DimGrid,DimBlock,kernelSizeH>>>(d_tmpbuffer,size,halfKernelWidthH);

  /// Bind temporary data texture
  cudaUnbindTexture(texRef);
  cudaUnbindTexture(convTexRef);
  cudaBindTexture (NULL ,texRef, d_tmpbuffer);
  cudaBindTexture (NULL, convTexRef, d_kernelV);

  kernel_1DConvolutionV_texture<<<DimGrid,DimBlock,kernelSizeV>>>(d_output,size,halfKernelWidthV);

  /// Free allocated memory
  cudaFree(d_tmpbuffer);
  cudaUnbindTexture(texRef);
  cudaUnbindTexture(convTexRef);
  CUT_CHECK_ERROR("Memory image free failed");

  return(d_output);
}

