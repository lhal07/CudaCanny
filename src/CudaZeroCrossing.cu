///CudaZeroCrossing.cu
/**
 * \author Luis Lourenço (2010)
 * \version 0.0.1
 * \since 26/10/10
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cutil.h>

#include "CudaZeroCrossing.h"

#define THREADS_PER_BLOCK 256


/// allocate texture variables
texture<float, 1, cudaReadModeElementType> texRef;

__global__ void kernel_zerocrossing(float* image, int3 size){
///this is the kernel to calculate the zero-crossing of the image. It is
///implemmented using texture fetching. The dependence of inter-block data makes
///the use of shared memory hard-boiled.

  float  pixel;
  float4 cross;
  float res = 0;

  int pixIdx = blockDim.x * blockIdx.x + threadIdx.x;

  ///output pixel index
  int2 pos;
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  if ((pos.x>0) && (pos.x<((size.x-1))) && (pos.y>0) && (pos.y<((size.y-1)))){

    pixel = tex1Dfetch(texRef,pixIdx);
    cross.x = tex1Dfetch(texRef,pixIdx-1);
    cross.y = tex1Dfetch(texRef,pixIdx-size.x);
    cross.z = tex1Dfetch(texRef,pixIdx+1);
    cross.w = tex1Dfetch(texRef,pixIdx+size.x);

    res = (((pixel*cross.x)<=0) *\
            (fabsf(pixel) < fabsf(cross.x)));

    res = res || ((((pixel*cross.y)<=0)) *\
            (fabsf(pixel) < fabsf(cross.y)));
    
    res = res || ((((pixel*cross.z)<=0)) *\
            (fabsf(pixel) <= fabsf(cross.z)));

    res = res || ((((pixel*cross.w)<=0)) *\
            (fabsf(pixel) <= fabsf(cross.w)));

  }

  image[pixIdx] = res;
  
}

float* cudaZeroCrossing(float *d_input, int width, int height){

  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;

  int blocksPerGrid = ((size.z) + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK;
  dim3 DimBlock(THREADS_PER_BLOCK,1,1);
  dim3 DimGrid(blocksPerGrid,1,1);

  float *d_img;
  cudaMalloc((void**) &d_img, (size.z*sizeof(float)));
  CUT_CHECK_ERROR("Memory image creation failed");

  /// bind a texture to the CUDA array
  cudaBindTexture (NULL, texRef, d_input);
  CUT_CHECK_ERROR("Texture bind failed");

  /// host side settable texture attributes
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;
  
  kernel_zerocrossing<<<DimGrid,DimBlock>>>(d_img, size);
  
  ///free allocated memory
  cudaUnbindTexture(texRef);
  CUT_CHECK_ERROR("Memory image free failed");
 
  return(d_img);

}
