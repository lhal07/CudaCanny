///CannySobelEdgeDetection.cu
/**
 * \author Luis Lourenço (2010)
 * \version 3.0.0
 * \since 15/09/10
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cutil.h>

#include "CudaSobelEdgeDetection.h"


/// allocate texture variables
texture<float, 1, cudaReadModeElementType> texRef;


__global__ void kernel_2DSobel(float *Magnitude, float *Direction, int3 size){
/// This is an less elaborated kernel version that calculate the sobel-x, sobel-y, 
/// then uses the calculated values to return to memory just the needed information.
/// That is, the magnitude and direction of the edge.
/// It ignores completely the borders of the image becausa the border's pixels
/// doesn't have all neighbors

  float2 g_i;
  g_i.x = g_i.y = 0;
  float4 diagonal;
  float4 cross;

  /// Pixel index of this thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;

  /// Output pixel index
  int2 pos;
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  ///Ignore the image borders
  if ((pos.x) && ((size.x-1)-pos.x) && (pos.y) && ((size.y-1)-pos.y)){

    /// Stores the neighbors of the pixel on variables, because they will be
    /// readen more than one time.
    diagonal.x = tex1Dfetch(texRef,(pixIdx-size.x-1));
    diagonal.y = tex1Dfetch(texRef,(pixIdx-size.x+1));
    diagonal.z = tex1Dfetch(texRef,(pixIdx+size.x-1));
    diagonal.w = tex1Dfetch(texRef,(pixIdx+size.x+1));
    cross.x = tex1Dfetch(texRef,(pixIdx-size.x));
    cross.y = tex1Dfetch(texRef,(pixIdx+size.x));
    cross.z = tex1Dfetch(texRef,(pixIdx-1));
    cross.w = tex1Dfetch(texRef,(pixIdx+1));

    /// SobelX
    g_i.x -= (diagonal.x+cross.z+cross.z+diagonal.z);
    g_i.x += (diagonal.y+cross.w+cross.w+diagonal.w);
    
    /// SobelY
    g_i.y -= (diagonal.z+cross.y+cross.y+diagonal.w);
    g_i.y += (diagonal.x+cross.x+cross.x+diagonal.y);
    
  }

  Magnitude[pixIdx] = sqrtf((g_i.x*g_i.x) + (g_i.y*g_i.y));

  Direction[pixIdx] = (g_i.x != 0)*(atanf(__fdividef(g_i.y,g_i.x)));

}

extern "C"
Tgrad* cudaSobel(Tgrad *d_gradient, const float *d_img, int width, int height){

  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;

  int threadsPerBlock = 256;
  int blocksPerGrid = ((size.z) + threadsPerBlock -1) >> 8;
  dim3 DimBlock(threadsPerBlock,1,1);
  dim3 DimGrid(blocksPerGrid,1,1);

  /// Allocate output memory to image data
  cudaMalloc((void**) &d_gradient->Magnitude, size.z*sizeof(float));
  cudaMalloc((void**) &d_gradient->Direction, size.z*sizeof(float));

  /// bind a texture to the CUDA array
  cudaBindTexture (NULL ,texRef, d_img);
  CUT_CHECK_ERROR("Texture bind failed");

  /// host side settable texture attributes
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;

  kernel_2DSobel<<<DimGrid,DimBlock>>>(d_gradient->Magnitude, d_gradient->Direction, size);

  /// unbind texture reference
  cudaUnbindTexture(texRef);
  CUT_CHECK_ERROR("Memory image free failed");

  return(d_gradient);
}

