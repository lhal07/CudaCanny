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
texture<float, 1, cudaReadModeElementType> mag_texRef;
texture<short2, 1, cudaReadModeElementType> dir_texRef;


__global__ void kernel_2DSobel(float *Magnitude, short2* Direction, int3 size){
/// This is an less elaborated kernel version that calculate the sobel-x, sobel-y, 
/// then uses the calculated values to return to memory just the needed information.
/// That is, the magnitude and direction of the edge.
/// It ignores completely the borders of the image becausa the border's pixels
/// doesn't have all neighbors

  float2 g_i;
  g_i.x = g_i.y = 0;
  int theta;
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

  /// Transform radian to degrees (multiply for 180/pi) to facilitate the
  /// aproximation on an integer variable.
  /// And sums 90 degrees to rotate the trigonometrical circle and eliminate the
  /// negative values.
  theta = (g_i.x != 0)*(int)(atanf(__fdividef(g_i.y,g_i.x))*__fdividef(180,M_PI)) + 90;

  /// Put the values between 158 and 180 degrees on the [0,22] interval.
  /// This way, all horizontal pixels will be in the interval of [0,22].
  if (theta > 157) theta -= 158;

  /// This calculation does this:
  //  direction
  //  interval  -> theta
  //  [0,22]    ->   0
  //  [23,67]   ->   1
  //  [68,112]  ->   2
  //  [113,157] ->   3
  theta = ceilf(__fdividef(theta-22,45));

  /// The pixel will compare it's value with it's perpendicular(90 degrees) 
  /// neighbor's here it's used short2 because it is 32bit lenght (this is 
  /// good to the store coalescence).
  /// theta -> ( x, y)
  ///   0   -> ( 0,-1)
  ///   1   -> (-1,-1)
  ///   2   -> ( 1, 0)
  ///   3   -> ( 1,-1)
  Direction[pixIdx] = make_short2(1-(theta == 0)-((theta == 1)<<1),(theta == 2)-1);

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

  unsigned int timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  ///< Start timer

  /// Allocate output memory to image data
  cudaMalloc((void**) &d_gradient->Strenght, size.z*sizeof(float));
  cudaMalloc((void**) &d_gradient->Direction, size.z*sizeof(short2));

  /// bind a texture to the CUDA array
  cudaBindTexture (NULL ,texRef, d_img);
  CUT_CHECK_ERROR("Texture bind failed");

  /// host side settable texture attributes
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;

  kernel_2DSobel<<<DimGrid,DimBlock>>>(d_gradient->Strenght, d_gradient->Direction, size);

  cudaThreadSynchronize();
  cutStopTimer( timer );  ///< Stop timer
  printf("Sobel time = %f ms\n",cutGetTimerValue( timer ));

  /// unbind texture reference
  cudaUnbindTexture(texRef);
  CUT_CHECK_ERROR("Memory image free failed");

  return(d_gradient);
}

