///canny.cu
/**
 * \author Luis Lourenço (2010)
 * \version 2.0.0
 * \since 20/05/10
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cutil.h>

#include "CudaCannyEdgeDetection.h"


/// allocate texture variables
texture<float, 1, cudaReadModeElementType> texRef;
texture<float, 1, cudaReadModeElementType> mag_texRef;
texture<float, 1, cudaReadModeElementType> dir_texRef;
texture<float, 1, cudaReadModeElementType> hysTexRef;


__global__ void nonMaximumSupression_texture(float* image, int3 size){
///this is the kernel to calculate the non-maximum supression of the image. It is
///implemmented using texture fetching. The dependence of inter-block data makes
///the use of shared memory hard-boiled.

  int pixIdx = blockDim.x * blockIdx.x + threadIdx.x;
  float mag = 0;
  short2 dir;

  ///output pixel index
  int2 pos;
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  if ((pos.x>0) && (pos.x<((size.x-1))) && (pos.y>0) && (pos.y<((size.y-1)))){

    /// Transform radian to degrees (multiply for 180/pi) to facilitate the
    /// aproximation on an integer variable.
    /// And sums 90 degrees to rotate the trigonometrical circle and eliminate the
    /// negative values.
    int theta = (tex1Dfetch(dir_texRef,pixIdx) * __fdividef(180,M_PI)) + 90;

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
    dir = make_short2(1-(theta == 0)-((theta == 1)<<1),(theta == 2)-1);


    mag = tex1Dfetch(mag_texRef,pixIdx);
    mag *= ((mag>=tex1Dfetch(mag_texRef,(pixIdx+(size.x*dir.y)+dir.x)))*(mag>tex1Dfetch(mag_texRef,(pixIdx-(size.x*dir.y)-dir.x))));
  }
  image[pixIdx] = mag;

}

float* gradientMaximumDetector(float *d_mag, float *d_dir, int width, int height){

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

///Non-maximum supression or Local Maximum Search

  float *d_img;
  cudaMalloc((void**) &d_img, (size.z*sizeof(float)));
  CUT_CHECK_ERROR("Memory image creation failed");

  /// bind a texture to the CUDA array
  cudaBindTexture (NULL ,mag_texRef, d_mag);
  CUT_CHECK_ERROR("Texture bind failed");

  /// host side settable texture attributes
  mag_texRef.normalized = false;
  mag_texRef.filterMode = cudaFilterModePoint;
  
  /// bind a texture to the CUDA array
  cudaBindTexture (NULL ,dir_texRef, d_dir);
  CUT_CHECK_ERROR("Texture bind failed");

  /// host side settable texture attributes
  dir_texRef.normalized = false;
  dir_texRef.filterMode = cudaFilterModePoint;
  
  nonMaximumSupression_texture<<<DimGrid,DimBlock>>>(d_img, size);

  ///free allocated memory
  cudaUnbindTexture(mag_texRef);
  cudaUnbindTexture(dir_texRef);
  CUT_CHECK_ERROR("Memory image free failed");
 
  cudaThreadSynchronize();
  cutStopTimer( timer );  ///< Stop timer
  printf("Maximum Detector time = %f ms\n",cutGetTimerValue( timer ));

  return(d_img);

}


__global__ void hysteresisPreparation(float *hysteresis, int3 size, const unsigned int t1, const unsigned int t2){

  ///pixel index of this thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
  float pixel;


  ///output pixel index
  int2 pos;
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  if(pixIdx < size.z){

    pixel = tex1Dfetch(texRef, pixIdx);

    ///Definitive edge = 128
    ///Possible edge = 255
    ///Non-edge = 0
    pixel = (127*(pixel>t2)+128)*(pixel>t1);

    hysteresis[pixIdx] = pixel;

  }

}

__global__ void hysteresisWrite(float *output, int3 size){

  ///pixel index of this thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
  float pixel;

  if(pixIdx < size.z){

    pixel = tex1Dfetch(hysTexRef, pixIdx);
    output[pixIdx] = (pixel==255) * pixel;

  }

}

__global__ void kernel_hysteresis_glm1D(float *hys_img, int3 size){

  __shared__ float s_slice[324];
  __shared__ int s_modified;     /// inner modified
  __shared__ int block_modified; /// outer modified
  float edge;

  // pixel position indexes on slice
  int2 slice_pos;
  slice_pos.y = threadIdx.x >> 4; // threadIdx.x / 16;
  slice_pos.x = threadIdx.x - (slice_pos.y << 4);

  int sliceIdx = threadIdx.x + 18 + 1;

  // pixel positions indexes on image
  int2 pos;
  pos.x = (slice_pos.x + (blockIdx.x << 4)) % size.x;
  pos.y = (((((blockIdx.x << 4))) / size.x) << 4 ) + slice_pos.y;

  // pixel position at the hysteresis image
  int pixIdx = pos.y * size.x + pos.x;

  // load center
  s_slice[sliceIdx] = hys_img[pixIdx];

  do{
    
    if (!threadIdx.x) block_modified = 0;

    /// load top
    if(!slice_pos.y){
      if(!slice_pos.x){
        s_slice[0] = ((pos.x>0)||(pos.y>0)) * hys_img[pixIdx-size.x-1];///<TL
      }
      s_slice[slice_pos.x+1] = ((pos.y>0)&&(pos.x<size.x-1)) * hys_img[pixIdx-size.x];
      if(slice_pos.x == (15)){
        s_slice[17] = ((pos.x<(size.x-1))&&(pos.y>0)) * hys_img[pixIdx-size.x+1];///<TR
      }
    }
    /// load bottom
    if(slice_pos.y == (15)){
      if(!slice_pos.x){
        s_slice[306] = ((pos.x>0)&&(pos.y<(size.y-1))) * hys_img[pixIdx+size.x-1];///<BL
      }
      s_slice[307+slice_pos.x] = ((pos.x<(size.x-1))&&(pos.y<(size.y-1))) * hys_img[pixIdx+size.x];
      if(threadIdx.x == (blockDim.x-1)){
        s_slice[323] = ((pos.x<(size.x-1))&&(pos.y<(size.y-1))) * hys_img[pixIdx+size.x+1];///<BR
      }
    }
    /// load left
    if(!threadIdx.x){
      s_slice[(slice_pos.y+1)*18] = ((pos.x>0)&&(pos.y<size.y-1)) * hys_img[pixIdx-1];
    }
    /// load right
    if(threadIdx.x == blockDim.x-1){
      s_slice[(slice_pos.y+2)*17] = ((pos.y<(size.y-1))&&(pos.x<(size.x-1))) * hys_img[pixIdx+1];
    }

    do{

      if (!threadIdx.x) s_modified = 0;

      __syncthreads();

      if(s_slice[sliceIdx] == 128){

        /// edge == 1 if at last one pixel's neighbour is a definitive edge 
        /// and edge == 0 if doesn't
        edge = (!(s_slice[sliceIdx-19] != 255) *\
               (s_slice[sliceIdx-18] != 255) *\
               (s_slice[sliceIdx-17] != 255) *\
               (s_slice[sliceIdx-1] != 255) *\
               (s_slice[sliceIdx+1] != 255) *\
               (s_slice[sliceIdx+17] != 255) *\
               (s_slice[sliceIdx+18] != 255) *\
               (s_slice[sliceIdx+19] != 255));
        if ( (edge)*(!s_modified)  ) block_modified = s_modified = 1;
        s_slice[sliceIdx] = 128 + (edge)*127;

      }

      __syncthreads();


    }while(s_modified);// end inner loop

    __syncthreads();

    //if ((pos.x<(size.x-1))&&(pos.y<(size.y-1))) 
    hys_img[pixIdx] = s_slice[sliceIdx];

  }while(block_modified);//end outer loop

}

void hysteresis(float *d_img, int width, int height, const unsigned int t1, const unsigned int t2){
 
  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;

  unsigned int timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  ///< Start timer

  int threadsPerBlock = 256;
  int blocksPerGrid = (size.z + threadsPerBlock -1) >> 8;
  dim3 DimBlock(threadsPerBlock);
  dim3 DimGrid(blocksPerGrid);

  float *d_hys;
  cudaMalloc((void**) &d_hys, (size.z*sizeof(float)));
  CUT_CHECK_ERROR("Memory hysteresis image creation failed");

  /// bind a texture to the CUDA array
  cudaBindTexture (NULL ,texRef, d_img);
  CUT_CHECK_ERROR("Texture bind failed");

  /// host side settable texture attributes
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;
 
  hysteresisPreparation<<<DimGrid,DimBlock>>>(d_hys, size, t1, t2);

  /// free allocated memory
  cudaUnbindTexture(texRef);
  CUT_CHECK_ERROR("Memory unbind failed");

  /// do not remove... ?
  int *tt;
  cudaMalloc((void**) &tt, (blocksPerGrid*sizeof(int)));

  kernel_hysteresis_glm1D<<<DimGrid,DimBlock>>>(d_hys, size);
  CUT_CHECK_ERROR("Hysteresis Kernel failed");

  /// bind a texture to the CUDA array
  cudaBindTexture (NULL ,hysTexRef, d_hys);
  CUT_CHECK_ERROR("Texture bind failed");

  /// host side settable texture attributes
  hysTexRef.normalized = false;
  hysTexRef.filterMode = cudaFilterModePoint;

  hysteresisWrite<<<DimGrid,DimBlock>>>(d_img, size);
  CUT_CHECK_ERROR("Hysteresis Write failed");

  /// free allocated memory
  cudaUnbindTexture(hysTexRef);
  CUT_CHECK_ERROR("Memory unbind failed");

  cudaFree(d_hys);
  CUT_CHECK_ERROR("Memory free failed");
  
  cudaThreadSynchronize();
  cutStopTimer( timer );  ///< Stop timer
  printf("Hysteresis time = %f ms\n",cutGetTimerValue( timer ));

}

