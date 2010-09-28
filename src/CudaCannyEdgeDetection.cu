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
#include <cudpp.h>

#include "CudaCannyEdgeDetection.h"


/// allocate texture variables
texture<float, 1, cudaReadModeElementType> texRef;
texture<float, 1, cudaReadModeElementType> mag_texRef;
texture<short2, 1, cudaReadModeElementType> dir_texRef;
texture<float, 1, cudaReadModeElementType> hysTexRef;


__global__ void nonMaximumSupression_texture(float* image, int3 size){
///this is the kernel to calculate the non-maximum supression of the image. It is
///implemmented using texture fetching. The dependence of inter-block data makes
///the use of shared memory hard-boiled.

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  float mag = 0;
  short2 dir;

  ///output pixel index
  int2 pos;
  pos.y = __fdividef(i,size.x);
  pos.x = i-(pos.y*size.x);

  if ((pos.x>0) && (pos.x<((size.x-1))) && (pos.y>0) && (pos.y<((size.y-1)))){

    mag = tex1Dfetch(mag_texRef,i);
    dir = tex1Dfetch(dir_texRef,i);
    mag *= ((mag>=tex1Dfetch(mag_texRef,(i+(size.x*dir.y)+dir.x)))*(mag>tex1Dfetch(mag_texRef,(i-(size.x*dir.y)-dir.x))));
  }
  image[i] = mag;

}

extern "C"
float* gradientMaximumDetector(float *d_mag, short2 *d_dir, int width, int height){

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

__device__ int reduceSum256(int tid, int sdata[]){
// data[] should be a 256 positios array

  if (tid < 128) { sdata[tid] += sdata[tid + 128]; }
  __syncthreads();
  if (tid < 64) { sdata[tid] += sdata[tid + 64]; }
  __syncthreads();
  if (tid < 32) { sdata[tid] += sdata[tid + 32]; }
  __syncthreads();
  if (tid < 16) { sdata[tid] += sdata[tid + 16]; }
  __syncthreads();
  if (tid < 8) { sdata[tid] += sdata[tid + 8]; }
  __syncthreads();
  if (tid < 4) { sdata[tid] += sdata[tid + 4]; }
  __syncthreads();
  if (tid < 2) { sdata[tid] += sdata[tid + 2]; }
  __syncthreads();
  if (tid == 0) { sdata[tid] += sdata[tid + 1]; }
  __syncthreads();
  
  return(sdata[0]);

}

__global__ void kernel_hysteresis_glm1D(float *hys_img, int3 size, int *modified){

  __shared__ float s_slice[324];
  __shared__ int s_modified[256];
  float edge;
  int i;

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

  for(i=0;i<256;i++){

    s_modified[threadIdx.x] = 0;

    if(s_slice[sliceIdx] == 128){

      __syncthreads();

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
      s_modified[threadIdx.x] = (edge);
      s_slice[sliceIdx] = 128 + (edge)*127;

    }

    reduceSum256(threadIdx.x,s_modified);

    if (!s_modified[0]) break;

  }// end inner loop

  if (!threadIdx.x) modified[blockIdx.x] = s_modified[0];

  __syncthreads();

  hys_img[pixIdx] = s_slice[sliceIdx];

}

extern "C"
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
  int nBlocks = blocksPerGrid;

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

  int *d_modified;
  cudaMalloc((void**) &d_modified, (nBlocks*sizeof(int)));
  CUT_CHECK_ERROR("Memory hysteresis image creation failed");

  int *d_modifsum;
  cudaMalloc((void**) &d_modifsum, (nBlocks*sizeof(int)));
  CUT_CHECK_ERROR("Memory hysteresis image creation failed");

  CUDPPConfiguration config;
  config.op = CUDPP_ADD;
  config.datatype = CUDPP_INT;
  config.algorithm = CUDPP_SCAN;
  config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;

  CUDPPHandle scanplan = 0;
  CUDPPResult result = cudppPlan(&scanplan, config, nBlocks, 1, 0);  

  if (CUDPP_SUCCESS != result){
    printf("Error creating CUDPPPlan\n");
    exit(-1);
  }

  int a;
  int cont[1];
  // The value of 100 loops was archieved empirically
  for(a = 0; a<100; a++){

    kernel_hysteresis_glm1D<<<DimGrid,DimBlock>>>(d_hys, size, d_modified);

    cudppScan(scanplan, d_modifsum, d_modified, nBlocks);
    cudaMemcpy(cont,d_modifsum,sizeof(int),cudaMemcpyDeviceToHost);

    if (!cont[0]) break;

  }
  CUT_CHECK_ERROR("Hysteresis Kernel failed");
  //printf("Histeresis Interations: %d\n",a+1);

  result = cudppDestroyPlan(scanplan);
  if (CUDPP_SUCCESS != result){
    printf("Error destroying CUDPPPlan\n");
    exit(-1);
  }

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
  cudaFree(d_modified);
  CUT_CHECK_ERROR("Memory free failed");
  
  cudaThreadSynchronize();
  cutStopTimer( timer );  ///< Stop timer
  printf("Hysteresis time = %f ms\n",cutGetTimerValue( timer ));

}

