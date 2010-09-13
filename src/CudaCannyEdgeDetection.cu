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
texture<float, 1, cudaReadModeElementType> gaussTexRef;
texture<float, 1, cudaReadModeElementType> mag_texRef;
texture<short2, 1, cudaReadModeElementType> dir_texRef;
texture<float, 1, cudaReadModeElementType> hysTexRef;



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
void cudaGaussian(float *d_output, const float *d_img, int3 size, const float gaussianVariance, unsigned int maxKernelWidth){

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

  /// Allocate temporary memory to image data
  float *d_tmp;
  cudaMalloc((void**) &d_tmp, size.z*sizeof(float));

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

  kernel_1DConvolutionH_texture<<<DimGrid,DimBlock,kernelSize>>>(d_tmp,size,halfkernelsize);

  /// Bind temporary data texture
  cudaUnbindTexture(texRef);
  cudaBindTexture (NULL ,texRef, d_tmp);

  kernel_1DConvolutionV_texture<<<DimGrid,DimBlock,kernelSize>>>(d_output,size,halfkernelsize);

  /// Free allocated memory
  cudaFree(d_tmp);
  cudaUnbindTexture(texRef);
  cudaFree(cudaGaussKernel);
  cudaUnbindTexture(gaussTexRef);
  CUT_CHECK_ERROR("Memory image free failed");

  cudaThreadSynchronize();
  cutStopTimer( timer );  /// Stop timer
  printf("Gaussian time = %f ms\n",cutGetTimerValue( timer ));

}


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
void cudaSobel(float* d_mag, short2 *d_dir, float *d_img, int3 size){

  int threadsPerBlock = 256;
  int blocksPerGrid = ((size.z) + threadsPerBlock -1) >> 8;
  dim3 DimBlock(threadsPerBlock,1,1);
  dim3 DimGrid(blocksPerGrid,1,1);

  unsigned int timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  ///< Start timer

  /// bind a texture to the CUDA array
  cudaBindTexture (NULL ,texRef, d_img);
  CUT_CHECK_ERROR("Texture bind failed");

  /// host side settable texture attributes
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;

  kernel_2DSobel<<<DimGrid,DimBlock>>>(d_mag, d_dir, size);

  cudaThreadSynchronize();
  cutStopTimer( timer );  ///< Stop timer
  printf("Sobel time = %f ms\n",cutGetTimerValue( timer ));

  /// unbind texture reference
  cudaUnbindTexture(texRef);
  CUT_CHECK_ERROR("Memory image free failed");

}


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
void gradientMaximumDetector(float *d_img, float *d_mag, short2 *d_dir, int3 size){

  int threadsPerBlock = 256;
  int blocksPerGrid = ((size.z) + threadsPerBlock -1) >> 8;
  dim3 DimBlock(threadsPerBlock,1,1);
  dim3 DimGrid(blocksPerGrid,1,1);

  unsigned int timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  ///< Start timer

///Non-maximum supression or Local Maximum Search

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
    ///Non-edge = 0 * pixel
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

  // thread index
  //int tid = blockIdx.x * blockDim.x + threadIdx.x;

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
void hysteresis(float *d_img, int3 size, const unsigned int t1, const unsigned int t2){

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


float* cudaCanny(const float *d_input, int width, int height, const float gaussianVariance, const unsigned int maxKernelWidth, const unsigned int t1, const unsigned int t2){

  printf(" Parameters:\n");
  printf(" |-Image Size: (%d,%d)\n",width,height);
  printf(" |-Variance: %f\n",gaussianVariance);
  printf(" |-Max Kernel Width: %d\n",maxKernelWidth);
  printf(" --Thresholds: (%d,%d)\n",t2,t1);

  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;

  /// Image Pointers
  float *d_blur;
  float *d_edges;

  /// Warmup
  unsigned int WarmupTimer = 0;
  cutCreateTimer( &WarmupTimer );
  cutStartTimer( WarmupTimer );
  int *rub;
  cudaMalloc( (void**)&rub, size.z * sizeof(int) );
  cudaFree( rub );
  CUT_CHECK_ERROR("Warmup failed");
  cudaThreadSynchronize();
  cutStopTimer( WarmupTimer );
  printf("Warmup time = %f ms\n",cutGetTimerValue( WarmupTimer ));

  unsigned int timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  ///< Start timer

  cudaMalloc((void**) &d_blur, (size.z*sizeof(float)));
  CUT_CHECK_ERROR("Memory hysteresis image creation failed");

  cudaGaussian(d_blur,d_input,size,gaussianVariance,maxKernelWidth);

  /// alocate memory on gpu for direction
  short2 *d_direction;
  cudaMalloc((void**) &d_direction, size.z*sizeof(short2));
  CUT_CHECK_ERROR("Image memory direction creation failed");
  float *d_magnitude;
  cudaMalloc((void**) &d_magnitude, size.z*sizeof(float));
  CUT_CHECK_ERROR("Memory temporary image creation failed");

  cudaSobel(d_magnitude,d_direction,d_blur,size);

  d_edges = d_blur;

  gradientMaximumDetector(d_edges,d_magnitude,d_direction,size);

  hysteresis(d_edges,size,t1,t2);

  cudaThreadSynchronize();
  cutStopTimer( timer );  ///< Stop timer
  printf("cudaCanny total time = %f ms\n",cutGetTimerValue( timer ));

  /// free memory used for image magnitude
  cudaFree(d_direction);
  CUT_CHECK_ERROR("Image direction memory free failed");
  cudaFree(d_magnitude);
  CUT_CHECK_ERROR("Image memory free failed");

  return(d_edges);
}
