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

/*********************** Teste ************************
  int i,j;
  int img[size.z];
  cudaMemcpy(img,d_image,size.z*sizeof(int),cudaMemcpyDeviceToHost);
  for(i=0; i<size.y;i++){
    for(j=0; j<size.x;j++){
      printf("%d ",img[(i*size.x)+j]);
    }
    printf("\n");
  }
  printf("\n");
******************* Fim do Teste *********************/

extern "C"
void cudaCanny(float *image, int width, int height, const float gaussianVariance, const unsigned int maxKernelWidth, const unsigned int t1, const unsigned int t2);




/// allocate texture variables
texture<float, 1, cudaReadModeElementType> texRef;
texture<float, 1, cudaReadModeElementType> gaussTexRef;
texture<float, 1, cudaReadModeElementType> maxTexRef;
texture<float, 1, cudaReadModeElementType> gx_texRef;
texture<float, 1, cudaReadModeElementType> gy_texRef;
texture<float, 1, cudaReadModeElementType> mag_texRef;
texture<short2, 1, cudaReadModeElementType> dir_texRef;
texture<unsigned char, 1, cudaReadModeElementType> charTexRef;
texture<int, 1, cudaReadModeElementType> hysTexRef;

/// convertImage_C2I_texture
/*
 * \details This kernel makes a convertion of the data from an array of char to
 * an array of int. It uses texture fetch to fast access of the input data.
 * \param output output array.
 * \param N number of positions of the array.
 */
__global__ void convertImage_C2I_texture(int *output, int N){

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i<N){
    output[i] = tex1Dfetch(charTexRef, i);
  }
}

extern "C"
void gpuInput(unsigned char *input, int *d_image, int numElements){

  ///a grid possui um bloco a mais, possivelmente incompleto
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock -1) >> 8;
  dim3 DimBlock(threadsPerBlock);
  dim3 DimGrid(blocksPerGrid);

  unsigned int timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  /// Start timer

  ///allocate temporary memory to image data
  unsigned char *d_tmp;
  cudaMalloc((void**) &d_tmp, numElements*sizeof(unsigned char));
  CUT_CHECK_ERROR("Memory temporary image creation failed");

  cudaMemcpy(d_tmp,input,numElements*sizeof(unsigned char),cudaMemcpyHostToDevice);
  CUT_CHECK_ERROR("Memory image copy failed");

  /// bind a texture to the CUDA array
  cudaBindTexture (NULL ,charTexRef, d_tmp);
  CUT_CHECK_ERROR("Texture bind failed");

  /// host side settable texture attributes
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;

  convertImage_C2I_texture<<<DimGrid,DimBlock>>>(d_image,numElements);

  cudaFree(d_tmp);
  cudaUnbindTexture(charTexRef);
  CUT_CHECK_ERROR("Memory image free failed");

  cudaThreadSynchronize();
  cutStopTimer( timer );  /// Stop timer
  printf("Image input time = %f ms\n",cutGetTimerValue( timer ));

}

__global__ void convertImage_I2C4_texture4(uchar4 *output, int N){
///in this version, the max value is read from texture memory

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  uchar4 tmp;

  if(i<(__fdividef(N,4))){
    //use make_uchar4()
    tmp.x = (char)tex1Dfetch(texRef, (i*4));
    tmp.y = (char)tex1Dfetch(texRef, (i*4)+1);
    tmp.z = (char)tex1Dfetch(texRef, (i*4)+2);
    tmp.w = (char)tex1Dfetch(texRef, (i*4)+3);
    output[i] = tmp;
  }
}

extern "C"
void gpuOutput(int *image, unsigned char *output, int numElements){

  int threadsPerBlock = 256;
  int blocksPerGrid = ((numElements/4) + threadsPerBlock -1) >> 8;
  dim3 DimBlock(threadsPerBlock);
  dim3 DimGrid(blocksPerGrid);

  unsigned int timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  /// Start timer

  uchar4 *d_tmp;
  cudaMalloc((void**) &d_tmp, (numElements));
  CUT_CHECK_ERROR("Memory temporary image creation failed");

  /// bind a texture to the CUDA array
  cudaBindTexture (NULL ,texRef, image);
  CUT_CHECK_ERROR("Texture bind failed");

  /// host side settable texture attributes
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;

  convertImage_I2C4_texture4<<<DimGrid,DimBlock>>>(d_tmp, numElements);

  cudaMemcpy(output, d_tmp, numElements, cudaMemcpyDeviceToHost);
  CUT_CHECK_ERROR("Memory image copy failed");

  cudaUnbindTexture(texRef);
  cudaFree(d_tmp);
  CUT_CHECK_ERROR("Memory image free failed");

  cudaThreadSynchronize();
  cutStopTimer( timer );  /// Stop timer
  printf("Image output time = %f ms\n",cutGetTimerValue( timer ));

}


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

__global__ void kernel_1DConvolutionH_texture(float *output, int3 size, int kernelsize){
///this version uses the texture memory to store the gaussian kernel and the
///image data

  int2 sum;
  int2 pos;

  extern __shared__ float s_gauss[];

  ///pixel index of this thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int halfkernelsize = kernelsize >> 1;

  ///output pixel index
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  if(threadIdx.x<kernelsize) s_gauss[threadIdx.x] = tex1Dfetch(gaussTexRef,threadIdx.x);
  
      sum.x = sum.y = 0;
    if ((pos.x >= (halfkernelsize)) && (pos.x <= (size.x-halfkernelsize-1))){
#pragma unroll
      for(int k=0;k<kernelsize;k++){
        sum.x += (tex1Dfetch(texRef, pixIdx+(k-(halfkernelsize)))*s_gauss[k]);
        sum.y += s_gauss[k];
      }
      sum.x = __fdividef(sum.x,sum.y);
    }

    output[pixIdx] = sum.x;
}

__global__ void kernel_1DConvolutionV_texture(float *output, int3 size, int kernelsize){
///this version uses the texture memory to store the gaussian kernel and the
///image data

  int2 sum;
  int2 pos;

  extern __shared__ float s_gauss[];

  ///pixel index of this thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int halfkernelsize = kernelsize >> 1;

  ///output pixel index
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  if(threadIdx.x<kernelsize) s_gauss[threadIdx.x] = tex1Dfetch(gaussTexRef,threadIdx.x);
  
      sum.x = sum.y = 0;
    if ((pos.y >= (halfkernelsize)) && (pos.y <= (size.y-halfkernelsize-1))){
#pragma unroll
      for(int k=0;k<kernelsize;k++){
        sum.x += (tex1Dfetch(texRef, pixIdx+(size.x*(k-(halfkernelsize))))*s_gauss[k]);
        sum.y += s_gauss[k];
      }
      sum.x = __fdividef(sum.x,sum.y);
    }

    output[pixIdx] = sum.x;
}

extern "C"
void cudaGaussian(float *d_img, int3 size, const float gaussianVariance, unsigned int maxKernelWidth){

  int threadsPerBlock = 256;
  int blocksPerGrid = ((size.z) + threadsPerBlock -1) >> 8;
  dim3 DimBlock(threadsPerBlock,1,1);
  dim3 DimGrid(blocksPerGrid,1,1);

  int kernelSize = maxKernelWidth*sizeof(float);

  unsigned int timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  /// Start timer

  if (maxKernelWidth < 1) maxKernelWidth = 1;
  if (maxKernelWidth%2 == 0) maxKernelWidth--;

///calculate gaussian mask
  float *cudaGaussKernel;
  cudaMalloc((void**)&cudaGaussKernel,kernelSize);

  calculateGaussianKernel<<<1,maxKernelWidth,kernelSize>>>(cudaGaussKernel, gaussianVariance, maxKernelWidth);

///use calculated mask to gaussian filter

  ///allocate temporary memory to image data
  float *d_tmp;
  cudaMalloc((void**) &d_tmp, size.z*sizeof(float));

  /// bind a texture to the CUDA array
  cudaBindTexture (NULL, gaussTexRef, cudaGaussKernel);
  CUT_CHECK_ERROR("Texture bind failed");

  /// host side settable texture attributes
  gaussTexRef.normalized = false;
  gaussTexRef.filterMode = cudaFilterModePoint;

  /// bind a texture to the CUDA array
  cudaBindTexture (NULL, texRef, d_img);
  CUT_CHECK_ERROR("Texture bind failed");

  /// host side settable texture attributes
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;

  kernel_1DConvolutionH_texture<<<DimGrid,DimBlock,kernelSize>>>(d_tmp,size,maxKernelWidth);

  ///bind temporary data texture
  cudaUnbindTexture(texRef);
  cudaBindTexture (NULL ,texRef, d_tmp);

  kernel_1DConvolutionV_texture<<<DimGrid,DimBlock,kernelSize>>>(d_img,size,maxKernelWidth);

  ///free allocated memory
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

__global__ void nonMaximumSupression_texture(float* image, int3 size, int borderSize){
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

  if ((pos.x>=borderSize) && (pos.x<=((size.x-1)-borderSize)) && (pos.y>=borderSize) && (pos.y<=((size.y-1)-borderSize))){

    mag = tex1Dfetch(mag_texRef,i);
    dir = tex1Dfetch(dir_texRef,i);
    mag *= ((mag>=tex1Dfetch(mag_texRef,(i+(size.x*dir.y)+dir.x)))*(mag>tex1Dfetch(mag_texRef,(i-(size.x*dir.y)-dir.x))));
  }
  image[i] = mag;

}

extern "C"
void gradientMaximumDetector(float *d_img, float *d_mag, short2 *d_dir, int3 size, int gaussKernelWidth){

  int threadsPerBlock = 256;
  int blocksPerGrid = ((size.z) + threadsPerBlock -1) >> 8;
  dim3 DimBlock(threadsPerBlock,1,1);
  dim3 DimGrid(blocksPerGrid,1,1);

  unsigned int timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  ///< Start timer

  int borderSize = (gaussKernelWidth>>1) + 1;

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
  
  nonMaximumSupression_texture<<<DimGrid,DimBlock>>>(d_img, size, borderSize);

  ///free allocated memory
  cudaUnbindTexture(mag_texRef);
  cudaUnbindTexture(dir_texRef);
  CUT_CHECK_ERROR("Memory image free failed");
 
  cudaThreadSynchronize();
  cutStopTimer( timer );  ///< Stop timer
  printf("Maximum Detector time = %f ms\n",cutGetTimerValue( timer ));

}

__global__ void hysteresisPreparation(int *hysteresis, int3 size, const unsigned int t1, const unsigned int t2){

  ///pixel index of this thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int pixel;


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
  int pixel;

  if(pixIdx < size.z){

    pixel = tex1Dfetch(hysTexRef, pixIdx);
    output[pixIdx] = (pixel==255) * pixel;

  }

}

__device__ int reduceSum256(int tid, int data[]){
// data[] should be a 256 positios array

  if (tid < 128) { data[tid] += data[tid + 128]; }
  __syncthreads();
  if (tid < 64) { data[tid] += data[tid + 64]; }
  __syncthreads();
  if (tid < 32) { data[tid] += data[tid + 32]; }
  __syncthreads();
  if (tid < 16) { data[tid] += data[tid + 16]; }
  __syncthreads();
  if (tid < 8) { data[tid] += data[tid + 8]; }
  __syncthreads();
  if (tid < 4) { data[tid] += data[tid + 4]; }
  __syncthreads();
  if (tid < 2) { data[tid] += data[tid + 2]; }
  __syncthreads();
  if (tid == 0) { data[tid] += data[tid + 1]; }

  return(data[0]);

}

__device__ int reduceSum(int tid, int data[], int N, int pow2){

  if (tid<pow2){

    if (tid>=N) data[tid] = 0;

    for(unsigned int s=pow2/2; s>0; s>>=1) {
      if (tid < s){
      data[tid] += data[tid + s];
    }
      __syncthreads();
    }

  }
  return(data[0]);

}


__global__ void kernel_hysteresis_glm3(int *hys_img, int3 size, int *modified, int N, int pow2){

  __shared__ float s_slice[18][18];
  __shared__ int s_modified[256];
  __shared__ int m;

  int block_tid = blockDim.x * threadIdx.y + threadIdx.x;
  int tid = block_tid + (blockIdx.x*blockDim.x*blockDim.y)+(blockIdx.y*blockDim.x*blockDim.y*gridDim.x);
  
  ///pixel index of this thread
  int2 pos;
  pos.x = blockIdx.x * blockDim.x + threadIdx.x;
  pos.y = blockIdx.y * blockDim.y + threadIdx.y;
  int pixIdx = pos.y * size.x + pos.x;
  int2 sliceIdx;
  sliceIdx.x = threadIdx.x+1;
  sliceIdx.y = threadIdx.y+1;
  int edge;
  int i;

  ///load center
  s_slice[sliceIdx.y][sliceIdx.x] = hys_img[pixIdx];

  s_modified[block_tid] = 1;

  do{

    if ((threadIdx.x + threadIdx.y) == 0) m = 0;

    if (m){
      //store center
      hys_img[pixIdx] = s_slice[sliceIdx.x][sliceIdx.y];
    }

    if((!i) || m){

     ///load top
      if(!threadIdx.y){
        if(!threadIdx.x){
          s_slice[0][0] = ((pos.x>0)||(pos.y>0)) * hys_img[pixIdx-size.x-1];///<TL
        }
        s_slice[0][sliceIdx.x] = (pos.y>0) * hys_img[pixIdx-size.x];
        if(threadIdx.x == (blockDim.x-1)){
           s_slice[0][blockDim.x+1] = ((pos.x<(size.x-1))&&(pos.y>0)) * hys_img[pixIdx-size.x+1];///<TR
        }
      }
      ///load bottom
      if(threadIdx.y == (blockDim.y-1)){
        if(!threadIdx.x){
          s_slice[blockDim.y+1][0] = ((pos.x>0)&&(pos.y<(size.y-1))) * hys_img[pixIdx+size.x-1];///<BL
        }
        s_slice[blockDim.y+1][sliceIdx.x] = (pos.y<(size.y-1)) * hys_img[pixIdx+size.x];
        if(threadIdx.x == (blockDim.x-1)){
          s_slice[blockDim.y+1][blockDim.x+1] = ((pos.x<(size.x-1))&&(pos.y<(size.y-1))) * hys_img[pixIdx+size.x+1];///<BR
        }
      }
  
      ///load left
      if(!threadIdx.x){
        s_slice[sliceIdx.y][0] = (pos.x>0) * hys_img[pixIdx-1];
      }
      ///load right
      if(threadIdx.x == blockDim.x-1){
        s_slice[sliceIdx.y][blockDim.x+1] = (pos.x<(size.x-1)) * hys_img[pixIdx+1];
      }
    }

    while(!reduceSum256(block_tid,s_modified)){

      s_modified[block_tid] = 0;

      if(s_slice[sliceIdx.y][sliceIdx.x] == 128){

        __syncthreads();

        /// edge == 1 if at last one pixel's neighbour is a definitive edge 
        /// and edge == 0 if doesn't
        edge = (!(s_slice[sliceIdx.y-1][sliceIdx.x-1] != 255) *\
                 (s_slice[sliceIdx.y-1][sliceIdx.x] != 255) *\
                 (s_slice[sliceIdx.y-1][sliceIdx.x+1] != 255) *\
                 (s_slice[sliceIdx.y][sliceIdx.x-1] != 255) *\
                 (s_slice[sliceIdx.y][sliceIdx.x+1] != 255) *\
                 (s_slice[sliceIdx.y+1][sliceIdx.x-1] != 255) *\
                 (s_slice[sliceIdx.y+1][sliceIdx.x] != 255) *\
                 (s_slice[sliceIdx.y+1][sliceIdx.x+1] != 255));

        s_modified[block_tid] = (edge);
        s_slice[sliceIdx.y][sliceIdx.x] = (float) 128 + (edge)*127;    
      }

      if ((threadIdx.x+threadIdx.y)==0) m = 1;

    }

    if ((threadIdx.x + threadIdx.y) == 0) modified[blockIdx.y*gridDim.x + blockIdx.x] = m;

  }while(!reduceSum(tid,modified,N,pow2));

  if((pos.x < (size.x)) && (pos.y < (size.y))){ 
    hys_img[pixIdx] = s_slice[sliceIdx.y][sliceIdx.x];
  }
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

  int threadsPerBlockDim = 16;
  int blocksPerGridX = ((size.x) + threadsPerBlockDim -1) >> 4;
  int blocksPerGridY = ((size.y) + threadsPerBlockDim -1) >> 4;
  int nBlocks = blocksPerGridX*blocksPerGridY;
  dim3 TwoDimBlock(16,16,1);
  dim3 TwoDimGrid(blocksPerGridX,blocksPerGridY,1);

  int pow2 = 1;
  while (pow2<nBlocks) pow2 = pow2 << 1;

  int *d_hys;
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
  cudaMalloc((void**) &d_modified, (pow2*sizeof(int)));
  CUT_CHECK_ERROR("Memory hysteresis image creation failed");

  kernel_hysteresis_glm3<<<TwoDimGrid,TwoDimBlock>>>(d_hys, size, d_modified, nBlocks, pow2);
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
  cudaFree(d_modified);
  CUT_CHECK_ERROR("Memory free failed");

  
  cudaThreadSynchronize();
  cutStopTimer( timer );  ///< Stop timer
  printf("Hysteresis time = %f ms\n",cutGetTimerValue( timer ));

}

void cudaCanny(float *image, int width, int height, const float gaussianVariance, const unsigned int maxKernelWidth, const unsigned int t1, const unsigned int t2){

  printf(" Parameters:\n");
  printf(" |-Image Size: (%d,%d)\n",width,height);
  printf(" |-Variance: %f\n",gaussianVariance);
  printf(" |-Max Kernel Width: %d\n",maxKernelWidth);
  printf(" --Thresholds: (%d,%d)\n",t2,t1);

  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;
  int imageSize = size.z*sizeof(float);

  ///Warmup
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

  /// alocate memory on gpu for image
  float *d_image;
  cudaMalloc((void**) &d_image, imageSize);
  CUT_CHECK_ERROR("Image memory creation failed");

//  gpuInput(image,d_image,size.z);
  cudaMemcpy(d_image,image,size.z*sizeof(float),cudaMemcpyHostToDevice);

  cudaGaussian(d_image,size,gaussianVariance,maxKernelWidth);

  /// alocate memory on gpu for direction
  short2 *d_direction;
  cudaMalloc((void**) &d_direction, size.z*sizeof(short2));
  CUT_CHECK_ERROR("Image memory direction creation failed");
  float *d_magnitude;
  cudaMalloc((void**) &d_magnitude, size.z*sizeof(float));
  CUT_CHECK_ERROR("Memory temporary image creation failed");

  cudaSobel(d_magnitude,d_direction,d_image,size);

  gradientMaximumDetector(d_image,d_magnitude,d_direction,size,maxKernelWidth);

  cudaFree(d_direction);
  CUT_CHECK_ERROR("Image direction memory free failed");
  cudaFree(d_magnitude);
  CUT_CHECK_ERROR("Image memory free failed");

  hysteresis(d_image,size,t1,t2);

//  gpuOutput(d_image,image,size.z);
  cudaMemcpy(image,d_image,size.z*sizeof(float),cudaMemcpyDeviceToHost);

  /// free memory used for image magnitude
  cudaFree(d_image);
  CUT_CHECK_ERROR("Image memory free failed");

  cudaThreadSynchronize();
  cutStopTimer( timer );  ///< Stop timer
  printf("cudaCanny total time = %f ms\n",cutGetTimerValue( timer ));

}
