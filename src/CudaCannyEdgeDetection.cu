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

#define THREADS_PER_BLOCK 256


/// allocate texture variables
texture<float, 1, cudaReadModeElementType> texRef;
texture<float, 1, cudaReadModeElementType> mag_texRef;
texture<float, 1, cudaReadModeElementType> hysTexRef;
texture<float, 1, cudaReadModeElementType> der_texRef;


__global__ void kernel_Compute2ndDerivativePos(float *Magnitude, int3 size){
/// This kernel receive the blurred image and it's second derivative and returns
/// the Gradient Magnitude of the image.
/// It ignores completely the borders of the image because the border's pixels
/// doesn't have all neighbors.

  /// Pixel index of this thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;

  /// Output pixel index
  int2 pos;
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  float4 cross_Lvv;
  float4 cross_L;

  float Lx = 0, 
        Ly = 0, 
        Lvvx = 0, 
        Lvvy = 0,
        gradMag = 0; 

  ///Ignore the image borders
  if ((pos.x) && ((size.x-1)-pos.x) && (pos.y) && ((size.y-1)-pos.y)){

    cross_L.x = tex1Dfetch(texRef,(pixIdx-size.x));
    cross_L.y = tex1Dfetch(texRef,(pixIdx+size.x));
    cross_L.z = tex1Dfetch(texRef,(pixIdx-1));
    cross_L.w = tex1Dfetch(texRef,(pixIdx+1));
    cross_Lvv.x = tex1Dfetch(der_texRef,(pixIdx-size.x));
    cross_Lvv.y = tex1Dfetch(der_texRef,(pixIdx+size.x));
    cross_Lvv.z = tex1Dfetch(der_texRef,(pixIdx-1));
    cross_Lvv.w = tex1Dfetch(der_texRef,(pixIdx+1));

    Lx = (-0.5*cross_L.z) + (0.5*cross_L.w);
    Ly = (0.5*cross_L.x) - (0.5*cross_L.y);

    Lvvx = (-0.5*cross_Lvv.z) + (0.5*cross_Lvv.w);
    Lvvy = (0.5*cross_Lvv.x) - (0.5*cross_Lvv.y);

    gradMag = sqrt((Lx*Lx)+(Ly*Ly));

  }

  Magnitude[pixIdx] = (((Lvvx*(Lx/gradMag)+Lvvy*(Ly/gradMag))<=0)*gradMag);

}

float* cuda2ndDerivativePos(const float *d_input, const float *d_Lvv, int width, int height){

  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;

  int blocksPerGrid = ((size.z) + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK;
  dim3 DimBlock(THREADS_PER_BLOCK,1,1);
  dim3 DimGrid(blocksPerGrid,1,1);

 /// Allocate output memory to image data
  float * d_mag;
  cudaMalloc((void**) &d_mag, (size.z*sizeof(float)));
  CUT_CHECK_ERROR("Memory image creation failed");

  /// bind a texture to the CUDA array
  cudaBindTexture (NULL, texRef, d_input);
  cudaBindTexture (NULL, der_texRef, d_Lvv);
  CUT_CHECK_ERROR("Texture bind failed");

  /// host side settable texture attributes
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;

  kernel_Compute2ndDerivativePos<<<DimGrid,DimBlock>>>(d_mag, size);

  /// unbind texture reference
  cudaUnbindTexture(texRef);
  cudaUnbindTexture(der_texRef);
  CUT_CHECK_ERROR("Memory image free failed");

  return(d_mag);

}


__global__ void kernel_Compute2ndDerivative(float *Lvv, int3 size){
/// This kernel receives a blurred image and returns it's second derivative.
/// It ignores completely the borders of the image becausa the border's pixels
/// doesn't have all neighbors

  /// Pixel index of this thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;

  /// Output pixel index
  int2 pos;
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  float4 diagonal;
  float4 cross;
  float pixel = tex1Dfetch(texRef,(pixIdx));

  float Lx = 0, 
        Ly = 0, 
        Lxx = 0, 
        Lxy = 0, 
        Lyy = 0;

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

    Lx = (-0.5*cross.z) + (0.5*cross.w);
    Ly = (0.5*cross.x) - (0.5*cross.y);
    Lxx = cross.z - (2.0*pixel) + cross.w;
    Lxy = (-0.25*diagonal.x) + (0.25*diagonal.y) + (0.25*diagonal.z) + (-0.25*diagonal.w);
    Lyy = cross.x -(2.0*pixel) + cross.y;

  }

  Lvv[pixIdx] = (((Lx*Lx)*Lxx) + (2.0*Lx*Ly*Lxy) + (Ly*Ly*Lyy))/((Lx*Lx) + (Ly*Ly));

}

float* cuda2ndDerivative(const float *d_input, int width, int height){

  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;

  int blocksPerGrid = ((size.z) + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK;
  dim3 DimBlock(THREADS_PER_BLOCK,1,1);
  dim3 DimGrid(blocksPerGrid,1,1);

 /// Allocate output memory to image data
  float * d_Lvv;
  cudaMalloc((void**) &d_Lvv, (size.z*sizeof(float)));
  CUT_CHECK_ERROR("Memory image creation failed");

  /// bind a texture to the CUDA array
  cudaBindTexture (NULL ,texRef, d_input);
  CUT_CHECK_ERROR("Texture bind failed");

  /// host side settable texture attributes
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;

  kernel_Compute2ndDerivative<<<DimGrid,DimBlock>>>(d_Lvv, size);

  /// unbind texture reference
  cudaUnbindTexture(texRef);
  CUT_CHECK_ERROR("Memory image free failed");

  return(d_Lvv);

}


__global__ void hysteresisPreparation_kernel(float *hysteresis, int3 size, const unsigned int t1, const unsigned int t2){

  ///pixel index of this thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
  float pixel;


  ///output pixel index
  int2 pos;
  pos.y = __fdividef(pixIdx,size.x);
  pos.x = pixIdx-(pos.y*size.x);

  if(pixIdx < size.z){

    pixel = tex1Dfetch(mag_texRef, pixIdx)*tex1Dfetch(texRef,pixIdx);

    hysteresis[pixIdx] = ((POSSIBLE_EDGE-1)*(pixel>t2)+POSSIBLE_EDGE)*(pixel>t1);

  }

}

__global__ void hysteresisWrite_kernel(float *output, int3 size){

  ///pixel index of this thread
  int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
  float pixel;

  if(pixIdx < size.z){

    pixel = tex1Dfetch(hysTexRef, pixIdx);
    output[pixIdx] = (pixel==DEFINITIVE_EDGE) * pixel;

  }

}

__global__ void kernel_hysteresis_glm1D(float *hys_img, int3 size, int *have_modified_pixels){

  __shared__ float s_slice[SLICE_WIDTH*SLICE_WIDTH];
  __shared__ int modified_block_pixels; /// control inner loop
  __shared__ int modified_image_pixels; /// control outer loop
  int edge;

  // pixel position indexes on block
  int2 slice_pos;
  slice_pos.y = threadIdx.x / SLICE_BLOCK_WIDTH;
  slice_pos.x = threadIdx.x - (slice_pos.y * SLICE_BLOCK_WIDTH);

  // pixel positions indexes on image
  int2 pos;
  pos.x = (slice_pos.x + (blockIdx.x * SLICE_BLOCK_WIDTH)) % size.x;
  pos.y = (((((blockIdx.x * SLICE_BLOCK_WIDTH))) / size.x) * SLICE_BLOCK_WIDTH ) + slice_pos.y;

  int sliceIdx = (slice_pos.y*SLICE_WIDTH) + slice_pos.x  + SLICE_WIDTH + 1;

  // pixel position at the hysteresis image
  int pixIdx = pos.y * size.x + pos.x;

  // load center
  s_slice[sliceIdx] = hys_img[pixIdx];

  if (!threadIdx.x) modified_image_pixels = NOT_MODIFIED;
  if (!pixIdx) have_modified_pixels[0] = 0;

  /// load top
  if(!slice_pos.y){
    if(!slice_pos.x){
      s_slice[0] = ((pos.x>0)&&(pos.y>0)) * hys_img[pixIdx-size.x-1];///<TL
    }
    s_slice[slice_pos.x+1] = ((pos.y>0)&&(pos.y<size.y-1)) * hys_img[pixIdx-size.x];
    if(slice_pos.x == (SLICE_BLOCK_WIDTH-1)){
      s_slice[SLICE_WIDTH-1] = ((pos.x<(size.x-1))&&(pos.y>0)) * hys_img[pixIdx-size.x+1];///<TR
    }
  }
  /// load bottom
  if(slice_pos.y == (SLICE_BLOCK_WIDTH-1)){
    if(!slice_pos.x){
      s_slice[SLICE_WIDTH*(SLICE_WIDTH-1)] = ((pos.x>0)&&(pos.y<(size.y-1))) * hys_img[pixIdx+size.x-1];///<BL
    }
    s_slice[(SLICE_WIDTH*(SLICE_WIDTH-1))+1+slice_pos.x] = ((pos.y<(size.y-1))) * hys_img[pixIdx+size.x];
    if(slice_pos.x == (SLICE_BLOCK_WIDTH-1)){
      s_slice[(SLICE_WIDTH*SLICE_WIDTH)-1] = ((pos.x<(size.x-1))&&(pos.y<(size.y-1))) * hys_img[pixIdx+size.x+1];///<BR
    }
  }
  /// load left
  if(!slice_pos.x){
    s_slice[(slice_pos.y+1)*SLICE_WIDTH] = ((pos.x>0)) * hys_img[pixIdx-1];
  }
  /// load right
  if(slice_pos.x == (SLICE_BLOCK_WIDTH-1)){
    s_slice[((slice_pos.y+2)*SLICE_WIDTH)-1] = (pos.x<(size.x-1)) * hys_img[pixIdx+1];
  }

  __syncthreads();

  do{

    if (!threadIdx.x) modified_block_pixels = NOT_MODIFIED;

    __syncthreads();

    if(s_slice[sliceIdx] == POSSIBLE_EDGE){

      /// edge = 1 if at last one pixel's neighbour is a definitive edge 
      /// and edge = 0 if doesn't
      edge = (!((s_slice[sliceIdx-SLICE_WIDTH-1] != DEFINITIVE_EDGE) *
                (s_slice[sliceIdx-SLICE_WIDTH] != DEFINITIVE_EDGE) *
                (s_slice[sliceIdx-SLICE_WIDTH+1] != DEFINITIVE_EDGE) *
                (s_slice[sliceIdx-1] != DEFINITIVE_EDGE) *
                (s_slice[sliceIdx+1] != DEFINITIVE_EDGE) *
                (s_slice[sliceIdx+SLICE_WIDTH-1] != DEFINITIVE_EDGE) *
                (s_slice[sliceIdx+SLICE_WIDTH] != DEFINITIVE_EDGE) *
                (s_slice[sliceIdx+SLICE_WIDTH+1] != DEFINITIVE_EDGE)));
      if ( (edge)*(!modified_block_pixels)  ) modified_image_pixels = modified_block_pixels = MODIFIED;
      s_slice[sliceIdx] = POSSIBLE_EDGE + ((edge)*(POSSIBLE_EDGE-1));

    }

    __syncthreads();


  }while(modified_block_pixels);// end inner loop

  // Update only the blocks that have modified pixels
  if (modified_image_pixels) {
    hys_img[pixIdx] = s_slice[sliceIdx];
  }
  if (!threadIdx.x) have_modified_pixels[0] += modified_image_pixels;

}

float* cudaHysteresis(float *d_img, float *d_mag, int width, int height, const unsigned int t1, const unsigned int t2){
 
  int3 size;
  size.x = width;
  size.y = height;
  size.z = width*height;

  int blocksPerGrid = (size.z + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK;
  dim3 DimBlock(THREADS_PER_BLOCK,1,1);
  dim3 DimGrid(blocksPerGrid,1,1);

  float *d_hys;
  cudaMalloc((void**) &d_hys, (size.z*sizeof(float)));
  CUT_CHECK_ERROR("Memory hysteresis image creation failed");

  /// bind a texture to the image
  cudaBindTexture (NULL ,texRef, d_img);
  cudaBindTexture (NULL ,mag_texRef, d_mag);
  CUT_CHECK_ERROR("Texture bind failed");

  /// host side settable texture attributes
  texRef.normalized = false;
  texRef.filterMode = cudaFilterModePoint;
 
  hysteresisPreparation_kernel<<<DimGrid,DimBlock>>>(d_hys, size, t1, t2);

  /// free allocated memory
  cudaUnbindTexture(texRef);
  cudaUnbindTexture(mag_texRef);
  CUT_CHECK_ERROR("Memory unbind failed");

  //??? - do not remove
  int *tt;
  cudaMalloc((void**) &tt, (blocksPerGrid*sizeof(int)));

  // counter of modifications
  int *modif;
  cudaMalloc((void**) &modif, (sizeof(int)));

  int cont[1];
  do{
    kernel_hysteresis_glm1D<<<DimGrid,DimBlock>>>(d_hys, size, modif);
    CUT_CHECK_ERROR("Hysteresis Kernel failed");
    cudaMemcpy(cont,modif,sizeof(int),cudaMemcpyDeviceToHost);
  }while(cont[0]);

  /// bind a texture to the CUDA array
  cudaBindTexture (NULL ,hysTexRef, d_hys);
  CUT_CHECK_ERROR("Texture bind failed");

  /// host side settable texture attributes
  hysTexRef.normalized = false;
  hysTexRef.filterMode = cudaFilterModePoint;

  hysteresisWrite_kernel<<<DimGrid,DimBlock>>>(d_img, size);
  CUT_CHECK_ERROR("Hysteresis Write failed");

  /// free allocated memory
  cudaUnbindTexture(hysTexRef);
  CUT_CHECK_ERROR("Memory unbind failed");

  cudaFree(d_hys);
  cudaFree(modif);
  cudaFree(tt);
  CUT_CHECK_ERROR("Memory free failed");
  
  float *d_edges;
  d_edges = d_img;
  return(d_edges);

}

