#set(CUDA_64_BIT_DEVICE_CODE
#  ON
#)

#CUDA3.0
#CUDA_INCLUDE_DIRECTORIES(
#  ${CMAKE_CURRENT_SOURCE_DIR}
#  /home/ppginf/luis/cuda/C/common/inc
#  /home/ppginf/luis/cuda/include
#  /home/ppginf/luis/cuda/lib
#  /home/ppginf/luis/cuda/lib64
#  /home/ppginf/luis/cuda/C/common/lib
#)
#INCLUDE_DIRECTORIES(
#  ${CMAKE_CURRENT_SOURCE_DIR}
#  /home/ppginf/luis/cuda/C/common/inc
#  /home/ppginf/luis/cuda/include
#  /home/ppginf/luis/cuda/lib
#  /home/ppginf/luis/cuda/lib64
#  /home/ppginf/luis/cuda/C/common/lib
#) 

#CUDA2.2
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  /home/ppginf/luis/cuda/common/inc
  /home/ppginf/luis/cuda/include
  /home/ppginf/luis/cuda/lib
  /home/ppginf/luis/cuda/common/lib
)
INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  /home/ppginf/luis/cuda/common/inc
  /home/ppginf/luis/cuda/include
  /home/ppginf/luis/cuda/lib
  /home/ppginf/luis/cuda/common/lib
) 

set(CUDA_NVCC_FLAGS 
  ${CUDA_NVCC_FLAGS}
#  --device-emulation
#  -cubin
#  -gencode=arch=compute_20,code=sm_20
#  -Xptxas -dlcm=ca
  --ptxas-options=-v
  -use_fast_math
#  -lcutil
)

CUDA_ADD_EXECUTABLE(../bin/cudacanny
  itkCudaInterface.txx
  itkCudaCannyEdgeDetectionImageFilter.txx
  itkCudaDiscreteGaussianImageFilter.txx
  itkCudaZeroCrossingImageFilter.txx
  itkCuda2DSeparableConvolutionImageFilter.txx
  CudaCannyEdgeDetection.cu
  CudaDiscreteGaussian.cu
  CudaZeroCrossing.cu
  Cuda2DSeparableConvolution.cu
  cudacanny.cxx
)

TARGET_LINK_LIBRARIES(../bin/cudacanny 
  ITKCommon
  ITKIO
#  /home/ppginf/luis/cuda/C/lib/libcutil_x86_64.a
  /home/ppginf/luis/cuda/lib/libcutil.a
)

CUDA_BUILD_CLEAN_TARGET()

