CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  /home/ppginf/luis/cuda/common/inc
  /home/ppginf/luis/cuda/common/inc/cudpp
  /home/ppginf/luis/cuda/include
  /home/ppginf/luis/cuda/lib
  /home/ppginf/luis/cuda/common/lib
)

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  /home/ppginf/luis/cuda/common/inc
  /home/ppginf/luis/cuda/common/inc/cudpp
  /home/ppginf/luis/cuda/include
  /home/ppginf/luis/cuda/lib
  /home/ppginf/luis/cuda/common/lib
) 

set(CUDA_NVCC_FLAGS 
  ${CUDA_NVCC_FLAGS}
#  --device-emulation
#  -cubin
  --ptxas-options=-v
  -use_fast_math
)

CUDA_ADD_EXECUTABLE(../bin/cudacanny
  itkCudaCannyEdgeDetectionImageFilter.txx
  itkCudaDiscreteGaussianImageFilter.txx
  itkCudaSobelEdgeDetectionImageFilter.txx
  CudaCannyEdgeDetection.cu
  CudaDiscreteGaussian.cu
  CudaSobelEdgeDetection.cu
  cudacanny.cxx
)

TARGET_LINK_LIBRARIES(../bin/cudacanny 
  ITKCommon
  ITKIO
  /home/ppginf/luis/cuda/lib/libcutil.a
  /home/ppginf/luis/cuda/common/lib/linux/libcudpp.a
)

CUDA_BUILD_CLEAN_TARGET()

