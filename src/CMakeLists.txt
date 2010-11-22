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
  itkCudaKernelConfigurator.txx
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
  /home/ppginf/luis/cuda/lib/libcutil.a
)

CUDA_BUILD_CLEAN_TARGET()

