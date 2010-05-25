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
) 

set(CUDA_NVCC_FLAGS 
  ${CUDA_NVCC_FLAGS}
#  --device-emulation
  -use_fast_math
)

CUDA_ADD_EXECUTABLE(../bin/cudacanny
  cudacanny.cxx
  canny.cu
)

TARGET_LINK_LIBRARIES(../bin/cudacanny 
  ITKCommon
  ITKIO
  /home/ppginf/luis/cuda/lib/libcutil.a
  /home/ppginf/luis/cuda/common/lib/linux/libcudpp.a
)

CUDA_BUILD_CLEAN_TARGET()

