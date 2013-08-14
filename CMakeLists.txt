# Note set CUDA_BUILD_EMULATION to OFF to enable Device mode compilation.

CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2)

PROJECT(cudacanny)

set(CUDA_TOOLKIT_ROOT_DIR 
/home/ppginf/luis/cuda
)

# find_package(CUDA)
FIND_PACKAGE(CUDA QUIET REQUIRED)

#TARGET_LINK_LIBRARIES(
#test ${CUDA_CUT_INCLUDE}
#)

set(ITK_DIR
/home/ppginf/luis/VRI/MyCITK/usr/local/lib/InsightToolkit
)

FIND_PACKAGE(ITK REQUIRED)

INCLUDE(${ITK_USE_FILE})

SUBDIRS(
  src
  )
