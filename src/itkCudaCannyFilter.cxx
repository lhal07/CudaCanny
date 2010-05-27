///cudacanny.h
/**
 * \author Luis Lourenço (2010)
 * \version 2.0.0
 * \since 20/05/10
 */

#include <stdio.h>
#include <stdlib.h>
#include "itkImage.h"
#include "itkPNGImageIO.h"
#include "itkCudaCannyFilter.h"

namespace itk{

/** Set the image used by the algorithm */
template<class TInputImage> 
void CudaCannyFilter<TInputImage>::SetInput(ImgPixelType* input){
  image = input;
}

/** Set the image dimensions used by the algorithm */
template<class TInputImage> 
void CudaCannyFilter<TInputImage>::SetSize(int w, int h){
  width = w;
  height = h;
}

/** Set the Variance parameter used by the Gaussian smoothing filter */
template<class TInputImage> 
void CudaCannyFilter<TInputImage>::SetVariance(float v){
  gaussianVariance = v;
  printf("%d \n",v);
}

/** Set the Maximum Gaussian Kernel Width parameter used by the Gaussian 
 * smoothing filter */
template<class TInputImage> 
void CudaCannyFilter<TInputImage>::SetMaxKernelWidth(unsigned int kw){
  maxKernelWidth = kw;
}

/** Set the Two Thresholds parameters used by the Hysteresis */
template<class TInputImage> 
void CudaCannyFilter<TInputImage>::SetThreshold(unsigned int T2, unsigned int T1){
  Th = T2;
  Tl = T1;
}

}
