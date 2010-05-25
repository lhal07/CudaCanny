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
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

extern "C"
void cudaCanny(unsigned char *input, int width, int height, const float gaussianVariance, const unsigned int maxKernelWidth, const unsigned int T1, const unsigned int T2);

namespace itk
{

/** \class CudaCannyFilter
 *
 * This filter is an implementation of a Canny edge detector in CUDA;
 * There are four major steps used in the edge-detection scheme:
 * (1) Smooth the input image with Gaussian filter.
 * (2) Calculate the second directional derivatives of the smoothed image. 
 * (3) Non-Maximum Suppression: the zero-crossings of 2nd derivative are found,
 *     and the sign of third derivative is used to find the correct extrema. 
 * (4) The hysteresis thresholding is applied to the gradient magnitude
 *      (multiplied with zero-crossings) of the smoothed image to find and 
 *      link edges.
 *
 */
template<class TImage> 
class CudaCannyFilter
{
public:
  /** Standard Self typedef. */
  typedef CudaCannyFilter Self;
  
  /** Image typedef support   */
  typedef TImage  InputImageType;
      
  /** Define pixel types. */
  typedef typename TImage::PixelType   ImgPixelType;
  typedef typename TImage::IndexType   IndexType;


private:
  ImgPixelType* image;
  int width;
  int height;
  float gaussianVariance;
  unsigned int maxKernelWidth;
  unsigned int Th;
  unsigned int Tl;

  void cudaCanny(ImgPixelType *image, int width, int height, const float gaussianVariance, const unsigned int maxKernelWidth, const unsigned int T1, const unsigned int T2);


public:
  /** Set the image used by the algorithm */
  void SetInput(ImgPixelType* input){
    image = input;
  }

  /** Set the image dimensions used by the algorithm */
  void SetSize(int w, int h){
    width = w;
    height = h;
  }

  /** Set the Variance parameter used by the Gaussian smoothing filter */
  void SetVariance(float v){
    gaussianVariance = v;
    printf("%d \n",v);
  }
 
  /** Set the Maximum Gaussian Kernel Width parameter used by the Gaussian 
   * smoothing filter */
  void SetMaxKernelWidth(unsigned int kw){
    maxKernelWidth = kw;
  }

  /** Set the Two Thresholds parameters used by the Hysteresis */
  void SetThreshold(unsigned int T2, unsigned int T1){
    Th = T2;
    Tl = T1;
  }

  /** Set the Variance parameter used by the Gaussian smoothing filter */
  void Update(){
    cudaCanny(image,width,height,gaussianVariance,maxKernelWidth,Tl,Th);
  }



};

} // end of namespace
