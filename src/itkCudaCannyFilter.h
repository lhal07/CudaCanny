///cudacanny.h
/**
 * \author Luis Lourenço (2010)
 * \version 2.0.0
 * \since 20/05/10
 */


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

public:

  void SetInput(ImgPixelType* input);
  void SetSize(int w, int h);
  void SetVariance(float v);
  void SetMaxKernelWidth(unsigned int kw);
  void SetThreshold(unsigned int T2, unsigned int T1);
  void Update();

};

} // end of namespace
