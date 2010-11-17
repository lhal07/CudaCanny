/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkCudaDiscreteGaussianImageFilter.h,v $
  Language:  C++
  Date:      $Date: 2010-11-10 12:27:21 $
  Version:   $Revision: 1.0 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCudaDiscreteGaussianImageFilter_h
#define __itkCudaDiscreteGaussianImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkCuda2DSeparableConvolutionImageFilter.h"
#include "itkImage.h"

#include "cuda.h"
#include "CudaDiscreteGaussian.h"

namespace itk
{
/**
 * \class CudaDiscreteGaussianImageFilter
 * \brief Blurs an image by separable convolution with discrete gaussian kernels.
 * This filter performs Gaussian blurring by separable convolution of an image
 * and a discrete Gaussian operator (kernel).
 *
 * \sa cuda1DGaussianOperator
 * \sa Image
 * 
 * \ingroup ImageEnhancement 
 * \ingroup ImageFeatureExtraction 
 */

template <class TInputImage, class TOutputImage >
class ITK_EXPORT CudaDiscreteGaussianImageFilter :
    public ImageToImageFilter< TInputImage, TOutputImage > 
{
public:
  /** Standard class typedefs. */
  typedef CudaDiscreteGaussianImageFilter                 Self;
  typedef ImageToImageFilter< TInputImage, TOutputImage > Superclass;
  typedef SmartPointer<Self>                              Pointer;
  typedef SmartPointer<const Self>                        ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaDiscreteGaussianImageFilter, ImageToImageFilter);
  
  /** Image type information. */
  typedef TInputImage  InputImageType;
  typedef TOutputImage OutputImageType;

  /** Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same. */
  typedef typename TOutputImage::PixelType         OutputPixelType;
  typedef typename TOutputImage::InternalPixelType OutputInternalPixelType;
  typedef typename TInputImage::PixelType          InputPixelType;
  typedef typename TInputImage::InternalPixelType  InputInternalPixelType;
  typedef Image<OutputPixelType,1> MaskImageType;

  TOutputImage * GetOutput();

  /** Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TOutputImage::ImageDimension);
  
  /** The variance for the discrete Gaussian kernel. */
  itkSetMacro(Variance, float);
  itkGetConstMacro(Variance, const float);

  /** Set the kernel to be no wider than MaximumKernelWidth pixels,
   *  even if MaximumError demands it. The default is 3 pixels. */
  itkGetConstMacro(MaximumKernelWidth, unsigned int);
  itkSetMacro(MaximumKernelWidth, unsigned int);

  /** Set the number of dimensions to smooth. Defaults to the image
   * dimension. Can be set to less than ImageDimension, smoothing all
   * the dimensions less than FilterDimensionality.  For instance, to
   * smooth the slices of a volume without smoothing in Z, set the
   * FilterDimensionality to 2. */
  itkGetConstMacro(FilterDimensionality, unsigned int);
  itkSetMacro(FilterDimensionality, unsigned int);
  
#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(OutputHasNumericTraitsCheck,
    (Concept::HasNumericTraits<OutputPixelType>));
  /** End concept checking */
#endif

protected:
  CudaDiscreteGaussianImageFilter()
    {
    m_Variance =1.0;
    m_MaximumKernelWidth = 3;
    m_FilterDimensionality = ImageDimension;
    m_CudaConvolutionFilter = Cuda2DSeparableConvolutionImageFilterType::New();
    }
  virtual ~CudaDiscreteGaussianImageFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Standard pipeline method. */
  void GenerateData();

  typedef Cuda2DSeparableConvolutionImageFilter<InputImageType, OutputImageType>
                                                      Cuda2DSeparableConvolutionImageFilterType;
  
private:
  CudaDiscreteGaussianImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  /** The variance of the gaussian blurring kernel in each dimensional direction. */
  float m_Variance;

  /** Maximum allowed kernel width for any dimension of the discrete Gaussian
      approximation */
  unsigned int m_MaximumKernelWidth;

  /** Number of dimensions to process. Default is all dimensions */
  unsigned int m_FilterDimensionality;

  /** Cuda2DSeparableConvolution to convolve the image  */
  typename Cuda2DSeparableConvolutionImageFilterType::Pointer m_CudaConvolutionFilter;
};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCudaDiscreteGaussianImageFilter.txx"
#endif

#endif
