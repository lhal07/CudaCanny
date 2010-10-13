/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkDiscreteGaussianImageFilter.h,v $
  Language:  C++
  Date:      $Date: 2009-04-25 12:27:21 $
  Version:   $Revision: 1.41 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCudaDiscreteGaussianImageFilter_h
#define __itkCudaDiscreteGaussianImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"

#include "cuda.h"
#include "CudaDiscreteGaussian.h"

namespace itk
{
/**
 * \class DiscreteGaussianImageFilter
 * \brief Blurs an image by separable convolution with discrete gaussian kernels.
 * This filter performs Gaussian blurring by separable convolution of an image
 * and a discrete Gaussian operator (kernel).
 *
 * The Gaussian operator used here was described by Tony Lindeberg (Discrete
 * Scale-Space Theory and the Scale-Space Primal Sketch.  Dissertation. Royal
 * Institute of Technology, Stockholm, Sweden. May 1991.) The Gaussian kernel
 * used here was designed so that smoothing and derivative operations commute
 * after discretization.
 *
 * The variance or standard deviation (sigma) will be evaluated as pixel units
 * if SetUseImageSpacing is off (false) or as physical units if
 * SetUseImageSpacing is on (true, default). The variance can be set
 * independently in each dimension.
 *
 * When the Gaussian kernel is small, this filter tends to run faster than
 * itk::RecursiveGaussianImageFilter.
 * 
 * \sa GaussianOperator
 * \sa Image
 * \sa Neighborhood
 * \sa NeighborhoodOperator
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

  /** Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TOutputImage::ImageDimension);
  
  /** The variance for the discrete Gaussian kernel.  Sets the variance
   * independently for each dimension, but 
   * see also SetVariance(const double v). The default is 0.0 in each
   * dimension. If UseImageSpacing is true, the units are the physical units
   * of your image.  If UseImageSpacing is false then the units are
   * pixels. */
  itkSetMacro(Variance, float);
  itkGetConstMacro(Variance, const float);

  /** Set the kernel to be no wider than MaximumKernelWidth pixels,
   *  even if MaximumError demands it. The default is 32 pixels. */
  itkGetConstMacro(MaximumKernelWidth, unsigned int);
  itkSetMacro(MaximumKernelWidth, unsigned int);

  /** Set the number of dimensions to smooth. Defaults to the image
   * dimension. Can be set to less than ImageDimension, smoothing all
   * the dimensions less than FilterDimensionality.  For instance, to
   * smooth the slices of a volume without smoothing in Z, set the
   * FilterDimensionality to 2. */
  itkGetConstMacro(FilterDimensionality, unsigned int);
  itkSetMacro(FilterDimensionality, unsigned int);
  
  /** Convenience Set methods for setting all dimensional parameters
   *  to the same values. */
//  void SetVariance (const float v)
//    {    m_Variance = v; }

  /** Sets a limit for growth of the kernel.  Small maximum error values with
   *  large variances will yield very large kernel sizes.  This value can be
   *  used to truncate a kernel in such instances.  A warning will be given on
   *  truncation of the kernel. */
//  void SetMaximumKernelWidth( unsigned int n )
//        {    m_MaximumKernelWidth = n; }

  /** DiscreteGaussianImageFilter needs a larger input requested region
   * than the output requested region (larger by the size of the
   * Gaussian kernel).  As such, DiscreteGaussianImageFilter needs to
   * provide an implementation for GenerateInputRequestedRegion() in
   * order to inform the pipeline execution model.
   * \sa ImageToImageFilter::GenerateInputRequestedRegion() */
  virtual void GenerateInputRequestedRegion() throw(InvalidRequestedRegionError);

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
    }
  virtual ~CudaDiscreteGaussianImageFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Standard pipeline method. While this class does not implement a
   * ThreadedGenerateData(), its GenerateData() delegates all
   * calculations to an NeighborhoodOperatorImageFilter.  Since the
   * NeighborhoodOperatorImageFilter is multithreaded, this filter is
   * multithreaded by default. */
  void GenerateData();

  
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

};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCudaDiscreteGaussianImageFilter.txx"
#endif

#endif
