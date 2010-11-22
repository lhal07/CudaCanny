/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkCudaCannyEdgeDetectionImageFilter.h,v $
  Language:  C++
  Date:      $Date: 2010-09-15 12:27:15 $
  Version:   $Revision: 3.0.0 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCudaCannyEdgeDetectionImageFilter_h
#define __itkCudaCannyEdgeDetectionImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "itkCudaKernelConfigurator.h"
#include "itkCudaDiscreteGaussianImageFilter.h"
#include "itkCudaZeroCrossingImageFilter.h"

#include "cuda.h"
#include "CudaCannyEdgeDetection.h"

namespace itk
{

/** \class CannyEdgeDetectionImageFilter
 *
 * This filter is an implementation of a Canny edge detector for scalar-valued
 * images.  Based on John Canny's paper "A Computational Approach to Edge 
 * Detection"(IEEE Transactions on Pattern Analysis and Machine Intelligence, 
 * Vol. PAMI-8, No.6, November 1986),  there are four major steps used in the 
 * edge-detection scheme:
 * (1) Smooth the input image with Gaussian filter.
 * (2) Calculate the second directional derivatives of the smoothed image. 
 * (3) Non-Maximum Suppression: the zero-crossings of 2nd derivative are found,
 *     and the sign of third derivative is used to find the correct extrema. 
 * (4) The hysteresis thresholding is applied to the gradient magnitude
 *      (multiplied with zero-crossings) of the smoothed image to find and 
 *      link edges.
 *
 * \par Inputs and Outputs
 * The input to this filter should be a scalar, real-valued Itk image of
 * arbitrary dimension.  The output should also be a scalar, real-value Itk
 * image of the same dimensionality.
 *
 * \par Parameters
 * There are four parameters for this filter that control the sub-filters used
 * by the algorithm.
 *
 * \par 
 * Variance and Maximum error are used in the Gaussian smoothing of the input
 * image.  See  itkDiscreteGaussianImageFilter for information on these
 * parameters.
 *
 * \par
 * Threshold is the lowest allowed value in the output image.  Its data type is 
 * the same as the data type of the output image. Any values below the
 * Threshold level will be replaced with the OutsideValue parameter value, whose
 * default is zero.
 */
template<class TInputImage, class TOutputImage>
class ITK_EXPORT CudaCannyEdgeDetectionImageFilter
  : public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard "Self" & Superclass typedef.  */
  typedef CudaCannyEdgeDetectionImageFilter                 Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
   
  /** Image typedef support   */
  typedef TInputImage  InputImageType;
  typedef TOutputImage OutputImageType;
      
  /** SmartPointer typedef support  */
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Define pixel types. */
  typedef typename TInputImage::PixelType   InputImagePixelType;
  typedef typename TOutputImage::PixelType  OutputImagePixelType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);  
    
  /** Typedef to describe the output image region type. */
  typedef typename TOutputImage::RegionType OutputImageRegionType;
  typedef typename TInputImage::RegionType  InputImageRegionType;
    
  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaCannyEdgeDetectionImageFilter, ImageToImageFilter);
  
  /** ImageDimension constant    */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      TOutputImage::ImageDimension);
  
  /** Standard get/set macros for filter parameters. */

  /** Set/Get the Variance parameter used by the Gaussian smoothing
      filter in this algorithm */
  virtual void SetVariance(float);
  virtual const float GetVariance();

  /** Sets a limit for growth of the kernel.  Small maximum error values with
   *  large variances will yield very large kernel sizes.  This value can be
   *  used to truncate a kernel in such instances.  A warning will be given on
   *  truncation of the kernel. */
  virtual void SetMaximumKernelWidth(unsigned int);
  virtual const unsigned int GetMaximumKernelWidth();

  ///* Set the Threshold value for detected edges. */
  itkSetMacro(UpperThreshold, OutputImagePixelType );
  itkGetConstMacro(UpperThreshold, OutputImagePixelType);
  itkSetMacro(LowerThreshold, OutputImagePixelType );
  itkGetConstMacro(LowerThreshold, OutputImagePixelType);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputHasNumericTraitsCheck,
    (Concept::HasNumericTraits<InputImagePixelType>));
  itkConceptMacro(OutputHasNumericTraitsCheck,
    (Concept::HasNumericTraits<OutputImagePixelType>));
  itkConceptMacro(SameDimensionCheck,
    (Concept::SameDimension<ImageDimension, OutputImageDimension>));
  itkConceptMacro(InputIsFloatingPointCheck,
    (Concept::IsFloatingPoint<InputImagePixelType>));
  itkConceptMacro(OutputIsFloatingPointCheck,
    (Concept::IsFloatingPoint<OutputImagePixelType>));
  /** End concept checking */
#endif

protected:
  CudaCannyEdgeDetectionImageFilter();
  CudaCannyEdgeDetectionImageFilter(const Self&) {}
  void PrintSelf(std::ostream& os, Indent indent) const;

  void GenerateData();

  typedef CudaDiscreteGaussianImageFilter<InputImageType, OutputImageType>
                                                      CudaGaussianImageFilterType;

  typedef CudaZeroCrossingImageFilter<OutputImageType, OutputImageType>
                                                      CudaZeroCrossingFilterType;

  typedef CudaKernelConfigurator CudaKernelConfiguratorType;

private:
  virtual ~CudaCannyEdgeDetectionImageFilter(){};

  /** Implementation of 2md Derivative and Magnitude calculation on Cuda*/
  void Cuda2ndDerivative();

  /** Implementation of Hysteresis Thresholding on Cuda */
  void CudaHysteresisThresholding();

  /** Upper threshold value for identifying edges. */
  OutputImagePixelType m_UpperThreshold;
  
  /** Lower threshold value for identifying edges. */
  OutputImagePixelType m_LowerThreshold;

  /** Update buffers used during calculationsof multiple steps */
  typename OutputImageType::Pointer m_UpdateBuffer1;

  /** CudaGaussian filter to smooth the input image  */
  typename CudaGaussianImageFilterType::Pointer m_CudaGaussianFilter;
 
  /** CudaZeroCrossing filter to detect zero crossings on the 2nd derivative
   * image */
  typename CudaZeroCrossingFilterType::Pointer m_CudaZeroCrossingFilter;

  typename CudaKernelConfiguratorType::Pointer m_CudaConf;
};

} //end of namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCudaCannyEdgeDetectionImageFilter.txx"
#endif
  
#endif
