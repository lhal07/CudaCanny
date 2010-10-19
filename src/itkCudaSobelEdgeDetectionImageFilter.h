/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkCudaSobelEdgeDetectionImageFilter.h,v $
  Language:  C++
  Date:      $Date: 2010-09-15 20:49:56 $
  Version:   $Revision: 3.0.0 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCudaSobelEdgeDetectionImageFilter_h
#define __itkCudaSobelEdgeDetectionImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"

#include "cuda.h"
#include "CudaSobelEdgeDetection.h"

namespace itk
{

/**
 * \class SobelEdgeDetectionImageFilter
 * \brief A 2D edge detection using the Sobel operator implemented in CUDA.
 *
 * This filter uses the Sobel operator to calculate the image gradient and then
 * finds the magnitude and direction of this gradient vector.  The Sobel gradient 
 * magnitude (square-root sum of squares) is an indication of edge strength.
 * 
 * \sa ImageToImageFilter
 * 
 * \ingroup ImageFeatureExtraction 
 *
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT CudaSobelEdgeDetectionImageFilter : 
    public ImageToImageFilter< TInputImage, TOutputImage > 
{
public:
  /**
   * Standard "Self" & Superclass typedef.
   */
  typedef CudaSobelEdgeDetectionImageFilter                   Self;
  typedef ImageToImageFilter< TInputImage, TOutputImage > Superclass;

  /**
   * Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same.
   */
  typedef typename TOutputImage::PixelType         OutputPixelType;
  typedef typename TOutputImage::InternalPixelType OutputInternalPixelType;
  typedef typename TInputImage::PixelType          InputPixelType;
  typedef typename TInputImage::InternalPixelType  InputInternalPixelType;
  typedef float                                    DirectionPixelType;
  typedef itk::Image<DirectionPixelType,2>         DirectionType;
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TOutputImage::ImageDimension );
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      TInputImage::ImageDimension );
  
  /**
   * Image typedef support
   */
  typedef TInputImage                      InputImageType;
  typedef TOutputImage                     OutputImageType;
  typedef typename InputImageType::Pointer InputImagePointer;

  /** 
   * Smart pointer typedef support 
   */
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;
  
  /**
   * Run-time type information (and related methods)
   */
  itkTypeMacro(CudaSobelEdgeDetectionImageFilter, ImageToImageFilter);
  
  /**
   * Method for creation through the object factory.
   */
  itkNewMacro(Self);

  /** Method for getting the gradient direction of each pixel */
  DirectionType* GetGradientDirection();

  /** Method for getting the gradient strenght/magnitude of each pixel */
  OutputImageType* GetGradientMagnitude();
  
  /**
   * SobelEdgeDetectionImageFilter needs a larger input requested region than
   * the output requested region (larger in the direction of the
   * derivative).  As such, SobelEdgeDetectionImageFilter needs to provide an
   * implementation for GenerateInputRequestedRegion() in order to
   * inform the pipeline execution model.
   *
   * \sa ImageToImageFilter::GenerateInputRequestedRegion()
   */
  virtual void GenerateInputRequestedRegion() throw(InvalidRequestedRegionError);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(SameDimensionCheck,
    (Concept::SameDimension<InputImageDimension, ImageDimension>));
  itkConceptMacro(OutputHasNumericTraitsCheck,
    (Concept::HasNumericTraits<OutputPixelType>));

#ifdef ITK_USE_STRICT_CONCEPT_CHECKING
  itkConceptMacro(OutputPixelIsFloatingPointCheck,
    (Concept::IsFloatingPoint<OutputPixelType>));
#endif

  /** End concept checking */
#endif

protected:
  CudaSobelEdgeDetectionImageFilter();
  virtual ~CudaSobelEdgeDetectionImageFilter() {}
  CudaSobelEdgeDetectionImageFilter(const Self&) {}
  //void operator=(const Self&) {}

  /**
   * Standard pipeline method. This class does not implement a
   * ThreadedGenerateData(), its GenerateData() call an CUDA function to
   * paralelize the operation.
   */
  void GenerateData();
  void PrintSelf(std::ostream& os, Indent indent) const
  { Superclass::PrintSelf(os,indent); }

  
private:

  /** GPU structure to the Gradient of the Image. 
   * Tgrad is defined on CudaSobelEdgeDetection.cu */
  Tgrad m_gradient;

};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCudaSobelEdgeDetectionImageFilter.txx"
#endif

#endif
