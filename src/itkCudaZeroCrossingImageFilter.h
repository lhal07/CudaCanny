/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkCudaZeroCrossingImageFilter.h,v $
  Language:  C++
  Date:      $Date: 2010-11-04 12:28:13 $
  Version:   $Revision: 0.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCudaZeroCrossingImageFilter_h
#define __itkCudaZeroCrossingImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"

#include "cuda.h"
#include "CudaZeroCrossing.h"

namespace itk
{  
/** \class CudaZeroCrossingImageFilter
 *
 *  This filter finds the closest pixel to the zero-crossings (sign changes) in
 *  a signed itk::Image.  Pixels closest to zero-crossings are labeled with
 *  a foreground value.  All other pixels are marked with a background value.
 *  The algorithm works by detecting differences in sign among neighbors using
 *  city-block style connectivity (4-neighbors in 2d, 6-neighbors in 3d, etc.).
 *  
 *  \par Inputs and Outputs
 *  The input to this filter is an itk::Image of arbitrary dimension.  The
 *  algorithm assumes a signed data type (zero-crossings are not defined for
 *  unsigned data types), and requires that operator>, operator<, operator==,
 *  and operator!= are defined.  
 *
 *  \par
 *  The output of the filter is a binary, labeled image of user-specified type.
 *  By default, zero-crossing pixels are labeled with a default ``foreground''
 *  value of itk::NumericTraits<OutputDataType>::One, where OutputDataType is
 *  the data type of the output image.  All other pixels are labeled with a
 *  default ``background'' value of itk::NumericTraits<OutputDataType>::Zero.
 *
 *  \par Parameters
 *  There are two parameters for this filter.  ForegroundValue is the value
 *  that marks zero-crossing pixels.  The BackgroundValue is the value given to 
 *  all other pixels.
 *
 *  \sa Image
 *  \ingroup ImageFeatureExtraction */
template<class TInputImage, class TOutputImage>
class ITK_EXPORT CudaZeroCrossingImageFilter
  : public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard "Self" & Superclass typedef. */
  typedef CudaZeroCrossingImageFilter                   Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  
  /** Image typedef support   */
  typedef TInputImage  InputImageType;
  typedef TOutputImage OutputImageType;
  
  /** SmartPointer typedef support  */ 
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;
  
  /** Define pixel types  */
  typedef typename TInputImage::PixelType   InputImagePixelType;
  typedef typename TOutputImage::PixelType  OutputImagePixelType;
  
  /** Method for creation through the object factory.  */
  itkNewMacro(Self);  
  
  /** Typedef to describe the output image region type. */
  typedef typename TOutputImage::RegionType OutputImageRegionType;
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaZeroCrossingImageFilter, ImageToImageFilter);
  
  /** ImageDimension enumeration   */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImage::ImageDimension );
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      TOutputImage::ImageDimension );
  
  /** Set/Get the label value for zero-crossing pixels. */
  itkSetMacro(ForegroundValue, OutputImagePixelType);
  itkGetConstMacro(ForegroundValue, OutputImagePixelType);
  
  /** Set/Get the label value for non-zero-crossing pixels. */
  itkSetMacro(BackgroundValue, OutputImagePixelType);
  itkGetConstMacro(BackgroundValue, OutputImagePixelType);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(OutputEqualityComparableCheck,
    (Concept::EqualityComparable<OutputImagePixelType>));
  itkConceptMacro(SameDimensionCheck,
    (Concept::SameDimension<ImageDimension, OutputImageDimension>));
  itkConceptMacro(InputComparableCheck,
    (Concept::Comparable<InputImagePixelType>));
  itkConceptMacro(OutputOStreamWritableCheck,
    (Concept::OStreamWritable<OutputImagePixelType>));
  /** End concept checking */
#endif

protected:
  CudaZeroCrossingImageFilter()
    {
    m_ForegroundValue = NumericTraits<OutputImagePixelType>::One;
    m_BackgroundValue = NumericTraits<OutputImagePixelType>::Zero;
    }
  ~CudaZeroCrossingImageFilter(){}
  void PrintSelf(std::ostream& os, Indent indent) const;
  
  CudaZeroCrossingImageFilter(const Self&) {}
  OutputImagePixelType m_BackgroundValue;
  OutputImagePixelType m_ForegroundValue;
  
  void GenerateData();
};
  
} //end of namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCudaZeroCrossingImageFilter.txx"
#endif
  
#endif
