/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkCuda2DSeparableConvolutionImageFilter.h,v $
  Language:  C++
  Date:      $Date: 2010-11-11 20:58:48 $
  Version:   $Revision: 0.1 $

  Copyright ( c ) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCuda2DSeparableConvolutionImageFilter_h
#define __itkCuda2DSeparableConvolutionImageFilter_h

#include "itkImageToImageFilter.h"

#include "cuda.h"
#include "Cuda2DSeparableConvolution.h"

namespace itk {

template<class TInputImage, class TOutputImage>
class ITK_EXPORT Cuda2DSeparableConvolutionImageFilter
: public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  typedef Cuda2DSeparableConvolutionImageFilter                Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage>        Superclass;
  typedef SmartPointer<Self>                                   Pointer;
  typedef SmartPointer<const Self>                             ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information ( and related methods ) */
  itkTypeMacro( Cuda2DSeparableConvolutionImageFilter, ImageToImageFilter );

  /** Dimensionality of input and output data is assumed to be the same. */
  itkStaticConstMacro( ImageDimension, unsigned int,
                       TInputImage::ImageDimension );


  typedef TInputImage                                    InputImageType;
  typedef TOutputImage                                   OutputImageType;
  typedef typename InputImageType::PixelType             InputPixelType;
  typedef typename OutputImageType::PixelType            OutputPixelType;
  typedef typename OutputImageType::RegionType           OutputRegionType;

  /** Cuda2DSeparableConvolutionImageFilter needs two smaller extra inputs 
   * (the image kernels to be aplyed on each direction)
   * requested region than output requested region.  As such, this filter
   * needs to provide an implementation for GenerateInputRequestedRegion() in
   * order to inform the pipeline execution model.
   * \sa ProcessObject::GenerateInputRequestedRegion()  */
//  virtual void GenerateInputRequestedRegion();

  void SetInputMaskHorizontal( OutputPixelType * , unsigned int);
  const OutputPixelType * GetInputMaskHorizontal( void );
  void SetInputMaskVertical( OutputPixelType * , unsigned int);
  const OutputPixelType * GetInputMaskVertical( void );

protected:
  /** de/constructor */
  Cuda2DSeparableConvolutionImageFilter();
  ~Cuda2DSeparableConvolutionImageFilter();

  void GenerateData();

private:
  Cuda2DSeparableConvolutionImageFilter( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented

private:

  OutputPixelType * m_Mask1;
  unsigned int      m_SizeMask1;
  OutputPixelType * m_Mask2;
  unsigned int      m_SizeMask2;

};
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCuda2DSeparableConvolutionImageFilter.txx"
#endif

#endif
