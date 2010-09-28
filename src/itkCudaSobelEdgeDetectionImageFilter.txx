/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkCudaSobelEdgeDetectionImageFilter.txx,v $
  Language:  C++
  Date:      $Date: 2010-09-15 20:50:03 $
  Version:   $Revision: 3.0.0 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCudaSobelEdgeDetectionImageFilter_txx
#define __itkCudaSobelEdgeDetectionImageFilter_txx
#include "itkCudaSobelEdgeDetectionImageFilter.h"


namespace itk
{

template <class TInputImage, class TOutputImage> 
CudaSobelEdgeDetectionImageFilter<TInputImage,TOutputImage>
::CudaSobelEdgeDetectionImageFilter()
{
 
  typename TOutputImage::Pointer strenght   = OutputImageType::New();
  typename DirectionType::Pointer direction = DirectionType::New();

  this->SetNumberOfRequiredOutputs( 2 );
  this->SetNthOutput( 0, strenght.GetPointer() );
  this->SetNthOutput( 1, direction.GetPointer() );

}

template <class TInputImage, class TOutputImage>
void 
CudaSobelEdgeDetectionImageFilter<TInputImage,TOutputImage>
::GenerateInputRequestedRegion() throw (InvalidRequestedRegionError)
{
  // call the superclass' implementation of this method. this should
  // copy the output requested region to the input requested region
  Superclass::GenerateInputRequestedRegion();
  
}

template <class TInputImage, class TOutputImage>
typename CudaSobelEdgeDetectionImageFilter<TInputImage,TOutputImage>::DirectionType * 
CudaSobelEdgeDetectionImageFilter<TInputImage,TOutputImage>
::GetGradientDirection()
{
  return( dynamic_cast<DirectionType *>(this->ProcessObject::GetOutput( 1 )) );
}

template <class TInputImage, class TOutputImage>
TOutputImage *
CudaSobelEdgeDetectionImageFilter<TInputImage,TOutputImage>
::GetGradientStrenght()
{
  return( dynamic_cast<TOutputImage *>(this->ProcessObject::GetOutput( 0 )) );
}

template< class TInputImage, class TOutputImage >
void
CudaSobelEdgeDetectionImageFilter< TInputImage, TOutputImage >
::GenerateData()
{

  // Set input, output and temporary pointers
  typename TInputImage::ConstPointer input = this->GetInput();
  typename TOutputImage::Pointer strenght = this->GetGradientStrenght();
  typename DirectionType::Pointer direction = this->GetGradientDirection();
  Tgrad * tmpGrad;
  
  // Allocate output image object
  strenght->SetBufferedRegion(strenght->GetRequestedRegion());
  strenght->Allocate();
  direction->SetBufferedRegion(direction->GetRequestedRegion());
  direction->Allocate();

  // Get image size
  typename OutputImageType::SizeType size = strenght->GetLargestPossibleRegion().GetSize();

  // Call cudaSobel. Defined on CudaSobelEdgeDetection.cu
  tmpGrad = cudaSobel(&m_gradient, input->GetDevicePointer(), size[0], size[1]);

  // Set image pointer to the output image
  strenght->GetPixelContainer()->SetDevicePointer(tmpGrad->Strenght, size[0]*size[1], true);
  direction->GetPixelContainer()->SetDevicePointer(tmpGrad->Direction, size[0]*size[1], true);

}

} // end namespace itk

#endif
