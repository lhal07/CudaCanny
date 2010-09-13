/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkCannyEdgeDetectionImageFilter.txx,v $
  Language:  C++
  Date:      $Date: 2009-08-17 12:01:33 $
  Version:   $Revision: 1.56 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCudaCannyEdgeDetectionImageFilter_txx
#define __itkCudaCannyEdgeDetectionImageFilter_txx
#include "itkCudaCannyEdgeDetectionImageFilter.h"

#include "itkZeroCrossingImageFilter.h"
#include "itkNeighborhoodInnerProduct.h"
#include "itkNumericTraits.h"
#include "itkProgressReporter.h"
#include "itkGradientMagnitudeImageFilter.h"

namespace itk
{
  
template <class TInputImage, class TOutputImage>
CudaCannyEdgeDetectionImageFilter<TInputImage, TOutputImage>::
CudaCannyEdgeDetectionImageFilter()
{
  
  m_Variance = NumericTraits<OutputImagePixelType>::Zero;
  m_OutsideValue = NumericTraits<OutputImagePixelType>::Zero;
  m_Threshold = NumericTraits<OutputImagePixelType>::Zero;
  m_UpperThreshold = NumericTraits<OutputImagePixelType>::Zero;
  m_LowerThreshold = NumericTraits<OutputImagePixelType>::Zero;

}
 
template <class TInputImage, class TOutputImage>
void 
CudaCannyEdgeDetectionImageFilter<TInputImage,TOutputImage>
::GenerateInputRequestedRegion()
{
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();
  return;  
}

template< class TInputImage, class TOutputImage >
void
CudaCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::GenerateData()
{
  // Get input and output image pointers
  typename InputImageType::ConstPointer input = this->GetInput();
  typename OutputImageType::Pointer output = this->GetOutput();
  typename OutputImageType::PixelType * ptr;

  // Allocate output image object
  typename OutputImageType::RegionType outputRegion;
  outputRegion.SetSize(input->GetLargestPossibleRegion().GetSize());
  outputRegion.SetIndex(input->GetLargestPossibleRegion().GetIndex());
  output->SetRegions(outputRegion);
  output->Allocate();

  // Get image size
  typename OutputImageType::SizeType size;
  size = output->GetLargestPossibleRegion().GetSize();

  // Call cudaCanny. Defined on canny.cu
  ptr = cudaCanny(input->GetDevicePointer(), size[0], size[1], (float) m_Variance, m_MaximumKernelWidth, m_LowerThreshold, m_UpperThreshold);

  // Set image pointer to the output image
  output->GetPixelContainer()->SetDevicePointer(ptr, size[0]*size[1], true);

}

template <class TInputImage, class TOutputImage>
void 
CudaCannyEdgeDetectionImageFilter<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Variance: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_Variance)
     << std::endl;
  os << indent << "Threshold: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_Threshold)
     << std::endl;
  os << indent << "UpperThreshold: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_UpperThreshold)
     << std::endl;
  os << indent << "LowerThreshold: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_LowerThreshold)
     << std::endl;
  os << indent << "OutsideValue: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_OutsideValue)
     << std::endl;
}

}//end of itk namespace
#endif
