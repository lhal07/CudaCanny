/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkCudaCannyEdgeDetectionImageFilter.txx,v $
  Language:  C++
  Date:      $Date: 2010-09-15 12:01:33 $
  Version:   $Revision: 3.0.0 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCudaCannyEdgeDetectionImageFilter_txx
#define __itkCudaCannyEdgeDetectionImageFilter_txx
#include "itkCudaCannyEdgeDetectionImageFilter.h"

namespace itk
{
  
template <class TInputImage, class TOutputImage>
CudaCannyEdgeDetectionImageFilter<TInputImage, TOutputImage>::
CudaCannyEdgeDetectionImageFilter()
{
  
  m_UpperThreshold = NumericTraits<OutputImagePixelType>::Zero;
  m_LowerThreshold = NumericTraits<OutputImagePixelType>::Zero;

  m_CudaGaussianFilter = CudaGaussianImageFilterType::New();
  m_CudaSobelFilter = CudaSobelImageFilterType::New();

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

template <class TInputImage, class TOutputImage>
void 
CudaCannyEdgeDetectionImageFilter<TInputImage,TOutputImage>
::SetVariance(float v)
{
  // call the class' implementation of this method
  m_CudaGaussianFilter->SetVariance(v);
  return;
}

template <class TInputImage, class TOutputImage>
const float 
CudaCannyEdgeDetectionImageFilter<TInputImage,TOutputImage>
::GetVariance()
{
  // call the class' implementation of this method
  return m_CudaGaussianFilter->GetVariance();
}

template <class TInputImage, class TOutputImage>
void 
CudaCannyEdgeDetectionImageFilter<TInputImage,TOutputImage>
::SetMaximumKernelWidth(unsigned int k)
{
  // call the class' implementation of this method
  m_CudaGaussianFilter->SetMaximumKernelWidth(k);
  return;
}

template <class TInputImage, class TOutputImage>
const unsigned int 
CudaCannyEdgeDetectionImageFilter<TInputImage,TOutputImage>
::GetMaximumKernelWidth()
{
  // call the class' implementation of this method
  return m_CudaGaussianFilter->GetMaximumKernelWidth();
}

template <class TInputImage, class TOutputImage>
void 
CudaCannyEdgeDetectionImageFilter<TInputImage,TOutputImage>
::CudaNonMaximumSupression()
{

  // Set input, output and temporary pointers
  typename OutputImageType::Pointer input  = m_CudaSobelFilter->GetGradientMagnitude();
  typename OutputImageType::Pointer output = this->GetOutput();
  typename TOutputImage::PixelType * tmpNMS;

  // Get image size
  typename OutputImageType::SizeType size = output->GetLargestPossibleRegion().GetSize();

  // Call CudaNMS. Defined on CudaCannyEdgeDetection.cu
  tmpNMS = cudaGradientMaximumDetector(input->GetDevicePointer(), m_CudaSobelFilter->GetGradientDirection()->GetDevicePointer(), size[0], size[1]);

  // Set NMS pointer on the output image
  output->GetPixelContainer()->SetDevicePointer(tmpNMS, size[0]*size[1], true);
}

template <class TInputImage, class TOutputImage>
void 
CudaCannyEdgeDetectionImageFilter<TInputImage,TOutputImage>
::CudaHysteresisThresholding()
{

  // Get the image ponter
  typename OutputImageType::Pointer output = this->GetOutput();

  // Get image size
  typename OutputImageType::SizeType size = output->GetLargestPossibleRegion().GetSize();

  // Call CudaHysteresis. Defined on CudaCannyEdgeDetection.cu
  cudaHysteresis(output->GetDevicePointer(), size[0], size[1], this->GetLowerThreshold(), this->GetUpperThreshold());

}

template< class TInputImage, class TOutputImage >
void
CudaCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::GenerateData()
{

  // Get input and output image pointers
  typename InputImageType::ConstPointer input = this->GetInput();
  typename OutputImageType::Pointer output = this->GetOutput();
 
  // Allocate output image object
  output->SetBufferedRegion(output->GetRequestedRegion());
  
  // 1.Apply the Gaussian Filter to the input image.-------
  m_CudaGaussianFilter->SetInput(input);
  
  unsigned int timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  /// Start timer

  m_CudaGaussianFilter->Update();

  cutStopTimer( timer );  /// Stop timer
  printf("Gaussian time = %f ms\n",cutGetTimerValue( timer ));

  // 2.Apply the Sobel Filter to detect edges on the image.-------
  m_CudaSobelFilter->SetInput(m_CudaGaussianFilter->GetOutput());
  
  timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  ///< Start timer

  m_CudaSobelFilter->Update();

  cutStopTimer( timer );  ///< Stop timer
  printf("Sobel time = %f ms\n",cutGetTimerValue( timer ));

  // 3. Apply NonMaximumSupression operation on the gradient edges. -------
  timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  ///< Start timer

  this->CudaNonMaximumSupression();

  cutStopTimer( timer );  ///< Stop timer
  printf("Maximum Detector time = %f ms\n",cutGetTimerValue( timer ));

  // 4. Apply Hysteresis Thresholding on the Maximum Values of gradient. -------
  timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  ///< Start timer

  this->CudaHysteresisThresholding();

  cutStopTimer( timer );  ///< Stop timer
  printf("Hysteresis time = %f ms\n",cutGetTimerValue( timer ));

}

template <class TInputImage, class TOutputImage>
void 
CudaCannyEdgeDetectionImageFilter<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "UpperThreshold: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_UpperThreshold)
     << std::endl;
  os << indent << "LowerThreshold: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_LowerThreshold)
     << std::endl;
}

}//end of itk namespace
#endif
