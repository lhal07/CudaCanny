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

#define THREADS_PER_BLOCK 256 

namespace itk
{
  
template <class TInputImage, class TOutputImage>
CudaCannyEdgeDetectionImageFilter<TInputImage, TOutputImage>::
CudaCannyEdgeDetectionImageFilter()
{

  m_UpperThreshold = NumericTraits<OutputImagePixelType>::Zero;
  m_LowerThreshold = NumericTraits<OutputImagePixelType>::Zero;

  m_CudaGaussianFilter = CudaGaussianImageFilterType::New();
  m_CudaZeroCrossingFilter = CudaZeroCrossingFilterType::New();
  m_CudaConf = CudaInterfaceType::New();

  m_UpdateBuffer1  = OutputImageType::New();

}

template <class TInputImage, class TOutputImage>
void 
CudaCannyEdgeDetectionImageFilter<TInputImage,TOutputImage>
::SetVariance(const typename ArrayType::ValueType v)
{
  // call the class' implementation of this method
  m_CudaGaussianFilter->SetVariance(v);
  return;
}

template <class TInputImage, class TOutputImage>
void 
CudaCannyEdgeDetectionImageFilter<TInputImage,TOutputImage>
::SetMaximumError(const typename ArrayType::ValueType v)
{
  // call the class' implementation of this method
  m_CudaGaussianFilter->SetMaximumError(v);
  return;
}

template <class TInputImage, class TOutputImage>
void 
CudaCannyEdgeDetectionImageFilter<TInputImage,TOutputImage>
::Cuda2ndDerivative()
{

  // Set input, output and temporary pointers
  typename OutputImageType::Pointer input  = m_CudaGaussianFilter->GetOutput();
  typename OutputImageType::Pointer output = this->GetOutput();
  typename OutputImageType::Pointer output1 = m_UpdateBuffer1;
  typename TOutputImage::PixelType * deriv;
  typename TOutputImage::PixelType * mag;

  // Get image size
  typename OutputImageType::SizeType size = output->GetLargestPossibleRegion().GetSize();

  // Call CudaNMS. Defined on CudaCannyEdgeDetection.cu
  deriv = cuda2ndDerivative(m_CudaConf->GetGridDim(),m_CudaConf->GetBlockDim(),input->GetDevicePointer(),size[0],size[1]);

  mag = cuda2ndDerivativePos(m_CudaConf->GetGridDim(),m_CudaConf->GetBlockDim(),input->GetDevicePointer(),deriv,size[0],size[1]);

  // Set NMS pointer on the output image
  output->GetPixelContainer()->SetDevicePointer(deriv, size[0]*size[1], true);
  output1->GetPixelContainer()->SetDevicePointer(mag, size[0]*size[1], true);

}

template <class TInputImage, class TOutputImage>
void 
CudaCannyEdgeDetectionImageFilter<TInputImage,TOutputImage>
::CudaHysteresisThresholding()
{

  // Get the image ponter
  typename OutputImageType::Pointer mag    = m_UpdateBuffer1;
  typename OutputImageType::Pointer deriv  = m_CudaZeroCrossingFilter->GetOutput();
  typename OutputImageType::Pointer output = this->GetOutput();
  typename TOutputImage::PixelType * edges;

  // Get image size
  typename OutputImageType::SizeType size = output->GetLargestPossibleRegion().GetSize();

  // Call CudaHysteresis. Defined on CudaCannyEdgeDetection.cu
  // This Multiplyes the Zero crossings of the Second derivative with the
  // magnitude gradients of the image.
  edges = cudaHysteresis(m_CudaConf->GetGridDim(),m_CudaConf->GetBlockDim(),deriv->GetDevicePointer(), mag->GetDevicePointer(), size[0], size[1], this->GetLowerThreshold(), this->GetUpperThreshold());

  // Set Canny output
  output->GetPixelContainer()->SetDevicePointer(edges, size[0]*size[1], true);
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
  m_UpdateBuffer1->CopyInformation( input );
  m_UpdateBuffer1->SetRequestedRegion(input->GetRequestedRegion());
  m_UpdateBuffer1->SetBufferedRegion(input->GetBufferedRegion());
  m_UpdateBuffer1->Allocate();  
  output->SetBufferedRegion(output->GetRequestedRegion());
  
  m_CudaConf->SetBlockDim(THREADS_PER_BLOCK,1,1);
  m_CudaConf->SetGridDim((this->GetInput()->GetPixelContainer()->Size()+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,1,1);

  // 1.Apply the Gaussian Filter to the input image.-------
  m_CudaGaussianFilter->SetInput(input);
  
  unsigned int timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  /// Start timer

  m_CudaGaussianFilter->Update();

  cutStopTimer( timer );  /// Stop timer
  printf("Gaussian time = %f ms\n",cutGetTimerValue( timer ));

  // 2. Calculate 2nd order directional derivative.-------
  // Calculate the 2nd order directional derivative of the smoothed image.
  // The output of this filter will be used to store the directional
  // derivative.
  timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  ///< Start timer

  this->Cuda2ndDerivative();

  cutStopTimer( timer );  ///< Stop timer
  printf("2nd Derivative time = %f ms\n",cutGetTimerValue( timer ));

  // 3. NonMaximumSupression.-------
  // Calculate the zero crossings of the 2nd directional derivative and write 
  // the result to output buffer. 
  m_CudaZeroCrossingFilter->SetInput(this->GetOutput());

  timer = 0;
  cutCreateTimer( &timer );
  cutStartTimer( timer );  ///< Start timer

  m_CudaZeroCrossingFilter->Update();

  cutStopTimer( timer );  ///< Stop timer
  printf("Maximum Supression time = %f ms\n",cutGetTimerValue( timer ));

  // 4. Hysteresis Thresholding.-------
  // Get all the edges corresponding to zerocrossings by multiplying the 2nd
  // directional derivative and the magnitude gradient on preparation.
  // Then do the double thresholding upon the edge responses.
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
