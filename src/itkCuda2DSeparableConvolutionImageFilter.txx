/*=========================================================================

Program:   Insight Segmentation & Registration Toolkit
Module:    $RCSfile: itkCuda2DSeparableConvolutionImageFilter.txx,v $
Language:  C++
Date:      $Date: 2010-11-11 15:03:32 $
Version:   $Revision: 0.1 $

Copyright (c) Insight Software Consortium. All rights reser
See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for detail.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCuda2DSeparableConvolutionImageFilter_txx
#define __itkCuda2DSeparableConvolutionImageFilter_txx

#include "itkCuda2DSeparableConvolutionImageFilter.h"

#include "itkImageBase.h"

#define THREADS_PER_BLOCK 256 

namespace itk {

template<class TInputImage, class TOutputImage>
Cuda2DSeparableConvolutionImageFilter<TInputImage, TOutputImage>
::Cuda2DSeparableConvolutionImageFilter()
{
  m_CudaConf = CudaInterfaceType::New();
}

template<class TInputImage, class TOutputImage>
Cuda2DSeparableConvolutionImageFilter<TInputImage, TOutputImage>
::~Cuda2DSeparableConvolutionImageFilter()
{
}

template<class TInputImage, class TOutputImage>
void
Cuda2DSeparableConvolutionImageFilter<TInputImage, TOutputImage>
::GenerateData() 
{

  typename TInputImage::ConstPointer input = this->GetInput();
  typename TOutputImage::Pointer output = this->GetOutput();
  typename TOutputImage::PixelType * ptr;
  
  m_CudaConf->SetBlockDim(THREADS_PER_BLOCK,1,1);
  m_CudaConf->SetGridDim((this->GetInput()->GetPixelContainer()->Size()+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,1,1);
  // Allocate output image object
  output->SetBufferedRegion(output->GetRequestedRegion());

  // Get image size
  typename OutputImageType::SizeType size;
  size = output->GetLargestPossibleRegion().GetSize();

  // Call cudaGaussian. Defined on CudaDiscreteGaussian.cu
  ptr = cuda2DSeparableConvolution(m_CudaConf->GetGridDim(),m_CudaConf->GetBlockDim(),input->GetDevicePointer(), size[0], size[1], m_Mask1, m_SizeMask1, m_Mask2, m_SizeMask2);

  // Set image pointer to the output image
  output->GetPixelContainer()->SetDevicePointer(ptr, size[0]*size[1], true);

}


/**
 * Set Input Mask Horizontal
 */
template <typename TInputImage, typename TOutputImage>
void
Cuda2DSeparableConvolutionImageFilter<TInputImage,TOutputImage>
::SetInputMaskHorizontal( OutputPixelType * input, unsigned int size )
{
  m_Mask1 = input;
  m_SizeMask1 = size;
}

/**
 * Get Input Mask Horizontal
 */
template <typename TInputImage, typename TOutputImage>
const typename TOutputImage::PixelType *
Cuda2DSeparableConvolutionImageFilter<TInputImage,TOutputImage>
::GetInputMaskHorizontal( void )
{
  return dynamic_cast<const OutputPixelType *>( m_Mask1 );
}


/**
 * Set Input Mask Vertical
 */
template <typename TInputImage, typename TOutputImage>
void
Cuda2DSeparableConvolutionImageFilter<TInputImage,TOutputImage>
::SetInputMaskVertical( OutputPixelType * input, unsigned int size )
{
  m_Mask2 = input;
  m_SizeMask2 = size;
}

/**
 * Get Input Mask Vertical
 */
template <typename TInputImage, typename TOutputImage>
const typename TOutputImage::PixelType *
Cuda2DSeparableConvolutionImageFilter<TInputImage,TOutputImage>
::GetInputMaskVertical( void )
{
  return dynamic_cast<const OutputPixelType *>( m_Mask2 );
}

}
#endif
