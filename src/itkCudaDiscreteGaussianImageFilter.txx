/*=========================================================================

Program:   Insight Segmentation & Registration Toolkit
Module:    $RCSfile: itkCudaDiscreteGaussianImageFilter.txx,v $
Language:  C++
Date:      $Date: 2010-11-10 12:44:26 $
Version:   $Revision: 1.0 $

Copyright (c) Insight Software Consortium. All rights reserved.
See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCudaDiscreteGaussianImageFilter_txx
#define __itkCudaDiscreteGaussianImageFilter_txx

#include "itkCudaDiscreteGaussianImageFilter.h"

namespace itk
{

template< class TInputImage, class TOutputImage >
void
CudaDiscreteGaussianImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  typename TInputImage::ConstPointer input = this->GetInput();
  typename TOutputImage::PixelType * mask;
  
  mask = cuda1DGaussianOperator(this->GetMaximumKernelWidth(), (float) this->GetVariance());

  m_CudaConvolutionFilter->SetInput(input);
  m_CudaConvolutionFilter->SetInputMaskHorizontal(mask, this->GetMaximumKernelWidth());
  m_CudaConvolutionFilter->SetInputMaskVertical(mask, this->GetMaximumKernelWidth());
  m_CudaConvolutionFilter->Update();

}

template< class TInputImage, class TOutputImage >
TOutputImage *
CudaDiscreteGaussianImageFilter<TInputImage, TOutputImage>
::GetOutput()
{

  return(m_CudaConvolutionFilter->GetOutput());

}

template< class TInputImage, class TOutputImage >
void
CudaDiscreteGaussianImageFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Variance: " << m_Variance << std::endl;
  os << indent << "MaximumKernelWidth: " << m_MaximumKernelWidth << std::endl;
  os << indent << "FilterDimensionality: " << m_FilterDimensionality << std::endl;
}

} // end namespace itk

#endif
