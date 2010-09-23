/*=========================================================================

Program:   Insight Segmentation & Registration Toolkit
Module:    $RCSfile: itkDiscreteGaussianImageFilter.txx,v $
Language:  C++
Date:      $Date: 2009-07-29 12:44:26 $
Version:   $Revision: 1.43 $

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
template <class TInputImage, class TOutputImage>
void 
CudaDiscreteGaussianImageFilter<TInputImage,TOutputImage>
::GenerateInputRequestedRegion() throw(InvalidRequestedRegionError)
{
  // call the superclass' implementation of this method. this should
  // copy the output requested region to the input requested region
  Superclass::GenerateInputRequestedRegion();
  
}


template< class TInputImage, class TOutputImage >
void
CudaDiscreteGaussianImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  typename TInputImage::ConstPointer input = this->GetInput();
  typename TOutputImage::Pointer output = this->GetOutput();
  typename TOutputImage::PixelType * ptr;
  
  // Allocate output image object
  output->SetBufferedRegion(output->GetRequestedRegion());
  output->Allocate();

  // Get image size
  typename OutputImageType::SizeType size;
  size = output->GetLargestPossibleRegion().GetSize();

  // Call cudaGaussian. Defined on CudaDiscreteGaussian.cu
  ptr = cudaDiscreteGaussian2D(input->GetDevicePointer(), size[0], size[1], (float) m_Variance, m_MaximumKernelWidth);

  // Set image pointer to the output image
  output->GetPixelContainer()->SetDevicePointer(ptr, size[0]*size[1], true);


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
