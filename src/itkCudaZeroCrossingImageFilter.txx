/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkCudaZeroCrossingImageFilter.txx,v $
  Language:  C++
  Date:      $Date: 2010-11-04 17:31:03 $
  Version:   $Revision: 0.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCudaZeroCrossingImageFilter_txx
#define __itkCudaZeroCrossingImageFilter_txx

#include "itkCudaZeroCrossingImageFilter.h"

#define THREADS_PER_BLOCK 256 

namespace itk
{

template< class TInputImage, class TOutputImage >
void
CudaZeroCrossingImageFilter< TInputImage, TOutputImage >
::GenerateData()
{
  // Set input, output and temporary pointers
  typename TInputImage::ConstPointer input  = this->GetInput();
  typename TOutputImage::Pointer output  = this->GetOutput();
  typename TOutputImage::PixelType * tmp;
  
  m_CudaConf->SetBlockDim(THREADS_PER_BLOCK,1,1);
  m_CudaConf->SetGridDim((this->GetInput()->GetPixelContainer()->Size()+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,1,1);

  // Allocate output image object
  output->SetBufferedRegion(output->GetRequestedRegion());

  // Get image size
  typename OutputImageType::SizeType size = output->GetLargestPossibleRegion().GetSize();

  // Call cudaSobel. Defined on CudaSobelEdgeDetection.cu
  tmp = cudaZeroCrossing(m_CudaConf->GetGridDim(),m_CudaConf->GetBlockDim(),input->GetDevicePointer(), size[0], size[1]);

  // Set image pointer to the output image
  output->GetPixelContainer()->SetDevicePointer(tmp, size[0]*size[1], true);

}

template< class TInputImage, class TOutputImage >
void
CudaZeroCrossingImageFilter< TInputImage, TOutputImage >
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "ForegroundValue: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_ForegroundValue)
     << std::endl;
  os << indent << "BackgroundValue: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_BackgroundValue)
     << std::endl;
}

}//end of itk namespace

#endif
