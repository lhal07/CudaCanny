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
#include "itkGaussianOperator.h"

#define THREADS_PER_BLOCK 256 

namespace itk
{

template< class TInputImage, class TOutputImage >
void
CudaDiscreteGaussianImageFilter<TInputImage, TOutputImage>
::GenerateData()
{

  typename TInputImage::ConstPointer input = this->GetInput();

  typename TOutputImage::Pointer output = this->GetOutput();
//  typename TOutputImage::PixelType * mask;

  // Determine the dimensionality to filter
  unsigned int filterDimensionality = m_FilterDimensionality;
  if (filterDimensionality > ImageDimension)
    {
    filterDimensionality = ImageDimension;
    }
  if (filterDimensionality == 0)
    {
    // no smoothing, copy input to output
    ImageRegionConstIterator<InputImageType> inIt(
      input,
      this->GetOutput()->GetRequestedRegion() );
    ImageRegionIterator<OutputImageType> outIt(
      output,
      this->GetOutput()->GetRequestedRegion() );

    while (!inIt.IsAtEnd())
      {
      outIt.Set( static_cast<OutputPixelType>(inIt.Get()) );
      ++inIt;
      ++outIt;
      }
    return;
    }

   // Create a series of operators
  typedef GaussianOperator<OutputPixelType, ImageDimension> OperatorType;
  std::vector<OperatorType> oper;
  oper.resize(filterDimensionality);

  // Set up the operators
  unsigned int i;
  for (i = 0; i < filterDimensionality; ++i){
    // we reverse the direction to minimize computation while, because
    // the largest dimension will be split slice wise for streaming 
    unsigned int reverse_i = filterDimensionality - i - 1;

    // Set up the operator for this dimension
    oper[reverse_i].SetDirection(i);
    if (m_UseImageSpacing == true)
      {
      if (input->GetSpacing()[i] == 0.0)
        {
        itkExceptionMacro(<< "Pixel spacing cannot be zero");
        }
      else
        {
        // convert the variance from physical units to pixels
        double s = input->GetSpacing()[i];
        s = s*s;
        oper[reverse_i].SetVariance(m_Variance[i] / s);
        }
      }
    else
      {
      oper[reverse_i].SetVariance(m_Variance[i]);
      }

    oper[reverse_i].SetMaximumKernelWidth(m_MaximumKernelWidth);
    oper[reverse_i].SetMaximumError(m_MaximumError[i]);
    oper[reverse_i].CreateDirectional();
  }

  m_CudaConf->SetBlockDim(oper[0].Size(),1,1);
  m_CudaConf->SetGridDim(1,1,1);

//  mask = cuda1DGaussianOperator(m_CudaConf->GetGridDim(),m_CudaConf->GetBlockDim(),this->GetMaximumKernelWidth(),(float) this->GetVariance());

  m_CudaConvolutionFilter->SetInput(input);
  m_CudaConvolutionFilter->SetInputMaskHorizontal(oper[0].GetBufferReference().begin(), oper[0].Size());
  m_CudaConvolutionFilter->SetInputMaskVertical(oper[1].GetBufferReference().begin(), oper[1].Size());
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
