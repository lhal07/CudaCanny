/*=========================================================================

Program:   Insight Segmentation & Registration Toolkit
Module:    $RCSfile: itkCudaKernelConfigurator.txx,v $
Language:  C++
Date:      $Date: 2010-11-19 19:10:47 $
Version:   $Revision: 0.1 $

Copyright (c) Insight Software Consortium. All rights reserved.
See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

Portions of this code are covered under the VTK copyright.
See VTKCopyright.txt or http://www.kitware.com/VTKCopyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCudaKernelConfigurator_txx
#define __itkCudaKernelConfigurator_txx

#include "itkCudaKernelConfigurator.h"
#include <cstring>
#include <stdlib.h>
#include <string.h>

namespace itk
{

      CudaKernelConfigurator
      ::CudaKernelConfigurator()
      {
        m_BlockDim.x = 1;
        m_BlockDim.y = 1;
        m_BlockDim.z = 1;
        m_GridDim.x = 1;
        m_GridDim.y = 1;
        m_GridDim.z = 1;
      }



      void 
      CudaKernelConfigurator
      ::PrintSelf(std::ostream& os, Indent indent) const
      {
         os << indent << "BlockDim.x: " << m_BlockDim.x << std::endl;
         os << indent << "BlockDim.y: " << m_BlockDim.y << std::endl;
         os << indent << "BlockDim.z: " << m_BlockDim.z << std::endl;
         os << indent << "GridDim.x: " << m_GridDim.x << std::endl;
         os << indent << "GridDim.y: " << m_GridDim.y << std::endl;
         os << indent << "GridDim.z: " << m_GridDim.z << std::endl;
      }


} // end namespace itk

#endif
