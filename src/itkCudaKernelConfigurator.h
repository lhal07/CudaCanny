/*=========================================================================

Program:   Insight Segmentation & Registration Toolkit
Module:    $RCSfile: itkCudaKernelConfigurator.h,v $
Language:  C++
Date:      $Date: 2010-11-19 12:24:09 $
Version:   $Revision: 0.1 $

Copyright (c) Insight Software Consortium. All rights reserved.
See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCudaKernelConfigurator_h
#define __itkCudaKernelConfigurator_h

#include <cuda.h>
#include <cutil_inline.h>
#include <utility>

namespace itk
{

   /** \class ImportImageContainer
    * Defines an itk::Image front-end to a standard C-array. This container
    * conforms to the ImageContainerInterface. This is a full-fleged Object,
    * so there is modification time, debug, and reference count information.
    */

    class ITK_EXPORT CudaKernelConfigurator :
      public Object
   {
      public:
         /** Standard class typedefs. */
         typedef CudaKernelConfigurator    Self;
         typedef Object                    Superclass;
         typedef SmartPointer<Self>        Pointer;
         typedef SmartPointer<const Self>  ConstPointer;

         /** Run-time type information (and related methods). */
         itkTypeMacro(CudaKernelConfigurator,Object);

         /** Method for creation through the object factory. */
         itkNewMacro(Self);

         /** Get the cuda block and grid dimensions */
         dim3 GetBlockDim() 
         { 
            return m_BlockDim;
         };

         dim3 GetGridDim()
         {
            return m_GridDim;
         };

         /** Set the cuda block and grid dimensions */
         void SetBlockDim(uint x, uint y, uint z)
         {
           m_BlockDim.x = x;
           m_BlockDim.y = y;
           m_BlockDim.z = z;
         };

         void SetGridDim(uint x, uint y, uint z)
         {
           m_GridDim.x = x;
           m_GridDim.y = y;
           m_GridDim.z = 1;
         };


      protected:
         CudaKernelConfigurator();
         virtual ~CudaKernelConfigurator(){};

         /** PrintSelf routine. Normally this is a protected internal method. It is
          * made public here so that Image can call this method.  Users should not
          * call this method but should call Print() instead. */
         void PrintSelf(std::ostream& os, Indent indent) const;

      private:
         CudaKernelConfigurator(const Self&); //purposely not implemented

         dim3 m_BlockDim;
         dim3 m_GridDim;
   };

} // end namespace itk

// Define instantiation macro for this template.
#define ITK_TEMPLATE_CudaKernelConfigurator(_, EXPORT, x, y) namespace itk { \
   _(2(class EXPORT CudaKernelConfigurator< ITK_TEMPLATE_2 x >)) \
   namespace Templates { typedef CudaKernelConfigurator<ITK_TEMPLATE_2 x > CudaKernelConfigurator##y; } \
}

#if ITK_TEMPLATE_EXPLICIT
# include "Templates/itkCudaKernelConfigurator+-.h"
#endif

#if ITK_TEMPLATE_TXX
# include "itkCudaKernelConfigurator.txx"
#endif

#endif
