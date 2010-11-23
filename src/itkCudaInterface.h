/*=========================================================================

Program:   Insight Segmentation & Registration Toolkit
Module:    $RCSfile: itkCudaInterface.h,v $
Language:  C++
Date:      $Date: 2010-11-19 12:24:09 $
Version:   $Revision: 0.1 $

Copyright (c) Insight Software Consortium. All rights reserved.
See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCudaInterface_h
#define __itkCudaInterface_h

#include <cuda.h>
#include <cutil_inline.h>
#include <utility>

namespace itk
{

    class ITK_EXPORT CudaInterface :
      public Object
   {
      public:
         /** Standard class typedefs. */
         typedef CudaInterface    Self;
         typedef Object                    Superclass;
         typedef SmartPointer<Self>        Pointer;
         typedef SmartPointer<const Self>  ConstPointer;

         /** Run-time type information (and related methods). */
         itkTypeMacro(CudaInterface,Object);

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
         CudaInterface();
         virtual ~CudaInterface(){};

         /** PrintSelf routine. Normally this is a protected internal method. It is
          * made public here so that Image can call this method.  Users should not
          * call this method but should call Print() instead. */
         void PrintSelf(std::ostream& os, Indent indent) const;

      private:
         CudaInterface(const Self&); //purposely not implemented

         dim3 m_BlockDim;
         dim3 m_GridDim;
   };

} // end namespace itk

// Define instantiation macro for this template.
#define ITK_TEMPLATE_CudaInterface(_, EXPORT, x, y) namespace itk { \
   _(2(class EXPORT CudaInterface< ITK_TEMPLATE_2 x >)) \
   namespace Templates { typedef CudaInterface<ITK_TEMPLATE_2 x > CudaInterface##y; } \
}

#if ITK_TEMPLATE_EXPLICIT
# include "Templates/itkCudaInterface+-.h"
#endif

#if ITK_TEMPLATE_TXX
# include "itkCudaInterface.txx"
#endif

#endif
