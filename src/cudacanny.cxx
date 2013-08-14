///cudacanny.cxx
/**
 * \author Luis Lourenço (2010)
 * \version 2.0.0
 * \since 20/05/10
 */
/* =============================================================================
 *
 *       Filename:  cudacanny.cxx
 *
 *    Description:  
 *
 *        Version:  2.0
 *        Created:  04-12-2009 11:05:30
 *       Revision:  none
 *       Compiler:  cmake
 *
 *         Author:  Luis Lourenço (2010)
 *        Company:  
 *
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCudaCannyEdgeDetectionImageFilter.h"

typedef unsigned char                                          ucharPixelType;
typedef float                                                       PixelType;
typedef itk::Image<PixelType,2>                                     ImageType;
typedef itk::Image<ucharPixelType,2>                           ucharImageType;
typedef itk::ImageFileReader< ImageType >                          ReaderType;
typedef itk::ImageFileWriter< ucharImageType >                     WriterType;
typedef itk::CudaCannyEdgeDetectionImageFilter< ImageType, ImageType > CannyFilter;
typedef itk::RescaleIntensityImageFilter< ImageType, ucharImageType > RescaleFilter;


int main (int argc, char** argv){

  // Verify the number of parameters in the command line
  if( argc < 6 ){
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " <inputImg> <outputImg> <sigma> <TH high> <TH low>" << std::endl;
    return EXIT_FAILURE;
  }

  cudaSetDevice(1);
//  cudaThreadSetCacheConfig(cudaFuncCachePreferShared);

  ImageType::Pointer res = ImageType::New();

  // Read parameters
  const float gaussianVariance = atof( argv[3] );
  const float t2 = atoi( argv[4] );
  const float t1 = atoi( argv[5] );

  // Input image
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( argv[1] );
  reader->Update();

  ///Warmup
  unsigned int WarmupTimer = 0;
  cutCreateTimer( &WarmupTimer );
  cutStartTimer( WarmupTimer );
  int *rub;
  cudaMalloc( (void**)&rub, reader->GetOutput()->GetPixelContainer()->Size() * sizeof(int) );
  cudaFree( rub );
  CUT_CHECK_ERROR("Warmup failed");
  cudaThreadSynchronize();
  cutStopTimer( WarmupTimer );
  printf("Warmup time = %f ms\n",cutGetTimerValue( WarmupTimer ));

  unsigned int CannyTimer = 0;
  cutCreateTimer( &CannyTimer );

  // Apply canny operator
  CannyFilter::Pointer canny = CannyFilter::New();
  canny->SetInput(reader->GetOutput());
  canny->SetVariance(gaussianVariance);
  canny->SetUpperThreshold(t2);
  canny->SetLowerThreshold(t1);

  cutStartTimer( CannyTimer );

  canny->Update();

  cutStopTimer( CannyTimer );

  // Rescale image to uchar. PNG does only supports uchar ou ushort
  RescaleFilter::Pointer rescale = RescaleFilter::New();
  rescale->SetOutputMinimum(   0 );
  rescale->SetOutputMaximum( 255 );
  rescale->SetInput( canny->GetOutput() );

  // Write image to file
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( argv[2] );
  writer->SetInput( rescale->GetOutput() );
  writer->Update();

  printf("cudaCanny time: %f ms\n",cutGetTimerValue( CannyTimer ));
  
  return EXIT_SUCCESS;
}
