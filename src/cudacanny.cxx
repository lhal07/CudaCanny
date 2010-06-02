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

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCudaCannyEdgeDetectionImageFilter.h"

//para teste
#include <time.h>
#include <sys/time.h>

typedef unsigned char                                          ucharPixelType;
typedef float                                                       PixelType;
typedef itk::Image<PixelType,2>                                     ImageType;
typedef itk::Image<ucharPixelType,2>                           ucharImageType;
typedef itk::ImageFileReader< ImageType >                          ReaderType;
typedef itk::ImageFileWriter< ucharImageType >                     WriterType;
//typedef itk::CannyEdgeDetectionImageFilter< FloatImageType, FloatImageType > CannyFilter;
typedef itk::CannyEdgeDetectionImageFilter< ImageType, ImageType > CannyFilter;
typedef itk::RescaleIntensityImageFilter< ImageType, ucharImageType > RescaleFilter;


int main (int argc, char** argv){

  // Verify the number of parameters in the command line
  if( argc < 7 ){
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " <inputImg> <outputImg> <sigma> <maxKernelWidth> <TH high> <TH low>" << std::endl;
    return EXIT_FAILURE;
  }

  struct timeval tv1,tv2;
  unsigned int time = 0;

  //read parameters
  const float gaussianVariance = atof( argv[3] );
  const unsigned int maxKernelWidth = atoi( argv[4] );
  const unsigned int t2 = atoi( argv[5] );
  const unsigned int t1 = atoi( argv[6] );

  //input image
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( argv[1] );
  reader->Update();

  gettimeofday(&tv1,NULL);

  /* apply canny operator */

  CannyFilter::Pointer canny = CannyFilter::New();
  canny->SetInput(reader->GetOutput());
  canny->SetVariance(gaussianVariance);
  canny->SetUpperThreshold(t2);
  canny->SetLowerThreshold(t1);
//  canny->SetMaxKernelWidth(maxKernelWidth);
//  canny->UpdateInCUDA(image->GetBufferPointer(),maxKernelWidth);
//  canny->Update();

  gettimeofday(&tv2,NULL);
 
  RescaleFilter::Pointer rescale = RescaleFilter::New();
  rescale->SetOutputMinimum(   0 );
  rescale->SetOutputMaximum( 255 );
  rescale->SetInput( canny->GetOutput() );
//  rescale->SetInput( canny->GetInput() ); // is canny overwriting it's input?

  //write image to file
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( argv[2] );
  writer->SetInput( rescale->GetOutput() );
  writer->Update();

  if(tv1.tv_usec > tv2.tv_usec)
    time = 1000000;

  printf("cudaCanny Time: %f ms\n",((tv2.tv_sec-tv1.tv_sec)*1000+(float)(time+tv2.tv_usec-tv1.tv_usec)/1000));
  
  return EXIT_SUCCESS;
}
