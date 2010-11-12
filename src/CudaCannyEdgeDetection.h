///canny.h
/**
 * \author Luis Lourenço (2010)
 * \version 2.2.1
 * \since 20/05/10
 */

// pixel types used on hysteresis
#define DEFINITIVE_EDGE 255
#define POSSIBLE_EDGE 128
#define NON_EDGE 0

// sizes for hysteresis' slice width
#define SLICE_WIDTH 18
#define SLICE_BLOCK_WIDTH 16

// there's a pixel modified or not on the slice
#define MODIFIED 1
#define NOT_MODIFIED 0

extern "C"
float* cuda2ndDerivativePos(const float *d_input, const float *d_Lvv, int width, int height);

extern "C"
float* cuda2ndDerivative(const float *d_input, int width, int height);

extern "C"
float* cudaHysteresis(float *d_img, float *d_gauss, int width, int height, float t1, float t2);
