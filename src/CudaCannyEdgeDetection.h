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
float* gradientMaximumDetector(float *d_mag, float *d_dir, int width, int height);

extern "C"
void hysteresis(float *d_img, int width, int height, const unsigned int t1, const unsigned int t2);
