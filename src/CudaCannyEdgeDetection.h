///canny.h
/**
 * \author Luis Lourenço (2010)
 * \version 2.2.1
 * \since 20/05/10
 */


extern "C"
float* gradientMaximumDetector(float *d_mag, float *d_dir, int width, int height);

extern "C"
void hysteresis(float *d_img, int width, int height, const unsigned int t1, const unsigned int t2);
