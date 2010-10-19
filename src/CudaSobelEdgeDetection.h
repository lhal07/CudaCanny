///CannySobelEdgeDetection.h
/**
 * \author luis lourenço (2010)
 * \version 3.0.0
 * \since 22/09/10
 */

template <class TMagnitude, class TDirection>
class Gradient
{
public:
  TMagnitude *Magnitude;
  TDirection *Direction;
};

typedef Gradient<float,float>  Tgrad;

extern "C"
Tgrad* cudaSobel(Tgrad* d_gradient, const float *d_img, int width, int height);

