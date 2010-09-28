///CannySobelEdgeDetection.h
/**
 * \author luis lourenço (2010)
 * \version 3.0.0
 * \since 22/09/10
 */

template <class TStrenght, class TDirection>
class Gradient
{
public:
  TStrenght  *Strenght;
  TDirection *Direction;
};

typedef Gradient<float,short2>  Tgrad;

extern "C"
Tgrad* cudaSobel(Tgrad* d_gradient, const float *d_img, int width, int height);

