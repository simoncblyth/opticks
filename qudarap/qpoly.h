#pragma once
/**

Polymorphism on device would be handy for 
treating different genstep types with common 
references.

Apparently that is possible so long as the objects are created on device.

https://stackoverflow.com/questions/22988244/polymorphism-and-derived-classes-in-cuda-cuda-thrust/23476510#23476510
 
Or perhaps there is a workaround "CUDA polymorphism workaround"

https://forums.developer.nvidia.com/t/is-there-a-function-polymorphism-trick-in-device-code/22407

* uses one class, and branches on type   

**/

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QPOLY_METHOD __host__ __device__ 
#else
   #define QPOLY_METHOD 
#endif 


struct Poly
{
   float width ; 
   float height;

   QPOLY_METHOD void set_param(float width_, float height_ )
   { 
       width=width_; 
       height=height_; 
   }
   QPOLY_METHOD virtual float area(){ return 0.f; }
};


struct RectangleV1 : Poly
{
    QPOLY_METHOD float area(){ return width * height + 0.1f ; }
};

struct RectangleV2 : Poly
{
    QPOLY_METHOD float area(){ return width * height + 0.2f ;  }
};

struct TriangleV1 : Poly 
{
    QPOLY_METHOD float area(){ return width*height/2.f + 0.1f ; }
};

struct TriangleV2 : Poly 
{
    QPOLY_METHOD float area(){ return width*height/2.f + 0.2f ; }
};








