/**
Robert Crovella example comparing use of cuComplex and thrust::complex

* https://stackoverflow.com/questions/9860711/cucomplex-h-and-exp

::

    epsilon:UseThrust blyth$ nvcc -o /tmp/UseThrustComplex UseThrustComplex.cu 
    epsilon:UseThrust blyth$ /tmp/UseThrustComplex
    thrust: 1.468694,2.287355, cuComplex: 1.468694,2.287355
    epsilon:UseThrust blyth$ cuda-memcheck /tmp/UseThrustComplex
    ========= CUDA-MEMCHECK
    thrust: 1.468694,2.287355, cuComplex: 1.468694,2.287355
    ========= ERROR SUMMARY: 0 errors

**/

#include <cuComplex.h>
#include <thrust/complex.h>
#include <stdio.h>

__host__ __device__
cuFloatComplex my_complex_exp (cuFloatComplex arg)
{
   cuFloatComplex res;
   float s, c;
   float e = expf(arg.x);
   sincosf(arg.y, &s, &c);
   res.x = c * e;
   res.y = s * e;
   return res;
}

__global__ void demo(){

  cuFloatComplex a = make_cuFloatComplex(1.0f, 1.0f);
  thrust::complex<float> b(1.0f, 1.0f);
  printf("thrust: %f,%f, cuComplex: %f,%f\n", exp(b).real(), exp(b).imag(), cuCrealf( my_complex_exp(a)), cuCimagf(my_complex_exp(a)));
}

int main()
{
  demo<<<1,1>>>();
  cudaDeviceSynchronize();
}

