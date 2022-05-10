/**
name=tcomplexTest ; nvcc $name.cu -I.. -I/usr/local/cuda -o /tmp/$name && /tmp/$name 

**/

#include <iostream>
#include <iomanip>

#include "scuda.h"
#include "tcomplex.h"


/*
 we test the follow calculation:

   a = 0 + 4i;
   sqrt of a = 2^0.5 + i* 2^0.5

*/

__global__ void test_complex()
{
     
   unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;

   cuFloatComplex a = make_cuFloatComplex(ix, iy);
   cuFloatComplex b = tcomplex::cuSqrtf(a); 

   float  a_rho = tcomplex::cuRhof(a);
   float  a_the = tcomplex::cuThetaf(a);
   
   float  b_rho = tcomplex::cuRhof(b);
   float  b_the = tcomplex::cuThetaf(b); 
   
   printf("//test_complex  (ix,iy) (%2i, %2i) (x,y) (%10.4f, %10.4f) a(rho,the)(%10.4f, %10.4f)  b(rho,the)(%10.4f, %10.4f ) \n", 
         ix, iy, a.x, a.y, a_rho, a_the, b_rho, b_the );
}


void test_device_complex_api()
{
    dim3 block(16,16); 
    dim3 grid(1,1);

    test_complex<<< grid , block >>>();  
    cudaDeviceSynchronize();
}

int main()
{
    test_device_complex_api();
    return 0 ; 
}


