// ./erfcinvf_Test.sh

#include <cstdlib>
#include <array>
#include "NP.hh"
#include "scuda.h"

const char* FOLD = getenv("FOLD") ? getenv("FOLD") : "/tmp" ; 

/**
_test_erfcinvf
-----------------

erfcinvf 
   inverse complementary error function, 0: +inf, 2: -inf outside 0,2:nan 

   * https://mathworld.wolfram.com/InverseErfc.html
 
mapping erfcinvf argument into 0->2 and its result by -M_SQRT2f 
gives match with S4MTRandGaussQ::transformQuick

**/

__global__ void _test_erfcinvf(float* ff, int ni, int nj)
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;

    float u = float(ix)/float(ni-1) ;  // standin for uniform rand between 0 and 1 
    float u2 = 2.f*u  ; 
    float v = -M_SQRT2f*erfcinvf(u2) ;   

    ff[ix*nj+0] = u ; 
    ff[ix*nj+1] = u2 ; 
    ff[ix*nj+2] = v ; 
}

void ConfigureLaunch(dim3& numBlocks, dim3& threadsPerBlock, unsigned width )
{ 
    threadsPerBlock.x = 512 ; 
    threadsPerBlock.y = 1 ; 
    threadsPerBlock.z = 1 ; 

    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = 1 ; 
    numBlocks.z = 1 ; 
}

void test_erfcinvf()
{
    int ni = 1000 ; 
    int nj = 3 ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    ConfigureLaunch(numBlocks, threadsPerBlock, ni ); 

    printf("//test_erfcinvf \n"); 
    NP* h = NP::Make<float>( ni, nj ) ; 
    int arr_bytes = h->arr_bytes() ;
    float* hh = h->values<float>(); 

    float* dd = nullptr ; 
    cudaMalloc(reinterpret_cast<void**>( &dd ), arr_bytes );     

    _test_erfcinvf<<<numBlocks,threadsPerBlock>>>(dd, ni, nj );  

    cudaMemcpy( hh, dd, arr_bytes, cudaMemcpyDeviceToHost ) ; 
    cudaDeviceSynchronize();

    h->save(FOLD,"test_erfcinvf.npy"); 
}
int main()
{
    test_erfcinvf();
    return 0 ; 
}


