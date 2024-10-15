// ~/o/sysrap/tests/curand_uniform_test.sh


#include <cstdlib>
#include <array>
#include "NP.hh"
#include "scuda.h"

#include "curand_kernel.h"

/**
_test_curand_uniform
-----------------------

**/

__global__ void _test_curand_uniform(float* ff, int ni, int nj)
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long seed = 0ull ; 
    unsigned long long subsequence = ix ;    // follow approach of ~/o/qudarap/QCurandState.cu 
    unsigned long long offset = 0ull ; 

    curandStateXORWOW rng ; 
    curand_init( seed, subsequence, offset, &rng ); 

    for(int j=0 ; j < nj ; j++) ff[ix*nj+j] = curand_uniform(&rng) ; 
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

void test_curand_uniform()
{
    int ni = 1000 ; 
    int nj = 16 ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    ConfigureLaunch(numBlocks, threadsPerBlock, ni ); 

    printf("//test_curand_uniform \n"); 
    NP* h = NP::Make<float>( ni, nj ) ; 
    int arr_bytes = h->arr_bytes() ;
    float* hh = h->values<float>(); 

    float* dd = nullptr ; 
    cudaMalloc(reinterpret_cast<void**>( &dd ), arr_bytes );     

    _test_curand_uniform<<<numBlocks,threadsPerBlock>>>(dd, ni, nj );  

    cudaMemcpy( hh, dd, arr_bytes, cudaMemcpyDeviceToHost ) ; 
    cudaDeviceSynchronize();

    h->save("$FOLD/curand_uniform_test.npy"); 
}
int main()
{
    test_curand_uniform();
    return 0 ; 
}


