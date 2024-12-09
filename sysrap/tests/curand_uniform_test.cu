// ~/o/sysrap/tests/curand_uniform_test.sh

#include <cstdlib>
#include <array>
#include "NP.hh"
#include "scuda.h"



#include "curand_kernel.h"
#include "curandlite/curandStatePhilox4_32_10_OpticksLite.h"

using opticks_curandState_t = curandStatePhilox4_32_10_OpticksLite ; 


/**
_test_curand_uniform
-----------------------

**/

template<typename T>
__global__ void _test_curand_uniform(float* ff, int ni, int nj)
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long seed = 0ull ; 
    unsigned long long subsequence = ix ;    // follow approach of ~/o/qudarap/QCurandState.cu 
    unsigned long long offset = 0ull ; 

    T rng ; 

    curand_init( seed, subsequence, offset, &rng ); 

    if(ix == 0) printf("//_test_curand_uniform sizeof(T) %lu \n", sizeof(T)); 

    int nk = nj/4 ;  

    for(int k=0 ; k < nk ; k++) 
    {
        float4 ans = curand_uniform4(&rng); 
        ff[4*(ix*nk+k)+0] = ans.x ;  
        ff[4*(ix*nk+k)+1] = ans.y ; 
        ff[4*(ix*nk+k)+2] = ans.z ; 
        ff[4*(ix*nk+k)+3] = ans.w ; 
    }
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



template<typename T>
void test_curand_uniform()
{
    int ni = 1000 ; 
    int nj = 16 ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    ConfigureLaunch(numBlocks, threadsPerBlock, ni ); 

    printf("//test_curand_uniform  sizeof(T) %d \n", sizeof(T) ); 
    NP* h = NP::Make<float>( ni, nj ) ; 
    int arr_bytes = h->arr_bytes() ;
    float* hh = h->values<float>(); 

    float* dd = nullptr ; 
    cudaMalloc(reinterpret_cast<void**>( &dd ), arr_bytes );     

    _test_curand_uniform<T><<<numBlocks,threadsPerBlock>>>(dd, ni, nj );  

    cudaMemcpy( hh, dd, arr_bytes, cudaMemcpyDeviceToHost ) ; 
    cudaDeviceSynchronize();

    h->save("$FOLD/curand_uniform_test.npy"); 
}
int main()
{
    int MODE = U::GetEnvInt("MODE", 0); 

    

    if(MODE == 0)
    {
        printf("test_curand_uniform<curandStateXORWOW>()"); 
        test_curand_uniform<curandStateXORWOW>();
    }
    else if(MODE == 1)
    {
        printf("test_curand_uniform<curandStatePhilox4_32_10>()"); 
        test_curand_uniform<curandStatePhilox4_32_10>();
    }
    else if(MODE == 2)
    {
        printf("test_curand_uniform<curandStatePhilox4_32_10_OpticksLite>()"); 
        test_curand_uniform<curandStatePhilox4_32_10_OpticksLite>();
    }
    else if(MODE == 3)
    {
        printf("test_curand_uniform<opticks_curandState_t>()"); 
        test_curand_uniform<opticks_curandState_t>();
    }




    return 0 ; 
}


