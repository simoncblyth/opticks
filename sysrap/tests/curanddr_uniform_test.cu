/**

~/o/sysrap/tests/curanddr_uniform_test.sh

**/



#include <cstdlib>
#include <array>
#include "NP.hh"
#include "scuda.h"

#include "curand-done-right/curanddr.hxx"



__global__ void _test_curanddr_uniform(float* ff, int ni, int nj)
{
    uint ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint nk = nj/4 ;  
    for(uint k=0 ; k < nk ; k++) 
    {
        float* ffk = ff + 4*(ix*nk + k) ;  
        curanddr::uniforms_into_buffer<4>( ffk, uint4{k,0,ix,0}, 0 ); 
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

void test_curanddr_uniform()
{
    int ni = 1000 ; 
    int nj = 16 ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    ConfigureLaunch(numBlocks, threadsPerBlock, ni ); 

    printf("//test_curanddr_uniform   \n" ); 
    NP* h = NP::Make<float>( ni, nj ) ; 
    int arr_bytes = h->arr_bytes() ;
    float* hh = h->values<float>(); 

    float* dd = nullptr ; 
    cudaMalloc(reinterpret_cast<void**>( &dd ), arr_bytes );     

    _test_curanddr_uniform<<<numBlocks,threadsPerBlock>>>(dd, ni, nj );  

    cudaMemcpy( hh, dd, arr_bytes, cudaMemcpyDeviceToHost ) ; 
    cudaDeviceSynchronize();

    h->save("$FOLD/curanddr_uniform_test.npy"); 
}
int main()
{
    test_curanddr_uniform();
    return 0 ; 
}


