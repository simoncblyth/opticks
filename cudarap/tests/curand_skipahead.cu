// nvcc curand_skipahead.cu -std=c++11 -ccbin=/usr/bin/clang -o /tmp/curand_skipahead && /tmp/curand_skipahead 
/**
curand_skipahead.cu

* https://docs.nvidia.com/cuda/curand/device-api-overview.html

**/
#include <string>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <curand_kernel.h>

#include "NP.hh"


__global__ void init_rng(int threads_per_launch, int thread_offset, curandState* rng_states, unsigned long long seed, unsigned long long offset)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;
    curand_init(seed, id + thread_offset , offset, &rng_states[id]);  
}

__global__ void skip_rng(int threads_per_launch, int thread_offset, curandState* rng_states, unsigned long long skip)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;
    skipahead( skip, &rng_states[id]) ;
}

__global__ void generate_rng(int threads_per_launch, int thread_offset, curandState* rng_states, unsigned long long skip, float* d_arr, unsigned nj, unsigned j0, unsigned j1 )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;

    if(skip > 0)
    {
        skipahead( skip, &rng_states[id]) ;
    }

    for(unsigned j=j0;j<j1;j++) 
    {
        float u = curand_uniform(&rng_states[id]); 
        d_arr[id*nj+j] = u ;  
        //printf("%f ", u); 
    }
    //printf("\n");
}


int main(int argc, char** argv)
{
    int mode = argc > 1 ? std::atoi(argv[1]) : 1 ; 
    printf("mode %d \n", mode ); 

    unsigned ni = 100 ; 
    unsigned nj = 16 ;  

    NP a("<f4", ni, nj ); 

    CUdeviceptr d_arr ;
    cudaMalloc((void**)&d_arr, a.arr_bytes() ); 
    cudaMemset((void*)d_arr, 0, a.arr_bytes() ); 

    unsigned long long seed = 0ull ; 
    unsigned long long offset = 0ull ; 

    size_t rng_bytes = ni*sizeof(curandState) ;
    CUdeviceptr d_rng_states ;
    cudaMalloc((void**)&d_rng_states, rng_bytes ); 
    cudaMemset((void*)d_rng_states, 0, rng_bytes ); 

    int blocks_per_launch = 1 ; 
    int threads_per_launch = ni ; 
    int threads_per_block = threads_per_launch ; 
    int thread_offset = 0 ;  // used when need multiple launches to cover the work items 

    curandState* launch_states = (curandState*)d_rng_states ; 

    init_rng<<<blocks_per_launch,threads_per_block>>>(threads_per_launch, thread_offset, launch_states, seed, offset );
   
    if( mode == 1 )
    { 
        unsigned long long skip = 0ull ; 
        generate_rng<<<blocks_per_launch,threads_per_block>>>(threads_per_launch, thread_offset, launch_states, skip, (float*)d_arr, nj, 0, nj ) ;     
    }
    else if(mode == 2)
    {
        //
        //   [0   1   2   3]   4   5   6    7   8   9    a    b   [c   d   e   f]  
        //                     --------------------------------

        generate_rng<<<blocks_per_launch,threads_per_block>>>(threads_per_launch, thread_offset, launch_states,   0, (float*)d_arr, nj,  0,   4 ) ;     
        generate_rng<<<blocks_per_launch,threads_per_block>>>(threads_per_launch, thread_offset, launch_states,   8, (float*)d_arr, nj,  12, 16 ) ;     
    }

    cudaMemcpy( static_cast<void*>( a.bytes() ), (void*)d_arr, a.arr_bytes(), cudaMemcpyDeviceToHost ); 

    std::stringstream ss ;   
    ss << "/tmp/curand_skipahead_" << mode << ".npy" ;
    std::string s = ss.str(); 
    const char* path = s.c_str() ; 
    std::cout << "saving " << path << std::endl ; 
    a.save(path); 

    cudaDeviceSynchronize(); 

    NPU::check(path); 
 
    return 0 ; 
}




