/**
curand_uniform_test.cc
=======================

::

    ~/o/sysrap/tests/curand_uniform_test.sh

    // https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/

**/


#include <cstdlib>
#include <array>
#include <chrono>

#include "NP.hh"
#include "SCurandState.h"
// HUH: nvcc ignoring "pragma once" from NP.hh NPU.hh with this combo, but macro guards OK

#include "scuda.h"
#include "sstr.h"
#include "srng.h"
#include "sstamp.h"

#include "curand_kernel.h"
#include "curandlite/curandStatePhilox4_32_10_OpticksLite.h"


using RNG0 = curandStateXORWOW ; 
using RNG1 = curandStateXORWOW ; 
using RNG2 = curandStatePhilox4_32_10 ; 
using RNG3 = curandStatePhilox4_32_10_OpticksLite ; 


/**
_test_curand_uniform
-----------------------

**/



struct KernelInfo
{
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 

    int ni ; 
    int nj ; 
    float ms ;   // milliseconds (1e-3 s)

    int64_t dt0 ;  // us from start of process 
    int64_t t0 ; 
    int64_t t1 ; 
    int64_t et ; 

    double dt ;  // microseconds (1e-6 s)
    const char* name ; 
    void*    states ; 
    float*   dd ; 

    bool four_by_four ; 
    bool download ; 

    std::string desc() const ; 
};

std::string KernelInfo::desc() const
{
    std::stringstream ss ; 
    ss 
        << " dt0 " << dt0 
        << " ms " << std::fixed << std::setw(10) << std::setprecision(6) << ms 
       // << " t0 " << sstamp::Format(t0) 
       // << " t1 " << sstamp::Format(t1) 
        << " [t1-t0;us] " << std::setw(8) << ( t1 - t0 )
        << " states " << ( states ? "YES" : "NO " ) 
        << " download " << ( download ? "YES" : "NO " ) 
        << " four_by_four " << ( four_by_four ? "YES" : "NO " ) 
        << " name " << ( name ? name : "-" ) 
        ; 

    std::string str = ss.str() ;
    return str ;  
}




template<typename T>
__global__ void _test_curand_uniform(float* ff, int ni, int nj, T* states, bool four_by_four)
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long subsequence = ix ;    // follow approach of ~/o/qudarap/QCurandState.cu 
    unsigned long long seed = 0ull ; 
    unsigned long long offset = 0ull ; 

    T rng ; 

    if( states == nullptr )
    {
        curand_init( seed, subsequence, offset, &rng ); 
    }
    else
    {
        rng = states[subsequence] ;  
    }

    if(four_by_four)
    {
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
    else
    {
        for(int j=0 ; j < nj ; j++) 
        {
            ff[ix*nj+j] = curand_uniform(&rng);  
        }

    }

}

void ConfigureLaunch(dim3& numBlocks, dim3& threadsPerBlock, unsigned width )
{ 
    threadsPerBlock.x = 1024 ; 
    threadsPerBlock.y = 1 ; 
    threadsPerBlock.z = 1 ; 

    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = 1 ; 
    numBlocks.z = 1 ; 
}


template<typename T>
void test_curand_uniform(KernelInfo& ki )
{
    ki.t0 = sstamp::Now(); 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    _test_curand_uniform<T><<<ki.numBlocks,ki.threadsPerBlock>>>(ki.dd, ki.ni, ki.nj, (T*)ki.states, ki.four_by_four );  
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);  // blocks CPU execution until the specified event is recorded

    ki.t1 = sstamp::Now(); 

    cudaEventElapsedTime(&ki.ms, start, stop);
}


int main()
{
    int NI = U::GetEnvInt("NI", 1000); 
    int NJ = U::GetEnvInt("NJ", 16 ); 

    int64_t t0 = sstamp::Now(); 

    NP* h = NP::Make<float>( NI, NJ ) ; 
    float* hh = h->values<float>(); 


    NP::INT nv = h->num_values(); 
    float* dd = SCU_::device_alloc<float>( nv, "randoms") ; 

    int64_t t1 = sstamp::Now(); 
    std::cout << " t1 - t0 : output allocations [us] " << ( t1 - t0 ) << "\n" ; 
    
    SCurandState cs ; 
    //std::cout << cs.desc() << "\n" ; 
    RNG0* d0 = cs.loadAndUpload<RNG0>(NI) ; 

    int64_t t2 = sstamp::Now(); 
    std::cout << " t2 - t1 : loadAndUpload [us] " << ( t2 - t1 ) << "\n" ; 
  
    std::array<KernelInfo,16> kis; 


    for(int m=0 ; m < kis.size() ; m++ )
    {
        int m4 = m % 4 ; 
        int g4 = m / 4 ; 

        KernelInfo& ki = kis[m] ; 
        ConfigureLaunch(ki.numBlocks, ki.threadsPerBlock, NI ); 
        int64_t t2 = sstamp::Now();  


        ki.dt0 = t2 - t1 ; 
        ki.ni = NI ; 
        ki.nj = NJ ; 
        ki.states = m4 == 1 ? d0 : nullptr ; 
        //ki.download = g4 == 1 ? true : false ;  
        ki.download = false ;  
        ki.dd = dd ; 
        ki.four_by_four = g4 % 2 == 1 ;  
     
       
        switch(m4)
        {
           case 0: test_curand_uniform<RNG0>(ki); ki.name = srng<RNG0>::NAME ; break ; 
           case 1: test_curand_uniform<RNG1>(ki); ki.name = srng<RNG1>::NAME ; break ; 
           case 2: test_curand_uniform<RNG2>(ki); ki.name = srng<RNG2>::NAME ; break ; 
           case 3: test_curand_uniform<RNG2>(ki); ki.name = srng<RNG3>::NAME ; break ; 
        }

        if(m4 == 0 ) std::cout << "\n" ;  
        std::cout << ki.desc() << "\n" ;

        if(ki.download)
        {
            cudaMemcpy( hh, dd, h->arr_bytes(), cudaMemcpyDeviceToHost ) ; 
            cudaDeviceSynchronize();

            std::string path = sstr::Format_("$FOLD/RNG%d.npy", m ); 
            h->save(path.c_str()); 
        }
    }

    return 0 ; 
}


