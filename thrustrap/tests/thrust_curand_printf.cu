
// http://docs.nvidia.com/cuda/curand/device-api-overview.html#thrust-and-curand-example
// http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MTGP/mtgp3.pdf

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h> 
#include <curand_kernel.h> 
#include <iostream> 
#include <iomanip> 

struct curand_printf 
{ 
    unsigned long long _seed ;
    unsigned long long _offset ;

    curand_printf( unsigned long long seed , unsigned long long offset )
       :
       _seed(seed),
       _offset(offset)
    {
    }
    
    __device__ 
    void operator()(unsigned id) 
    { 
        unsigned int N = 16; // samples per thread 
        unsigned thread_offset = 0 ;  
        curandState s; 
        curand_init(_seed, id + thread_offset, _offset, &s); 
        printf(" id:%4u thread_offset:%u \n", id, thread_offset );  
        for(unsigned i = 0; i < N; ++i) 
        { 
            float x = curand_uniform(&s); 
            printf(" %lf ", x );  
            if( i % 4 == 3 ) printf("\n") ; 
        } 
    } 
}; 

/*

__device__ void
curand_init (
    unsigned long long seed, 
    unsigned long long sequence,
    unsigned long long offset, 
    curandState_t *state)

The curand_init() function sets up an initial state allocated by the caller
using the given seed, sequence number, and offset within the sequence.
Different seeds are guaranteed to produce different starting states and
different sequences. The same seed always produces the same state and the same
sequence. The state set up will be the state after 2^67 sequence + offset calls
to curand() from the seed state.

*/


int main(int argc, char** argv) 
{ 
     int i0 = argc > 1 ? atoi(argv[1]) : 0 ; 
     int i1 = argc > 2 ? atoi(argv[2]) : i0+1 ; 
     std::cout 
         << argv[0]
         << std::endl 
         ;

     std::cout 
         << " i0 " << i0  
         << " i1 " << i1
         << std::endl 
         ; 
     thrust::for_each( 
                thrust::counting_iterator<int>(i0), 
                thrust::counting_iterator<int>(i1), 
                curand_printf(0,0));

    cudaDeviceSynchronize();  
    return 0; 
} 

