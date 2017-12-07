
// http://docs.nvidia.com/cuda/curand/device-api-overview.html#thrust-and-curand-example
// http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MTGP/mtgp3.pdf

#include <cassert> 
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h> 
#include <curand_kernel.h> 
#include <iostream> 
#include <iomanip> 


template <typename T>
struct curand_printf 
{ 
    T _seed ;
    T _offset ;
    T _seq0 ; 
    T _seq1 ; 
    T _zero ; 
 
    curand_printf( T seed , T offset, T seq0, T seq1 )
       :
       _seed(seed),
       _offset(offset),
       _seq0(seq0),
       _seq1(seq1),
       _zero(0)
    {
    }
    
    __device__ 
    void operator()(unsigned id) 
    { 
        unsigned thread_offset = 0 ;  
        curandState s; 
        curand_init(_seed, id + thread_offset, _offset, &s); 
        printf(" id:%4u thread_offset:%u seq0:%llu seq1:%llu \n", id, thread_offset, _seq0, _seq1 );  
 
        for(T i = _zero ; i < _seq1 ; ++i) 
        { 
            float x = curand_uniform(&s); 
            if( i < _seq0 ) continue ; 

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
     int q0 = argc > 3 ? atoi(argv[3]) : 0 ; 
     int q1 = argc > 4 ? atoi(argv[4]) : 16 ; 

     std::cout 
         << argv[0]
         << std::endl  
         << " i0 " << i0  
         << " i1 " << i1
         << " q0 " << q0  
         << " q1 " << q1
         << std::endl 
         ; 

     assert( i0 >= 0 && i1 >= 0 );
     assert( q0 >= 0 && q1 >= 0 );
     assert( i0 < i1 );
     assert( q0 < q1 );

     thrust::for_each( 
                thrust::counting_iterator<int>(i0), 
                thrust::counting_iterator<int>(i1), 
                curand_printf<unsigned long long>(0,0,q0,q1));

    cudaDeviceSynchronize();  
    return 0; 
} 

