#include <cstdio>
#include "curand_kernel.h"
#include "qcurandstate.h"
#include "QUDA_CHECK.h"
#include "SLaunchSequence.h"



__global__ void _QCurandState_curand_init(int threads_per_launch, int thread_offset, qcurandstate* cs, curandState* states_thread_offset )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;
    curand_init(cs->seed, id+thread_offset, cs->offset, states_thread_offset + id );  

    //if( id == 0 ) printf("// _QCurandState_curand_init thread_offset %d \n", thread_offset ); 
}


void before_kernel( cudaEvent_t& start, cudaEvent_t& stop )
{
    QUDA_CHECK( cudaEventCreate( &start ) );
    QUDA_CHECK( cudaEventCreate( &stop ) );
    QUDA_CHECK( cudaEventRecord( start,0 ) );
}
float after_kernel( cudaEvent_t& start, cudaEvent_t& stop )
{
    float kernel_time = 0.f ;

    QUDA_CHECK( cudaEventRecord( stop,0 ) );
    QUDA_CHECK( cudaEventSynchronize(stop) );

    QUDA_CHECK( cudaEventElapsedTime(&kernel_time, start, stop) );
    QUDA_CHECK( cudaEventDestroy( start ) );
    QUDA_CHECK( cudaEventDestroy( stop ) );

    QUDA_CHECK( cudaDeviceSynchronize() );

    return kernel_time ;
}

extern "C" void QCurandState_curand_init(SLaunchSequence* seq,  qcurandstate* cs, qcurandstate* d_cs) 
{
    // NB this is still on CPU, dereferencing d_cs here will BUS_ERROR 

    printf("//QCurandState_curand_init seq.items %d cs %p  d_cs %p cs.num %llu \n", seq->items, cs, d_cs, cs->num );  

    cudaEvent_t start, stop ;

    for(unsigned i=0 ; i < seq->launches.size() ; i++)
    {
        SLaunch& l = seq->launches[i] ; 
        printf("// l.sequence_index %d  l.blocks_per_launch %d l.threads_per_block %d  l.threads_per_launch %d l.thread_offset %d  \n", 
                   l.sequence_index,    l.blocks_per_launch,   l.threads_per_block,    l.threads_per_launch,   l.thread_offset  );  

        curandState* states_thread_offset = cs->states  + l.thread_offset ; 
     
        before_kernel( start, stop );

        _QCurandState_curand_init<<<l.blocks_per_launch,l.threads_per_block>>>( l.threads_per_launch, l.thread_offset, d_cs, states_thread_offset  );  

        l.kernel_time = after_kernel( start, stop ); 
    }

} 




