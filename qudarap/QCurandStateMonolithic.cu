#include <cstdio>
#include "curand_kernel.h"
#include "qcurandwrap.h"
#include "qrng.h"

#include "QUDA_CHECK.h"
#include "SLaunchSequence.h"


__global__ void _QCurandStateMonolithic_curand_init(int threads_per_launch, int id_offset, qcurandwrap* cs, RNG* states_thread_offset )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;
    curand_init(cs->seed, id+id_offset, cs->offset, states_thread_offset + id );  

    //if( id == 0 ) printf("// _QCurandStateMonolithic_curand_init id_offset %d \n", id_offset ); 
}



extern "C" void QCurandStateMonolithic_curand_init(SLaunchSequence* seq,  qcurandwrap* cs, qcurandwrap* d_cs) 
{
    // NB this is still on CPU, dereferencing d_cs here will BUS_ERROR 

    printf("//QCurandStateMonolithic_curand_init seq.items %d cs %p  d_cs %p cs.num %llu \n", seq->items, cs, d_cs, cs->num );  

    cudaEvent_t start, stop ;

    for(unsigned i=0 ; i < seq->launches.size() ; i++)
    {
        SLaunch& l = seq->launches[i] ; 
        printf("// l.sequence_index %d  l.blocks_per_launch %d l.threads_per_block %d  l.threads_per_launch %d l.thread_offset %d  \n", 
                   l.sequence_index,    l.blocks_per_launch,   l.threads_per_block,    l.threads_per_launch,   l.thread_offset  );  

        int id_offset = l.thread_offset ;   

        RNG* states_thread_offset = cs->states  + l.thread_offset ; 
     
        QUDA::before_kernel( start, stop );

        _QCurandStateMonolithic_curand_init<<<l.blocks_per_launch,l.threads_per_block>>>( l.threads_per_launch, id_offset, d_cs, states_thread_offset  );  

        l.kernel_time = QUDA::after_kernel( start, stop ); 
    }
} 


