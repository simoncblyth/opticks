#include <cstdio>
#include "curand_kernel.h"
#include "QUDA_CHECK.h"
#include "SLaunchSequence.h"
#include "scurandref.h"
#include "qrng.h"


/**
_QCurandState_curand_init_chunk
---------------------------------

id 
   [0:threads_per_launch]

states_thread_offset 
   enables multiple launches to write into the correct output slot

**/


__global__ void _QCurandState_curand_init_chunk(int threads_per_launch, int id_offset, scurandref* cr, RNG* states_thread_offset )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;
    curand_init(cr->seed, id+id_offset, cr->offset, states_thread_offset + id );  

    //if( id == 0 ) printf("// _QCurandState_curand_init_chunk id_offset %d \n", id_offset ); 
}


/**
QCurandState_curand_init_chunk
--------------------------------

NB cr and d_cr hold the same values, however cr is host pointer and d_cr is device pointer
cr->states is device pointer, note that pointer arithmetic works on device pointer 

Because are writing states just for a chunk do not need a chunk_offset on the output side,
but do need chunk_offset for the input side. 

**/

extern "C" void QCurandState_curand_init_chunk(SLaunchSequence* seq,  scurandref* cr, scurandref* d_cr) 
{
    // NB this is still on CPU, dereferencing d_cs here will BUS_ERROR 

    printf("//QCurandState_curand_init_chunk seq.items %d cr %p  d_cr %p cr.num %llu cr.chunk_offset %llu \n", seq->items, cr, d_cr, cr->num, cr->chunk_offset );  

    cudaEvent_t start, stop ;

    for(unsigned i=0 ; i < seq->launches.size() ; i++)
    {
        SLaunch& l = seq->launches[i] ; 

        if(0) printf("// l.sequence_index %d  l.blocks_per_launch %d l.threads_per_block %d  l.threads_per_launch %d l.thread_offset %d  \n", 
                         l.sequence_index,    l.blocks_per_launch,   l.threads_per_block,    l.threads_per_launch,   l.thread_offset  );  

        int id_offset = l.thread_offset + cr->chunk_offset ;   

        RNG* states_thread_offset = cr->states  + l.thread_offset ; 
     
        QUDA::before_kernel( start, stop );

        _QCurandState_curand_init_chunk<<<l.blocks_per_launch,l.threads_per_block>>>( l.threads_per_launch, id_offset, d_cr, states_thread_offset  );  

        l.kernel_time = QUDA::after_kernel( start, stop ); 
    }
} 


