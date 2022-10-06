#include <cstdio>
#include "curand_kernel.h"
#include "qcurandstate.h"

__global__ void _QCurandState_curand_init(int threads_per_launch, int thread_offset, qcurandstate* cs)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;
    curand_init(cs->seed, id+thread_offset, cs->offset, &cs->states[id]);  
}


extern "C" void QCurandState_curand_init(qcurandstate* cs, qcurandstate* d_cs) 
{
    printf("//QCurandState_curand_init cs %p d_cs %p cs.num %llu \n", cs, d_cs, cs->num );  // NB cannot dereference d_cs here, still on CPU  

   // _QCurandState_curand_init<<<numBlocks,threadsPerBlock>>>( width, height  );  
} 



