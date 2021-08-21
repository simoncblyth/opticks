#include <stdio.h>
#include "scuda.h"
#include "squad.h"
#include "qevent.h"

__global__ void _QEvent_checkEvt(qevent* evt, unsigned width, unsigned height)
{
    unsigned ix = blockIdx.x*blockDim.x + threadIdx.x;
    if( ix >= width ) return ;  

    unsigned photon_id = ix ; 
    unsigned genstep_id = evt->se[photon_id] ; 
    const quad6& gs = evt->gs[genstep_id] ; 
    int gencode = gs.q0.i.x ; 

    printf("//_QEvent_checkEvt width %d height %d photon_id %3d genstep_id %3d  gs.q0.i ( %3d %3d %3d %3d )  gencode %d \n", 
       width,
       height,
       photon_id, 
       genstep_id, 
       gs.q0.i.x, 
       gs.q0.i.y,
       gs.q0.i.z, 
       gs.q0.i.w,
       gencode 
      );  
}

extern "C" void QEvent_checkEvt(dim3 numBlocks, dim3 threadsPerBlock, qevent* evt, unsigned width, unsigned height ) 
{
    printf("//QEvent_checkEvt width %d height %d \n", width, height );  
    _QEvent_checkEvt<<<numBlocks,threadsPerBlock>>>( evt, width, height  );
} 


