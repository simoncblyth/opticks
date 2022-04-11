#include <stdio.h>
#include "scuda.h"
#include "squad.h"
#include "qevent.h"

#include "iexpand.h"
#include "strided_range.h"
#include <thrust/device_vector.h>


__global__ void _QEvent_checkEvt(qevent* evt, unsigned width, unsigned height)
{
    unsigned ix = blockIdx.x*blockDim.x + threadIdx.x;
    if( ix >= width ) return ;  

    unsigned photon_id = ix ; 
    unsigned genstep_id = evt->seed[photon_id] ; 
    const quad6& gs = evt->genstep[genstep_id] ; 
    int gencode = gs.q0.i.x ; 
    unsigned num_photon = evt->num_photon ; 

    printf("//_QEvent_checkEvt width %d height %d photon_id %3d genstep_id %3d  gs.q0.i ( %3d %3d %3d %3d )  gencode %d num_photon %d \n", 
       width,
       height,
       photon_id, 
       genstep_id, 
       gs.q0.i.x, 
       gs.q0.i.y,
       gs.q0.i.z, 
       gs.q0.i.w,
       gencode, 
       num_photon 
      );  
}

extern "C" void QEvent_checkEvt(dim3 numBlocks, dim3 threadsPerBlock, qevent* evt, unsigned width, unsigned height ) 
{
    printf("//QEvent_checkEvt width %d height %d \n", width, height );  
    _QEvent_checkEvt<<<numBlocks,threadsPerBlock>>>( evt, width, height  );
} 

/**
QEvent_count_genstep_photons
-------------------------------

NB this needs nvcc compilation for the thrust but 
the method itself does not run on the device although the 
things that it invokes do. So the qevent* argument needs
to be the CPU side copy of the instance that is holding 
GPU side pointers.

**/

extern "C" unsigned QEvent_count_genstep_photons(qevent* evt)
{
    printf("//QEvent_count_genstep_photons \n");      

    typedef typename thrust::device_vector<int>::iterator Iterator;

    thrust::device_ptr<int> t_gs = thrust::device_pointer_cast( (int*)evt->genstep ) ; 

    const unsigned itemsize = 6*4 ; 
                                          
    strided_range<Iterator> gs_pho( t_gs + 3, t_gs + evt->num_genstep, itemsize );    // begin, end, stride 

    int num_seeds = thrust::reduce(gs_pho.begin(), gs_pho.end() );

    return num_seeds ; 
} 



/*

    QBuf<int>* seed = nullptr ; 

    if( num_seeds > 0 )
    {
        seed = QBuf<int>::Alloc(num_seeds); 
        // TODO: wish to reuse the seed buffer 

        thrust::device_ptr<int> t_seed = thrust::device_pointer_cast((int*)seed->d) ; 

        iexpand(num_pho.begin(), num_pho.end(), t_seed, t_seed + seed->num_items );  
    }
    return seed ; 
*/



