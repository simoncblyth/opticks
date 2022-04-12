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

NB this needs nvcc compilation due to the use of thrust but 
the method itself does not run on the device although the 
methods it invokes do run on the device. 

So the qevent* argument must be the CPU side instance 
which must be is holding GPU side pointers.

**/


//#ifdef DEBUG_QEVENT
struct printf_functor
{
    __host__ __device__ void operator()(int x){ printf("printf_functor %d\n", x); }
};
//#endif


/**
QEvent_count_genstep_photons
-----------------------------

Notice how using strided_range needs itemsize stride twice, 
because are grabbing single ints from each quad6 6*4 genstep 

**/


extern "C" unsigned QEvent_count_genstep_photons(qevent* evt)
{
    typedef typename thrust::device_vector<int>::iterator Iterator;

    thrust::device_ptr<int> t_gs = thrust::device_pointer_cast( (int*)evt->genstep ) ; 

#ifdef DEBUG_QEVENT
    printf("//QEvent_count_genstep_photons qevent::genstep_numphoton_offset %d  qevent::genstep_itemsize  %d  \n", 
            qevent::genstep_numphoton_offset, qevent::genstep_itemsize ); 
#endif

    strided_range<Iterator> gs_pho( 
        t_gs + qevent::genstep_numphoton_offset, 
        t_gs + evt->num_genstep*qevent::genstep_itemsize , 
        qevent::genstep_itemsize );    // begin, end, stride 

    evt->num_seed = thrust::reduce(gs_pho.begin(), gs_pho.end() );

#ifdef DEBUG_QEVENT
    //thrust::for_each( gs_pho.begin(), gs_pho.end(), printf_functor() );  
    printf("//QEvent_count_genstep_photons evt.num_genstep %d evt.num_seed %d \n", evt->num_genstep, evt->num_seed ); 
#endif

    return evt->num_seed ; 
} 


/**
QEvent_fill_seed_buffer
-------------------------

Populates seed buffer using the numbers of photons per genstep from the genstep buffer.

See thrustrap/tests/iexpand_stridedTest.cu for the lead up to this

1. use GPU side genstep array to add the numbers of photons
   from each genstep giving the total number of photons and seeds *num_seeds*
   from all the gensteps

2. populate it by repeating genstep indices into it, 
   according to the number of photons in each genstep 
   
t_gs+qevent::genstep_numphoton_offset 
   q0.u.w of the quad6 genstep, which contains the number of photons 
   for this genstep


**/

extern "C" void QEvent_fill_seed_buffer(qevent* evt )
{
    printf("//QEvent_fill_seed_buffer evt.num_genstep %d evt.num_seed %d \n", evt->num_genstep, evt->num_seed );      

    assert( evt->seed && evt->num_seed > 0 ); 

    thrust::device_ptr<int> t_seed = thrust::device_pointer_cast(evt->seed) ; 

    typedef typename thrust::device_vector<int>::iterator Iterator;

    thrust::device_ptr<int> t_gs = thrust::device_pointer_cast( (int*)evt->genstep ) ; 

    strided_range<Iterator> gs_pho( 
           t_gs + qevent::genstep_numphoton_offset, 
           t_gs + evt->num_genstep*qevent::genstep_itemsize, 
           qevent::genstep_itemsize );    // begin, end, stride 


    thrust::for_each( gs_pho.begin(), gs_pho.end(), printf_functor() );  

    iexpand( gs_pho.begin(), gs_pho.end(), t_seed, t_seed + evt->num_seed );  
}

