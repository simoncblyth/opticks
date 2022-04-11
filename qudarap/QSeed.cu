#include <stdio.h>

#include "scuda.h"

#include "QBuf.hh"
#include "iexpand.h"
#include "strided_range.h"
#include <thrust/device_vector.h>

/**
QSeed_create_photon_seeds
---------------------------

See thrustrap/tests/iexpand_stridedTest.cu for the lead up to this

1. use GPU side genstep array to add the numbers of photons
   from each genstep giving the total number of photons and seeds *num_seeds*
   from all the gensteps

2. allocate GPU seed array and populate it by repeating genstep indices into it, 
   according to the number of photons in each genstep 
   
pgs+3 
   q0.u.w of the quad6 genstep, which contains the number of photons 
   for this genstep


HMM: am veering towards splitting up what this does into short QEvent.cc/QEvent.cu methods 

**/

extern QBuf<int>* QSeed_create_photon_seeds(QBuf<float>* gs )
{
    printf("//QSeed_create_photon_seeds \n");      

    typedef typename thrust::device_vector<int>::iterator Iterator;

    thrust::device_ptr<int> pgs = thrust::device_pointer_cast( (int*)gs->d ) ; 

    unsigned itemsize = 6*4 ; 

    strided_range<Iterator> num_pho( pgs + 3, pgs + gs->num_items, itemsize );    // begin, end, stride 

    int num_seeds = thrust::reduce(num_pho.begin(), num_pho.end() );

    QBuf<int>* seed = nullptr ; 

    if( num_seeds > 0 )
    {
        seed = QBuf<int>::Alloc(num_seeds); 
        // TODO: wish to reuse the seed buffer 

        thrust::device_ptr<int> t_seed = thrust::device_pointer_cast((int*)seed->d) ; 

        iexpand(num_pho.begin(), num_pho.end(), t_seed, t_seed + seed->num_items );  
    }
    return seed ; 
} 

