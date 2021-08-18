#include <stdio.h>

#include "scuda.h"
#include "SBuf.hh"
#include "iexpand.h"
#include "strided_range.h"
#include <thrust/device_vector.h>

/**
QSeed_create_photon_seeds
---------------------------

See thrustrap/tests/iexpand_stridedTest.cu for the lead up to this

**/

extern SBuf<int> QSeed_create_photon_seeds(SBuf<quad6> gs)
{
    printf("//QSeed_create_photon_seeds \n");      

    typedef typename thrust::device_vector<int>::iterator Iterator;

    thrust::device_ptr<int> pgs = thrust::device_pointer_cast( (int*)gs.ptr ) ; 

    strided_range<Iterator> np( pgs + 3, pgs + gs.num_items*6*4, 6*4 );    // begin, end, stride 

    int num_photons = thrust::reduce(np.begin(), np.end() );

    SBuf<int> dseed = SBuf<int>::Alloc(num_photons); 

    thrust::device_ptr<int> pseed = thrust::device_pointer_cast((int*)dseed.ptr) ; 

    iexpand(np.begin(), np.end(), pseed, pseed + dseed.num_items );  

    return dseed ; 
} 


