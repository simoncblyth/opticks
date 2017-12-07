
#include <iostream> 
#include <iomanip> 


#include <thrust/device_vector.h>
#include <curand_kernel.h> 

#include "SSys.hh"
#include "NPY.hpp"
#include "TRngBuf.hh"
#include "TUtil.hh"

#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    LOG(info) << argv[0] ;

    static const unsigned NI = 100000 ; 
    static const unsigned NJ = 16 ; 
    static const unsigned NK = 16 ; 

    NPY<float>* ox = NPY<float>::make(NI, NJ, NK);
    ox->zero();

    thrust::device_vector<float> dox(NI*NJ*NK);

    CBufSpec spec = make_bufspec<float>(dox); 

    TRngBuf trb(NI, NJ*NK, spec );

    trb.generate(); 

    trb.download<float>(ox, true) ; 
  
    const char* path = "$TMP/TRngBufTest.npy" ; 
    //
    //  import os, numpy as np ; a = np.load(os.path.expandvars("$TMP/TRngBufTest.npy"))

    ox->save(path)  ;

    SSys::npdump(path, "np.float32", NULL, "suppress=True,precision=8" );

    cudaDeviceSynchronize();  
}


