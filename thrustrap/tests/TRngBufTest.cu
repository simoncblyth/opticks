
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

    //static const unsigned N = 1000000 ; 
    static const unsigned N = 100000 ; 
    //static const unsigned N = 1001 ; 

    NPY<float>* ox = NPY<float>::make(N, 4, 4);
    ox->zero();

    thrust::device_vector<float> dox(N*4*4);

    CBufSpec spec = make_bufspec<float>(dox); 

    TRngBuf trb(N, 16, spec );

    trb.generate(); 

    trb.download<float>(ox, true) ; 
  
    const char* path = "$TMP/TRngBufTest.npy" ; 

    ox->save(path)  ;

    SSys::npdump(path, "np.float32", NULL, "suppress=True,precision=8" );

    cudaDeviceSynchronize();  
}


