
#include <iostream> 
#include <iomanip> 


#include <thrust/device_vector.h>
//#include <curand_kernel.h> 

#include "SSys.hh"
#include "SStr.hh"
#include "BFile.hh"
#include "NPY.hpp"
#include "TRngBuf.hh"
#include "TUtil.hh"

#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ;

    static const unsigned IBASE = SSys::getenvint("TRngBuf_IBASE", 0) ; 
    static const unsigned NI = 100000 ; 
    static const unsigned NJ = 16 ; 
    static const unsigned NK = 16 ; 


    NPY<double>* ox = NPY<double>::make(NI, NJ, NK);

    ox->zero();

    thrust::device_vector<double> dox(NI*NJ*NK);

    CBufSpec spec = make_bufspec<double>(dox); 

    TRngBuf<double> trb(NI, NJ*NK, spec );

    trb.setIBase(IBASE) ; 

    trb.generate(); 

    trb.download<double>(ox, true) ; 



  
    const char* path = SStr::Concat("$TMP/TRngBufTest_", IBASE, ".npy") ; 

    LOG(info) << " save " << path ; 

    ox->save(path)  ;

    std::string spath = BFile::FormPath(path); 

    SSys::npdump(spath.c_str(), "np.float64", NULL, "suppress=True,precision=8" );

    cudaDeviceSynchronize();  
}


