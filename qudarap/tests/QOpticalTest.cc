#include <cuda_runtime.h>
#include "scuda.h"
#include "SStr.hh"
#include "SSys.hh"
#include "SPath.hh"
#include "NP.hh"

#include "SOpticksResource.hh"

#include "QOptical.hh"
#include "OPTICKS_LOG.hh"

void test_check( const QOptical& qo )
{
    LOG(info) << qo.desc() ; 
    qo.check(); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* cfbase = SOpticksResource::CFBase("CFBASE") ; 
    LOG(info) << " cfbase " << cfbase ; 

    bool exists = NP::Exists(cfbase, "CSGFoundry/optical.npy") ; 

    NP* optical = exists ? NP::Load(cfbase, "CSGFoundry/optical.npy") : nullptr ; 

    if( optical == nullptr )
    {
        LOG(error) << " creating placeholder optical buffer " ; 
        optical = NP::Make<unsigned>(10, 4) ; 
        optical->fillIndexFlat(); 
    }

    QOptical qo(optical) ; 

    test_check(qo); 

    cudaDeviceSynchronize(); 

    return 0 ; 
}
