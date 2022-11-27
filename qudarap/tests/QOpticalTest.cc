/**
QOpticalTest.cc
===================

TODO: combine optical with bnd as they are so closely related it 
makes no sense to treat them separately 


**/
#include <cuda_runtime.h>
#include "scuda.h"
#include "SStr.hh"
#include "SSys.hh"
#include "SPath.hh"
#include "NP.hh"

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

    const char* optical_path = SPath::Resolve("$CFBaseFromGEOM/CSGFoundry/SSim/optical.npy", NOOP ) ;
    NP* optical = NP::Load(optical_path) ; 
   
    LOG(info) 
        << " optical_path " << optical_path
        << " optical " << ( optical ? optical->sstr() : "-" )
        ;

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
