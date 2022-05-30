/**
CSGOptiXSimTest
======================

Canonically used by cxsim.sh 

**/

#include <cuda_runtime.h>

#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "QSim.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //SEventConfig::SetCompMask("genstep,photon,hit,domain,record,rec,seq")

    SEvt evt ;  // holds gensteps 

    CSGFoundry* fd = CSGFoundry::Load() ;  // standard OPTICKS_KEY CFBase/CSGFoundry geometry and SSim

    CSGOptiX* cx = CSGOptiX::Create(fd);   // uploads geometry, instanciates QSim 

    SEvt::AddCarrierGenstep(); 

    QSim* qs = cx->sim ; 

    qs->simulate(); 

    cudaDeviceSynchronize(); 

    qs->save();  // uses SGeo::LastUploadCFBase_OutDir to place outputs into CFBase/ExecutableName folder sibling to CSGFoundry  
 
    return 0 ; 
}
