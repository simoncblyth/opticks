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
#include "QEvent.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt evt ; 

    CSGFoundry* fd = CSGFoundry::Load() ;  // standard OPTICKS_KEY CFBase/CSGFoundry geometry and SSim

    CSGOptiX* cx = CSGOptiX::Create(fd);   // uploads geometry, instanciates QSim 

    SEvt::AddCarrierGenstep(); 

    cx->simulate(); 

    cudaDeviceSynchronize(); 

    cx->event->save();
 
    return 0 ; 
}
