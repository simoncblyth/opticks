/**
CSGOptiXSimulateTest
======================

Canonically used by cxsim.sh combining CFBASE_LOCAL simple test geometry (eg GeoChain) 
with standard CFBASE basis CSGFoundry/SSim input arrays. 

Notice that the basis SSim input arrays are loaded without the standard geometry
using the intentional arms length relationship between CSGFoundry and SSim. 

**/

#include <cuda_runtime.h>

#include "SSim.hh"
#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "QEvent.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt evt ; 

    const SSim* ssim = SSim::Load() ;  // standard CFBase/CSGFoundry/SSim

    CSGFoundry* fdl = CSGFoundry::Load("$CFBASE_LOCAL", "CSGFoundry") ;  // local geometry 

    fdl->setOverrideSim(ssim);    // local geometry with standard SSim inputs 

    CSGOptiX* cx = CSGOptiX::Create(fdl);  // uploads geometry, instaciates QSim 

    SEvt::AddCarrierGenstep(); 

    cx->simulate(); 

    cudaDeviceSynchronize(); 

    cx->event->save();
 
    return 0 ; 
}
