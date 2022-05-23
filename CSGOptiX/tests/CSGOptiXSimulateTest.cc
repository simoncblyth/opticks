/**
CSGOptiXSimulateTest
======================

Canonically used by cxsimulate.sh combining CFBASE_LOCAL simple test geometry (eg GeoChain) 
with standard CFBASE basis CSGFoundry/SSim input arrays. 

Notice that the standard SSim input arrays are loaded without the corresponding standard geometry
using the intentional arms length (SSim subdirectory/NPFold) relationship between CSGFoundry and SSim. 

**/

#include <cuda_runtime.h>

#include "SSim.hh"
#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "QSim.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt evt ; 

    const SSim* ssim = SSim::Load() ;  // standard $CFBase/CSGFoundry/SSim

    CSGFoundry* fdl = CSGFoundry::Load("$CFBASE_LOCAL", "CSGFoundry") ;  // local geometry 

    fdl->setOverrideSim(ssim);    // local geometry with standard SSim inputs 

    CSGOptiX* cx = CSGOptiX::Create(fdl);  // uploads geometry, instanciates QSim 

    QSim* qs = cx->sim ; 

    SEvt::AddCarrierGenstep(); 

    qs->simulate(); 

    cudaDeviceSynchronize(); 

    qs->save();
 
    return 0 ; 
}
