/**
CSGOptiXSimulateTest
======================

Canonically used by cxsim.sh combining CFBASE_LOCAL simple test geometry (eg GeoChain) 
with standard CFBASE basis CSGFoundry/SSim input arrays. 

Notice that the basis SSim input arrays are loaded without the standard geometry
using the intentional arms length relationship between CSGFoundry and SSim. 

**/

#include <cuda_runtime.h>

#include "SSys.hh"
#include "SPath.hh"
#include "SSim.hh"
#include "SOpticks.hh"
#include "OPTICKS_LOG.hh"
#include "SEvent.hh"
//#include "SEventConfig.hh"

#include "CSGFoundry.h"
#include "CSGOptiX.h"

#include "QSim.hh"
#include "QEvent.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    //SOpticks::WriteOutputDirScript(SEventConfig::OutFold()) ; // writes CSGOptiXSimulateTest_OUTPUT_DIR.sh in PWD 


    const SSim* ssim = SSim::Load() ;  // standard CFBase/CSGFoundry/SSim

    CSGFoundry* fdl = CSGFoundry::Load("$CFBASE_LOCAL", "CSGFoundry") ;  // local geometry 

    fdl->setOverrideSim(ssim);   // local geometry with standard SSim inputs 


    CSGOptiX* cx = CSGOptiX::Create(fdl);  // uploads geometry 

    // TODO: to be more like real running should collect gensteps down to SEvt, 
    //       and then simulate can do the same in test or real running
    //       ie pick up the collected gensteps and launch 

    cx->setGenstep(SEvent::MakeCarrierGensteps());  // Thru to QEvent, uploads and creates seed buffer

    cx->simulate();   // does OptiX launch 

    cudaDeviceSynchronize(); 

    cx->event->save();
 
    return 0 ; 
}
