/**
CSGOptiXSimulateTest
======================

Canonically used by cxsim.sh combining CFBASE_LOCAL simple test geometry (eg GeoChain) 
with standard CFBASE basis CSGFoundry/SSim input arrays. 

Notice that the basis SSim input arrays are loaded without the standard geometry
using the intentional arms length relationship between CSGFoundry and SSim. 

**/

#include <cuda_runtime.h>

#include "NP.hh"

#include "SSys.hh"
#include "SPath.hh"
#include "SSim.hh"
#include "SOpticks.hh"
#include "OPTICKS_LOG.hh"
#include "SEvent.hh"
#include "SEventConfig.hh"
#include "OpticksGenstep.h"

#include "CSGFoundry.h"
#include "CSGGenstep.h"
#include "CSGOptiX.h"

#include "QSim.hh"
#include "QEvent.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* outfold = SEventConfig::OutFold();  
    LOG(info) << " outfold [" << outfold << "]"  ; 
    SOpticks::WriteOutputDirScript(outfold) ; // writes CSGOptiXSimulateTest_OUTPUT_DIR.sh in PWD 

    const SSim* ssim = SSim::Load() ;  // standard CFBase/CSGFoundry/SSim
    QSim::UploadComponents(ssim); 

    QSim* sim = new QSim ; 
    LOG(info) << sim->desc(); 

    const char* cfbase_local = SSys::getenvvar("CFBASE_LOCAL") ; 
    assert(cfbase_local) ; 
    CSGFoundry* fdl = CSGFoundry::Load(cfbase_local, "CSGFoundry") ; 
    std::cout << std::setw(20) << "fdl.cfbase" << ":" << fdl->cfbase  << std::endl ; 

    CSGOptiX* cx = CSGOptiX::Create(fdl); 

    cx->setGenstep(SEvent::MakeCarrierGensteps());     
    cx->simulate();  

    cudaDeviceSynchronize(); 

    const char* odir = SPath::Resolve(cfbase_local, "CSGOptiXSimulateTest", DIRPATH ); 
    LOG(info) << " odir " << odir ; 
    cx->event->save(odir); 
 
    return 0 ; 
}
