/**
CXRaindropTest : Used from cxs_raindrop.sh 
==============================================

**/

#include <cuda_runtime.h>

#include "NP.hh"
#include "SSys.hh"
#include "SCVD.h"
#include "SPath.hh"
#include "SEventConfig.hh"
#include "SOpticks.hh"
#include "SOpticksResource.hh"
#include "SEvent.hh"
#include "OPTICKS_LOG.hh"

#include "SProp.hh"
#include "SSim.hh"

#include "QBnd.hh"
#include "QSim.hh"
#include "QEvent.hh"

#include "CSGFoundry.h"
#include "CSGOptiX.h"

#ifdef WITH_SGLM
#else
#include "Opticks.hh"
#endif

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    SCVD::ConfigureVisibleDevices(); 
    SEventConfig::SetRGMode("simulate"); 
#ifdef WITH_SGLM
#else
    Opticks::Configure(argc, argv ); 
#endif
    const char* outfold = SEventConfig::OutFold();  
    LOG(info) << " outfold [" << outfold << "]"  ; 
    SOpticks::WriteOutputDirScript(outfold) ; // writes CSGOptiXSimulateTest_OUTPUT_DIR.sh in PWD 


    const char* Rock_Air = "Rock/perfectAbsorbSurface/perfectAbsorbSurface/Air" ; 
    const char* Air_Water = "Air///Water" ; 

    // load standard SSim input arrays $CFBase/CSGFoundry/SSim  and some additional boundaries 
    SSim* ssim = SSim::Load();
    ssim->addFake(Rock_Air, Air_Water); 
    LOG(info) << std::endl << ssim->descOptical()  ; 


    const char* cfbase_local = SSys::getenvvar("CFBASE_LOCAL") ; assert(cfbase_local) ; 
    LOG(fatal) << "MIXING CSGFoundry combining basis cfbase with cfbase_local "; 
    std::cout << std::setw(20) << "cfbase_local" << ":" << cfbase_local  << std::endl ; 

    // load raindrop geometry and customize to use the boundaries added above 
    CSGFoundry* fdl = CSGFoundry::Load(cfbase_local, "CSGFoundry") ; 

    fdl->setOverrideSim(ssim);  
    fdl->setPrimBoundary( 0, Rock_Air ); 
    fdl->setPrimBoundary( 1, Air_Water );   
    // HMM: notice these boundary changes are not persisted 
    std::cout << "fdl.detailPrim " << std::endl << fdl->detailPrim() ; 

    // DIRTY: persisting new bnd and optical into the source directory : FOR USE FROM PYTHON
    ssim->save(cfbase_local, "CSGFoundry/SSim" );   // ONLY APPROPRIATE IN SMALL TESTS




    fdl->upload();  // HMM: WILL CAUSE ASSERT IN CSGOptiX::Create AS CANNOT UPLOAD TWICE 


    float4 ce = make_float4( 0.f, 0.f, 0.f, 100.f );   // TODO: this should come from the geometry 

    SEventConfig::SetMaxExtent( ce.w );  // must do this config before QEvent::init which happens with CSGOptiX instanciation
    SEventConfig::SetMaxTime( 10.f ); 

    CSGOptiX* cx = CSGOptiX::Create(fdl);  

    cx->setGenstep(SEvent::MakeTorchGensteps());     
    cx->simulate();  

    cudaDeviceSynchronize(); 
    cx->event->save(outfold); 

    return 0 ; 
}
