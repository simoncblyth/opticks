/**
CXRaindropTest : Used from cxs_raindrop.sh 
==============================================

**/

#include <cuda_runtime.h>

#include "OPTICKS_LOG.hh"
#include "SCVD.h"
#include "SEventConfig.hh"
#include "SEvt.hh"
#include "SSim.hh"
#include "QEvent.hh"
#include "CSGFoundry.h"
#include "CSGOptiX.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    SCVD::ConfigureVisibleDevices(); 
    SEventConfig::SetRGMode("simulate"); 

    const char* Rock_Air = "Rock/perfectAbsorbSurface/perfectAbsorbSurface/Air" ; 
    const char* Air_Water = "Air///Water" ; 
    SSim* ssim = SSim::Load();
    ssim->addFake(Rock_Air, Air_Water); 
    LOG(info) << std::endl << ssim->descOptical()  ; 

    CSGFoundry* fdl = CSGFoundry::Load("$CFBASE_LOCAL", "CSGFoundry") ; 

    fdl->setOverrideSim(ssim);  

    fdl->setPrimBoundary( 0, Rock_Air ); 
    fdl->setPrimBoundary( 1, Air_Water );    // notice these fdl boundary changes are not persisted
    std::cout << "fdl.detailPrim " << std::endl << fdl->detailPrim() ; 

    ssim->save("$CFBASE_LOCAL/CSGFoundry/SSim" ); // DIRTY: FOR PYTHON CONSUMPTION

    float4 ce = make_float4( 0.f, 0.f, 0.f, 100.f );  // TODO: this should come from the geometry 

    SEventConfig::SetMaxExtent( ce.w );  // must do this config before QEvent::init which happens with CSGOptiX instanciation
    SEventConfig::SetMaxTime( 10.f ); 

    CSGOptiX* cx = CSGOptiX::Create(fdl); // encumbent SSim used for QSim setup in here 

    SEvt::AddTorchGenstep();      

    cx->simulate();  

    cudaDeviceSynchronize(); 

    cx->event->save();  // TODO: this should talk to QEvent not cx as event handling does not need CSGOptiX 
    return 0 ; 
}
