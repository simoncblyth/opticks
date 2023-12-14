/**
CXRaindropTest : Used from cxs_raindrop.sh 
==============================================

**/

#include <cuda_runtime.h>

#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "SEvt.hh"
#include "SSim.hh"
#include "QSim.hh"
#include "CSGFoundry.h"
#include "CSGOptiX.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEventConfig::SetRGModeSimulate(); 
    SEventConfig::SetDebugLite(); 

    SEvt* evt = SEvt::Create(SEvt::EGPU) ; 

    const char* Rock_Air = "Rock/perfectAbsorbSurface/perfectAbsorbSurface/Air" ; 
    const char* Air_Water = "Air///Water" ; 
    SSim* ssim = SSim::Load();
    ssim->addFake(Rock_Air, Air_Water); 
    //LOG(info) << ssim->descOptical()  ; 

    CSGFoundry* fdl = CSGFoundry::Load("$CFBASE_LOCAL", "CSGFoundry") ; 

    fdl->setOverrideSim(ssim);  

    fdl->setPrimBoundary( 0, Rock_Air ); 
    fdl->setPrimBoundary( 1, Air_Water );    // notice these fdl boundary changes are not persisted

    //LOG(info) << "fdl.detailPrim " << std::endl << fdl->detailPrim() ; 

    ssim->save("$CFBASE_LOCAL/CSGFoundry/SSim" ); // DIRTY: FOR PYTHON CONSUMPTION

    float4 ce = make_float4( 0.f, 0.f, 0.f, 100.f );  // TODO: this should come from the geometry 

    SEventConfig::SetMaxExtent( ce.w );  // must do this config before QEvent::init which happens with CSGOptiX instanciation
    SEventConfig::SetMaxTime( 10.f ); 

    CSGOptiX* cx = CSGOptiX::Create(fdl); // encumbent SSim used for QSim setup in here 

    QSim* qs = cx->sim ; 

    if(!SEvt::HasInputPhoton(SEvt::EGPU)) SEvt::AddTorchGenstep();      

    int eventID = 0 ; 
    bool end = true ; 
    qs->simulate(eventID, end);  

    cudaDeviceSynchronize(); 

    evt->save(); 


    return 0 ; 
}
