/**
CSGOptiXSimTest
======================

Canonically used by cxsim.sh 

**/

#include <cuda_runtime.h>

#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "SEvt.hh"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "QSim.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* comp = "genstep,photon,hit,domain,record,rec,seq" ;  // NB no "simtrace" here
    SEventConfig::SetGatherComp(comp); 
    SEventConfig::SetSaveComp(comp); 
    // TODO: this now automated ? check that 


    SEvt* evt = SEvt::Create(SEvt::EGPU)  ;  // holds gensteps and output NPFold of component arrays
    SEvt::AddCarrierGenstep();   // normally gensteps added after geometry setup, but can be before in this simple test 

    // TODO: this is missing setFrame

    CSGFoundry* fd = CSGFoundry::Load() ;  // standard OPTICKS_KEY CFBase/CSGFoundry geometry and SSim

    CSGOptiX* cx = CSGOptiX::Create(fd);   // uploads geometry, instanciates QSim 

    QSim* qs = cx->sim ; 

    int eventID = 0 ; 
    bool end = true ; 

    qs->simulate(eventID, end);  // internally calls CSGOptiX::simulate_launch following genstep uploading by QSim

    cudaDeviceSynchronize(); 

    evt->save();  // uses SGeo::LastUploadCFBase_OutDir to place outputs into CFBase/ExecutableName folder sibling to CSGFoundry  
 
    return 0 ; 
}
