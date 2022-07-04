#include <cuda_runtime.h>
#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "SOpticksResource.hh"
#include "SSys.hh"
#include "SEvt.hh"

// GGeo creation done when starting from gdml or live G4, still needs Opticks instance,  
// TODO: avoid this by replacing with automated SOpticks instanciated by OPTICKS_LOG
#include "Opticks.hh"   

#include "CSGFoundry.h"
#include "U4VolumeMaker.hh"
#include "U4Material.hh"
#include "G4CXOpticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    U4Material::LoadBnd();   // "back" creation of G4 material properties from the Opticks bnd.npy obtained from SSim::Load 
    // this is needed for U4VolumeMaker::PV to find the G4 materials

    SEventConfig::SetRGModeSimulate();  
    SEventConfig::SetStandardFullDebug(); 
    SEvt evt ;    // SEvt must be instanciated before QEvent

    Opticks::Configure(argc, argv, "--gparts_transform_offset" );  

    G4CXOpticks gx ;  

    if(SSys::hasenvvar(SOpticksResource::SomeGDMLPath_))
    {
        gx.setGeometry(SOpticksResource::SomeGDMLPath()); 
    }
    else if(SSys::hasenvvar(SOpticksResource::CFBASE_))
    {
        gx.setGeometry(CSGFoundry::Load()); 
    }
    else if(SSys::hasenvvar("GEOM"))
    {
        gx.setGeometry( U4VolumeMaker::PV() );   
    }
    else
    {
        LOG(fatal) << " failed to setGeometry " ; 
        assert(0); 
    }

    gx.simulate(); 

    cudaDeviceSynchronize(); 
    evt.save(); 
 
    return 0 ; 
}
