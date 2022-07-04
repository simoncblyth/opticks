
#include <cuda_runtime.h>

#include "SEventConfig.hh"
#include "SEvt.hh"
#include "OPTICKS_LOG.hh"
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

    SEventConfig::SetRGMode("simulate");  
    SEventConfig::SetStandardFullDebug(); 
    SEvt evt ;    // SEvt must be instanciated before QEvent

    // GGeo creation done when starting from gdml or live G4, still needs Opticks instance,  
    // TODO: avoid this by replacing with automated SOpticks instanciated by OPTICKS_LOG
    Opticks::Configure(argc, argv, "--gparts_transform_offset" );  


    G4CXOpticks gx ;  

    //gx.setGeometry(SPath::SomeGDMLPath()); 
    //gx.setGeometry(CSGFoundry::Load()); 
    gx.setGeometry( U4VolumeMaker::PV() );   // sensitive to GEOM envvar

    gx.simulate(); 

    cudaDeviceSynchronize(); 
    evt.save(); 
 
    return 0 ; 
}
