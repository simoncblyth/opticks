/**
G4CXSimulateTest.cc
====================

GGeo creation done when starting from gdml or live G4, still needs Opticks instance,  
TODO: avoid this by replacing with automated SOpticks instanciated by OPTICKS_LOG
and reimplemnting geometry translation in a new more direct "Geo" package  

Formerly did "U4Material::LoadBnd()" here in main 
which created G4 materials from the persisted bnd.npy 
that requires CFBASE or OPTICKS_KEY to derive it
in order to load SSim from $CFBase/CSGFoundry/SSim)

Clearly that is not acceptable when running from live Geant4 or GDML,
so cannot U4Material::LoadBnd from main. 

The new workflow equivalent would be to access the SSim instance
and get the bnd.npy arrays that way.  

**/

#include <cuda_runtime.h>
#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "Opticks.hh"   
#include "G4CXOpticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    Opticks::Configure(argc, argv, "--gparts_transform_offset --allownokey" );  

    SEventConfig::SetRGModeSimulate();    // HMM: maybe discern from the simulate/simtrace/render call ? is it needed ahead of that ?
    SEventConfig::SetStandardFullDebug(); // controls which and dimensions of SEvt arrays 

    G4CXOpticks gx ;  
    gx.setGeometry(); 
#ifdef __APPLE__
    return 0 ; 
#endif
    gx.simulate(); 

    cudaDeviceSynchronize(); 
    gx.save();    // $DefaultOutputDir   /tmp/$USER/opticks/$GEOM/SProc::ExecutableName  then ALL from setRelDir
 
    return 0 ; 
}
