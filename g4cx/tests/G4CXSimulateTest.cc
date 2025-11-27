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

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "G4CXOpticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

#ifdef WITH_CUDA
    LOG(info) << "[ cu first " ; 
    cudaDeviceSynchronize(); 
    LOG(info) << "] cu first " ; 
#endif

    SEventConfig::SetDebugLite(); // controls which and dimensions of SEvt arrays 

    G4CXOpticks* gx = G4CXOpticks::SetGeometry() ;  

    int eventID = 0 ; 
    bool end = true ; 

    gx->simulate(eventID, end); 

    return 0 ; 
}
