/**
G4CXSimulateTest.cc
====================

GGeo creation done when starting from gdml or live G4, still needs Opticks instance,  
TODO: avoid this by replacing with automated SOpticks instanciated by OPTICKS_LOG
and reimplemnting geometry translation in a new more direct "Geo" package  

**/

#include <cuda_runtime.h>
#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "SEvt.hh"
#include "Opticks.hh"   
#include "U4Material.hh"
#include "G4CXOpticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    Opticks::Configure(argc, argv, "--gparts_transform_offset" );  

    U4Material::LoadBnd();  
    // TODO: relocate inside G4CXOpticks::setGeometry
    // create G4 materials from SSim::Load bnd.npy, used by U4VolumeMaker::PV PMTSim
    // HMM: probably dont want to do this when running from GDML

    SEventConfig::SetRGModeSimulate();    // HMM: maybe discern from the simulate/simtrace/render call ? is it needed ahead of that ?
    SEventConfig::SetStandardFullDebug(); // controls which and dimensions of SEvt arrays 

    SEvt evt ;    
    evt.setReldir("ALL"); 

    G4CXOpticks gx ;  
    gx.setGeometry(); 
    gx.simulate(); 

    cudaDeviceSynchronize(); 
    evt.save();    // $DefaultOutputDir   /tmp/$USER/opticks/SProc::ExecutableName/GEOM  then ALL from setRelDir
 
    return 0 ; 
}
