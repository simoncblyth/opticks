/**
G4CXSimtraceTest.cc :  WIP : drawing on cx:tests/CSGOptiXSimtraceTest.cc
==========================================================================

TODO: arrange for combined G4CXTest.cc that does all three : simulate, simtrace, render

* head in that direction by making the main more high level, and hence converge 

**/

#include <cuda_runtime.h>

#include "OPTICKS_LOG.hh"
#include "SOpticks.hh"


#include "SEventConfig.hh"
#include "SEvt.hh"
#include "Opticks.hh"   
#include "U4Material.hh"
#include "G4CXOpticks.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    U4Material::LoadBnd(); // create G4 materials from SSim::Load bnd.npy, used by U4VolumeMaker::PV PMTSim

    SEventConfig::SetRGModeSimtrace();    // HMM: maybe discern from the simulate/simtrace/render call ? is it needed ahead of that ?
    SEventConfig::SetCompMask("genstep,simtrace"); 

    SOpticks::WriteOutputDirScript() ;   // still needed ?

    SEvt evt ;    

    Opticks::Configure(argc, argv, "--gparts_transform_offset" );  

    G4CXOpticks gx ;  
    gx.setGeometry(); 
    gx.simtrace(); 

    cudaDeviceSynchronize(); 
    evt.save();    // $DefaultOutputDir   /tmp/$USER/opticks/SProc::ExecutableName/GEOM  then ALL from setRelDir
 
    return 0 ; 
}
