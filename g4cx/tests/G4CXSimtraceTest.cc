/**
G4CXSimtraceTest.cc :  based on cx:tests/CSGOptiXSimtraceTest.cc
==========================================================================


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
    SEventConfig::SetCompMask("genstep,simtrace"); // defaults for this should vary with the RGMode  

    SOpticks::WriteOutputDirScript() ;   // still needed ? could be done by SEvt ctor ? 

    SEvt evt ;    

    Opticks::Configure(argc, argv, "--gparts_transform_offset" );  

    G4CXOpticks gx ;  
    gx.setGeometry(); 
    gx.simtrace(); 

    cudaDeviceSynchronize(); 
    evt.save();    // $DefaultOutputDir   /tmp/$USER/opticks/SProc::ExecutableName/GEOM  
 
    return 0 ; 
}
