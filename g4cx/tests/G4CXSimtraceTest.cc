/**
G4CXSimtraceTest.cc :  based on cx:tests/CSGOptiXSimtraceTest.cc
==========================================================================

**/

#include <cuda_runtime.h>

#include "OPTICKS_LOG.hh"
#include "Opticks.hh"   
#include "SEventConfig.hh"
#include "G4CXOpticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    Opticks::Configure(argc, argv, "--gparts_transform_offset --allownokey" );  

    SEventConfig::SetRGModeSimtrace();    // HMM: maybe discern from the simulate/simtrace/render call ? is it needed ahead of that ?
    SEventConfig::SetCompMask("genstep,simtrace"); // defaults for this should vary with the RGMode  

    G4CXOpticks gx ;  
    gx.setGeometry(); 
#ifdef __APPLE__
    return 0 ; 
#endif
    gx.simtrace(); 

    cudaDeviceSynchronize(); 
    gx.save();    // $DefaultOutputDir   /tmp/$USER/opticks/SProc::ExecutableName/GEOM  
 
    return 0 ; 
}
