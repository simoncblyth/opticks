/**
G4CXSimtraceTest.cc :  based on cx:tests/CSGOptiXSimtraceTest.cc
==========================================================================

**/

#include <cuda_runtime.h>

#include "OPTICKS_LOG.hh"
//#include "SSim.hh"
#include "SEventConfig.hh"
#include "G4CXOpticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    LOG(info) << "[ cu first " ; 
    cudaDeviceSynchronize(); 
    LOG(info) << "] cu first " ; 

    SEventConfig::SetRGModeSimtrace();   

    // SEventConfig::SetCompMask("genstep,simtrace");  // should now be automatic from SEventConfig::CompMaskAuto
    // SSim::Create(); moved this into G4CXOpticks::G4CXOpticks to avoid the repetition ?

    G4CXOpticks gx ;  
    gx.setGeometry(); 
    gx.simtrace(); 

    cudaDeviceSynchronize(); 
    gx.saveEvent();    // $DefaultOutputDir   /tmp/$USER/opticks/SProc::ExecutableName/GEOM  
 
    return 0 ; 
}
