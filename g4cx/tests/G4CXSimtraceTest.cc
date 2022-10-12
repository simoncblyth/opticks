/**
G4CXSimtraceTest.cc :  based on cx:tests/CSGOptiXSimtraceTest.cc
==========================================================================

Canonically used from g4cx/gxt.sh 

**/

#include <cuda_runtime.h>

#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "G4CXOpticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //LOG(info) << "[ cu first " ; 
    cudaDeviceSynchronize(); 
    //LOG(info) << "] cu first " ; 

    SEventConfig::SetRGModeSimtrace();   

    G4CXOpticks gx ;  
    gx.setGeometry(); 
    gx.simtrace(); 

    cudaDeviceSynchronize(); 
    gx.saveEvent();    // $DefaultOutputDir   /tmp/$USER/opticks/SProc::ExecutableName/GEOM  
 
    return 0 ; 
}
