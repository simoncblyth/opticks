/**
G4CXRenderTest.cc
===================

TODO: investigate if SEvt could be used in render mode too 
(eg to save the frame and image files) in order
to make environment more similar in all modes

**/
#include <cuda_runtime.h>
#include "SEventConfig.hh"
#include "OPTICKS_LOG.hh"
#include "G4CXOpticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    LOG(info) << "[ cu first " ; 
    cudaDeviceSynchronize(); 
    LOG(info) << "] cu first " ; 

    SEventConfig::SetRGModeRender();  

    G4CXOpticks* gx = G4CXOpticks::SetGeometry() ;  // sensitive to SomGDMLPath, GEOM, CFBASE

    gx->render();       // sensitive to MOI, EYE, LOOK, UP
 
    return 0 ; 
}
