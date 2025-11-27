/**
G4CXRenderTest.cc
===================

TODO: investigate if SEvt could be used in render mode too
(eg to save the frame and image files) in order
to make environment more similar in all modes

The SEventConfig::Initialize is needed to SetDevice
otherwise CSGOptiX instanciation is skipped.

**/

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "SEventConfig.hh"
#include "OPTICKS_LOG.hh"
#include "G4CXOpticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

#ifdef WITH_CUDA
    LOG(info) << "[ cu first " ;
    cudaDeviceSynchronize();
    LOG(info) << "] cu first " ;
#endif

    SEventConfig::SetRGModeRender();
    SEventConfig::Initialize();   // for simulation this auto-called from SEvt::SEvt

    LOG(info) << "[ SetGeometry " ;
    G4CXOpticks* gx = G4CXOpticks::SetGeometry() ;  // sensitive to SomGDMLPath, GEOM, CFBASE
    LOG(info) << "] SetGeometry " ;

    gx->render();       // sensitive to MOI, EYE, LOOK, UP

    return 0 ;
}
