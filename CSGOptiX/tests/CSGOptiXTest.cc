/**
CSGOptiXTest : Low level single ray intersect testing 
=======================================================

**/

#include <cuda_runtime.h>
#include "SPath.hh"
#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

#include "CSGFoundry.h"
#include "CSGOptiX.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv ); 
    ok.configure(); 
    ok.setRaygenMode(0) ;  // override --raygenmode option 

    CSGFoundry* fd = CSGFoundry::Load(); 
    fd->upload(); 

    CSGOptiX cx(&ok, fd); 

    float4 ce = make_float4(0.f, 0.f, 0.f, 100.f );  
    cx.setComposition(ce, nullptr, nullptr); 

    double dt = cx.render(); 
    LOG(info) << " dt " << dt ;  
    cx.snap(); 
 
    cudaDeviceSynchronize(); 
    return 0 ; 
}
