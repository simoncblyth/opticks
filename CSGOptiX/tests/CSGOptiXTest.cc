/**
CSGOptiXTest : Low level single ray intersect testing 
=======================================================

Focus of this is basic machinery testing, for more detailed testing 
see the other tests. 

**/

#include <cuda_runtime.h>
#include "SPath.hh"
#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

#include "RG.h"
#include "CSGFoundry.h"
#include "CSGOptiX.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv ); 
    ok.configure(); 

    int raygenmode = RG_RENDER ;  

    ok.setRaygenMode(raygenmode) ;  // override --raygenmode option 

    CSGFoundry* fd = CSGFoundry::Load(); 
    fd->upload(); 

    CSGOptiX cx(&ok, fd); 

    float4 ce = make_float4(0.f, 0.f, 0.f, 100.f );  
    cx.setComposition(ce, nullptr, nullptr); 

    if( cx.raygenmode == RG_RENDER )
    {
        double dt = cx.render(); 
        LOG(info) << " dt " << dt ;  
        cx.snap(); 
    }
    else if ( cx.raygenmode == RG_SIMTRACE )
    {

    }
    else if ( cx.raygenmode == RG_SIMULATE )
    {

    }
 
    cudaDeviceSynchronize(); 
    return 0 ; 
}
