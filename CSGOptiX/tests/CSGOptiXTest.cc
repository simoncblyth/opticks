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
#include "SOpticksResource.hh"
#include "Opticks.hh"

#include "RG.h"
#include "CSGFoundry.h"
#include "CSGOptiX.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    int raygenmode = RG::Type(SSys::getenvvar("RGMODE", "simulate"));
    LOG(info) 
        << " raygenmode " << raygenmode 
        << " RG::Name(raygenmode) " << RG::Name(raygenmode) 
        ; 

    Opticks ok(argc, argv ); 
    ok.configure(); 
    ok.setRaygenMode(raygenmode); 

    CSGFoundry* fd = CSGFoundry::Load(); 
    if( fd == nullptr ) return 1 ; 
    fd->upload(); 

    CSGOptiX cx(&ok, fd); 
    float4 ce = make_float4(0.f, 0.f, 0.f, 100.f );  
    cx.setComposition(ce); 

    if( cx.raygenmode == RG_RENDER )
    {
        cx.render(); 
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
