/**
CSGOptiXTest : Low level single ray intersect testing 
=======================================================

Focus of this is basic machinery testing, for more detailed testing 
see the other tests. 

**/

#include <cuda_runtime.h>

#include "SRG.h"
#include "SPath.hh"
#include "SSys.hh"
#include "SEventConfig.hh"
#include "OPTICKS_LOG.hh"
#include "SOpticksResource.hh"

#include "CSGFoundry.h"
#include "CSGOptiX.h"


#include "Opticks.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv ); 
    ok.configure(); 

    CSGFoundry* fd = CSGFoundry::Load(); 
    if( fd == nullptr ) return 1 ; 
    fd->upload(); 

    CSGOptiX cx(&ok, fd); 
    float4 ce = make_float4(0.f, 0.f, 0.f, 100.f );  
    cx.setComposition(ce); 

    if( cx.raygenmode == SRG_RENDER )
    {
        cx.render(); 
        cx.snap(); 
    }
    else if ( cx.raygenmode == SRG_SIMTRACE )
    {

    }
    else if ( cx.raygenmode == SRG_SIMULATE )
    {

    }
 
    cudaDeviceSynchronize(); 
    return 0 ; 
}
