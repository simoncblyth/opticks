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

#ifdef WITH_SGLM
#else
#include "Opticks.hh"
#endif


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

#ifdef WITH_SGLM
#else
    Opticks ok(argc, argv ); 
    ok.configure(); 
#endif

    CSGFoundry* fd = CSGFoundry::Load(); 
    if( fd == nullptr ) return 1 ; 

    CSGOptiX* cx = CSGOptiX::Create(fd); 


    float4 ce = make_float4(0.f, 0.f, 0.f, 100.f );  
    cx->setComposition(ce); 

    if( cx->raygenmode == SRG_RENDER )
    {
        cx->render(); 
        cx->snap(); 
    }
    else if ( cx->raygenmode == SRG_SIMTRACE )
    {

    }
    else if ( cx->raygenmode == SRG_SIMULATE )
    {

    }
 
    cudaDeviceSynchronize(); 
    return 0 ; 
}
