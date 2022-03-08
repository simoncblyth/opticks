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
    ok.setRaygenMode(1) ; // override --raygenmode option 


    const char* default_geom = "BoxFourBoxUnion" ; 
    int createdirs = 2 ; 
    const char* default_cfbase = SPath::Resolve("$TMP/GeoChain", default_geom, createdirs );   // 2: dirpath
    const char* cfbase = SSys::getenvvar("CFBASE", default_cfbase ); 
    LOG(info) << " cfbase " << cfbase ; 
    
    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    fd->upload(); 

    CSGOptiX cx(&ok, fd); 

 
    cudaDeviceSynchronize(); 
    return 0 ; 
}
