/**
CSGFoundry_getCenterExtent_Test.cc
====================================

::

    MOI=solidXJfixture:10 CSGFoundry_getCenterExtent_Test
    MOI=solidXJfixture:9,solidXJfixture:10,solidXJfixture:20,solidXJfixture:30,solidXJfixture:40,88:50 CSGFoundry_getCenterExtent_Test
    ## MOI can be comma delimited

    MOI=88:0 CSGFoundry_getCenterExtent_Test
    ## can use mesh index or letters at start of the mesh name 


    MOI=Hama:0:1000       CSGFoundry_getCenterExtent_Test
    MOI=H:0:1000          CSGFoundry_getCenterExtent_Test
    MOI=H:0:0,H:0:1       CSGFoundry_getCenterExtent_Test

See also CSGTargetTest.cc

**/

#include "SSys.hh"
#include "SStr.hh"
#include "Opticks.hh"
#include "CSGFoundry.h"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    const char* cfbase = ok.getFoundryBase("CFBASE") ; 
    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 

    LOG(info) 
        << "cfbase " << cfbase 
        << "foundry " << fd->desc() 
        ; 
    fd->summary(); 

    const char* MOI = SSys::getenvvar("MOI", "sWorld:0:0"); 
    std::vector<std::string> vmoi ; 
    SStr::Split(MOI, ',',  vmoi );  
    LOG(info) << " MOI " << MOI << " vmoi.size " << vmoi.size() ; 

    qat4 q ; 
    q.init(); 

    for(unsigned i=0 ; i < vmoi.size() ; i++)
    {   
        const char* moi = vmoi[i].c_str() ; 

        int midx, mord, iidx ; 
        fd->parseMOI(midx, mord, iidx,  moi ); 

        float4 gce = make_float4( 0.f, 0.f, 0.f, 0.f ); 
        int rc = fd->getCenterExtent(gce, midx, mord, iidx, &q) ;
        assert( rc == 0 ); 

        std::cout
            << " MOI " << moi 
            << " midx " << midx 
            << " mord " << mord 
            << " iidx " << iidx
            << " gce " << gce 
            << std::endl 
            << " q " << q 
            << std::endl 
            ; 

    }
    return 0 ; 
}




