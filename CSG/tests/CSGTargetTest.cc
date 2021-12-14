/**
CSGTargetTest
==============

::

    MOI=PMT_20inch:0:1 CSGTargetTest 
    MOI=Hama           CSGTargetTest 
    MOI=Hama:0:0       CSGTargetTest 
    MOI=Hama:0:1000    CSGTargetTest 
    MOI=104            CSGTargetTest
    MOI=105            CSGTargetTest
    MOI=sWorld         CSGTargetTest
    MOI=H              CSGTargetTest
    MOI=N              CSGTargetTest
    MOI=P              CSGTargetTest
    MOI=uni_acrylic3:0:100 CSGTargetTest 

**/
#include "SSys.hh"
#include "Opticks.hh"
#include "OPTICKS_LOG.hh"

#include "scuda.h"
#include "sqat4.h"
#include "CSGFoundry.h"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    const char* cfbase = ok.getFoundryBase("CFBASE") ; 
    LOG(info) << "cfbase " << cfbase ; 

    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    LOG(info) << "foundry " << fd->desc() ; 
    //fd->summary(); 

    const char* moi = SSys::getenvvar("MOI", "sWorld:0:0"); 
    int midx, mord, iidx ; 
    fd->parseMOI(midx, mord, iidx,  moi );  
    const char* name = midx > -1 ? fd->getName(midx) : nullptr ; 

    qat4 q0 ; q0.zero(); 
    float4 ce = make_float4( 0.f, 0.f, 0.f, 1000.f ); 

    fd->getCenterExtent(ce, midx, mord, iidx, &q0 );  

    LOG(info) 
        << " MOI " << moi 
        << " midx " << midx 
        << " mord " << mord 
        << " iidx " << iidx
        << " name " << name 
        << std::endl 
        << " ce " << ce 
        << std::endl 
        << " q0 " << q0 
        ;   


    qat4 q1 ; q1.zero(); 

    fd->getTransform(q1, midx, mord, iidx ); 

    LOG(info) << std::endl << "q1" << q1 << std::endl ; 

    assert( qat4::compare(q0, q1, 0.f) == 0 ); 


    return 0 ; 
}



