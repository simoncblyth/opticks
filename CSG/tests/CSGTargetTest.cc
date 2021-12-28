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

    MOI=


**/
#include "SSys.hh"
#include "SStr.hh"
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

    //const char* MMOI = SSys::getenvvar("MMOI", "solidXJfixture:0-63"); 
    // TODO: interpret multi-MOI 

    const char* MOI = SSys::getenvvar("MOI", "sWorld:0:0"); 
    std::vector<std::string> vmoi ; 
    SStr::Split(MOI, ',',  vmoi ); 

    LOG(info) << " MOI " << MOI << " vmoi.size " << vmoi.size() ; 

    qat4 q0 ; 
    q0.zero(); 

    qat4 q1 ; 
    q1.zero(); 

    float4 ce = make_float4( 0.f, 0.f, 0.f, 1000.f ); 

    for(unsigned pass=0 ; pass < 3 ; pass++)
    for(unsigned i=0 ; i < vmoi.size() ; i++)
    {
         const char* moi = vmoi[i].c_str() ; 

        int midx, mord, iidx ; 
        fd->parseMOI(midx, mord, iidx,  moi );  
        const char* name = midx > -1 ? fd->getName(midx) : nullptr ; 

        fd->getCenterExtent(ce, midx, mord, iidx, &q0 );  
        fd->getTransform(q1, midx, mord, iidx ); 
        bool q_match = qat4::compare(q0, q1, 0.f) == 0 ; 

        if( pass == 0 )
        {
            std::cout 
                << " moi " << std::setw(15) << moi
                << " midx " << std::setw(5) << midx 
                << " mord " << std::setw(5) << mord 
                << " iidx " << std::setw(6) << iidx
                << " name " << std::setw(10) << name 
                << std::endl 
                ;
        } 
        else if( pass == 1 )
        {
            std::cout 
                << " moi " << std::setw(15) << moi
                << " ce " <<  ce 
                << std::endl 
                ;
        }
        else if( pass == 2 )
        {
            std::cout 
                << " moi " << std::setw(15) << moi
                << " q0 " << q0
                << std::endl 
                ;
        }
        assert( q_match ); 
    }


    return 0 ; 
}



