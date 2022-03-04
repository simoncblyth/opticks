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


struct CSGTargetTest
{
    Opticks*        ok ; 
    const char* cfbase ; 
    CSGFoundry*     fd ; 
    float4          ce ; 

    qat4 q0 ; 
    qat4 q1 ; 

    CSGTargetTest(int argc, char** argv); 
    void dumpMOI(const char* MOI); 
    void dumpALL(); 
};


CSGTargetTest::CSGTargetTest(int argc, char** argv)
    :
    ok(Opticks::Configure(argc, argv)), 
    cfbase(ok->getFoundryBase("CFBASE")),
    fd(CSGFoundry::Load(cfbase, "CSGFoundry")),
    ce(make_float4( 0.f, 0.f, 0.f, 1000.f ))
{
    LOG(info) << "cfbase " << cfbase ; 
    LOG(info) << "foundry " << fd->desc() ; 
    //fd->summary(); 

    q0.zero(); 
    q1.zero(); 
}

void CSGTargetTest::dumpMOI( const char* MOI )
{
    std::vector<std::string> vmoi ; 
    SStr::Split(MOI, ',',  vmoi ); 

    LOG(info) << " MOI " << MOI << " vmoi.size " << vmoi.size() ; 


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
                << " name " << std::setw(10) << ( name  ? name : "-" )
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
}

void CSGTargetTest::dumpALL()
{
    unsigned num_prim = fd->getNumPrim(); 
    LOG(info) 
         << " fd.getNumPrim " << num_prim 
         << " fd.meshname.size " << fd->meshname.size() 
         ; 

    for(unsigned primIdx=0 ; primIdx < num_prim ; primIdx++)
    {
        const CSGPrim* pr = fd->getPrim(primIdx); 
        unsigned meshIdx = pr->meshIdx();  
        float4 lce = pr->ce();

        std::cout 
            << " primIdx " << std::setw(4) << primIdx 
            << " lce ( "
            << " " << std::setw(10) << std::fixed << std::setprecision(2) << lce.x
            << " " << std::setw(10) << std::fixed << std::setprecision(2) << lce.y
            << " " << std::setw(10) << std::fixed << std::setprecision(2) << lce.z
            << " " << std::setw(10) << std::fixed << std::setprecision(2) << lce.w
            << " )" 
            << " lce.w/1000  "
            << " " << std::setw(10) << std::fixed << std::setprecision(2) << lce.w/1000.f
            << " meshIdx " << std::setw(4) << meshIdx 
            << " " << fd->meshname[meshIdx]
            << std::endl ; 
    }
}





int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGTargetTest tt(argc, argv); 

    const char* MOI = SSys::getenvvar("MOI", nullptr ); 

    if(MOI) 
    {
        tt.dumpMOI(MOI); 
    }
    else
    {
        tt.dumpALL(); 
    }

    return 0 ; 
}



