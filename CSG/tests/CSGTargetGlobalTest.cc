/**
CSGTargetGlobalTest
=====================

MOI=solidXJfixture:64 CSGTargetGlobalTest

**/
#include "NP.hh"
#include "SPath.hh"
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


    qat4 q0 ; 
    q0.zero(); 

    qat4 q1 ; 
    q1.zero(); 

    float4 ce = make_float4( 0.f, 0.f, 0.f, 1000.f ); 

    const char* moi = SSys::getenvvar("MOI", "solidXJfixture:64"); 

    int midx, max_mord, iidx ; 
    fd->parseMOI(midx, max_mord, iidx,  moi );  
    const char* name = midx > -1 ? fd->getName(midx) : nullptr ; 

    NP* ces = NP::Make<float>(max_mord, 4 ); 
    float* ces_data = ces->values<float>() ;   

    std::cout 
        << " moi " << std::setw(15) << moi
        << " midx " << std::setw(5) << midx 
        << " max_mord " << std::setw(5) << max_mord 
        << " iidx " << std::setw(6) << iidx
        << " name " << std::setw(10) << name 
        << std::endl 
        ;

    assert( iidx == 0 ); 

    for(int mord=0 ; mord < max_mord ; mord++ )
    {
        fd->getCenterExtent(ce, midx, mord, iidx, &q0 );  
        fd->getTransform(q1, midx, mord, iidx ); 
        bool q_match = qat4::compare(q0, q1, 0.f) == 0 ; 

        ces_data[mord*4 + 0] = ce.x ; 
        ces_data[mord*4 + 1] = ce.y ; 
        ces_data[mord*4 + 2] = ce.z ; 
        ces_data[mord*4 + 3] = ce.w ;   

        std::cout 
            << " mord " << std::setw(6) << mord
            << " ce " <<  ce 
            << std::endl 
            ;

        assert( q_match ); 
    }

    const char* base = "$TMP/CSGTargetGlobalTest" ; 
    int create_dirs = 1 ; // 1:filepath 
    const char* path = SPath::Resolve(base, moi, "ce.npy", create_dirs ); 
    LOG(info) << "writing " << path ; 
    ces->save(path); 

    return 0 ; 
}



