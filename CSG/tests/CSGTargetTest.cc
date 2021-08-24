/**
CSGTargetTest
==============

MOI=PMT_20inch:0:1 CSGTargetTest 


**/
#include "SSys.hh"
#include "OPTICKS_LOG.hh"

#include "scuda.h"
#include "CSGFoundry.h"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGFoundry* fd = CSGFoundry::Load(SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" ), "CSGFoundry"); 
    LOG(info) << "foundry " << fd->desc() ; 
    //fd->summary(); 

    const char* moi = SSys::getenvvar("MOI", "sWorld:0:0"); 
    int midx, mord, iidx ; 
    fd->parseMOI(midx, mord, iidx,  moi );  
    const char* name = midx > -1 ? fd->getName(midx) : nullptr ; 

    float4 ce = make_float4( 0.f, 0.f, 0.f, 1000.f ); 
    fd->getCenterExtent(ce, midx, mord, iidx );  

    LOG(info) 
        << " MOI " << moi 
        << " midx " << midx 
        << " mord " << mord 
        << " iidx " << iidx
        << " name " << name 
        << " ce " << ce 
        ;   


    return 0 ; 
}



