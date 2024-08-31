/**
CSGFoundry_IntersectPrimTest.cc
================================

1. create small CSGFoundry configured via GEOM using CSGMaker
2. pick a prim (maybe using MOI) 

TODO : 3. intersect_prim on CPU with test rays
TODO : 4. save intersects

* decided to revive CSGScan initially 


HMM : maybe for small geometry tests could instead of operating at 
CSGNode level with CSGMaker
could operate at stree level and do similar to CSGFoundry_CreateFromSimTest.cc ? 

**/

#include "ssys.h"
#include "SSim.hh"
#include "CSGFoundry.h"
#include "CSGMaker.h"
#include "OPTICKS_LOG.hh"


struct CSGFoundry_IntersectPrimTest
{  
    const char* geom ; 
    SSim*      sim ; 
    CSGFoundry* fd ; 
    CSGSolid*   so ; 
 
    CSGFoundry_IntersectPrimTest();
    void init(); 
}; 

inline CSGFoundry_IntersectPrimTest::CSGFoundry_IntersectPrimTest()
    :
    geom(ssys::getenvvar("GEOM", "JustOrb" )),
    sim(SSim::Create()),
    fd(new CSGFoundry), 
    so(nullptr)
{
    init(); 
}

inline void CSGFoundry_IntersectPrimTest::init()
{
    bool can_make = CSGMaker::CanMake(geom) ; 
    if(!can_make) std::cerr << "FATAL : cannot make " << geom << "\n" ; 
    assert(can_make); 

    so = fd->maker->make(geom); 

    fd->setGeom(geom);  
    fd->addTranPlaceholder();  
    fd->addInstancePlaceholder(); 

    // avoid tripping some checks 
    fd->addMeshName(geom);   
    fd->addSolidMMLabel(geom);  
    fd->setMeta<std::string>("source", "CSGFoundry_IntersectPrimTest::init" ); 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGFoundry_IntersectPrimTest t ; 

    std::cout << t.fd->desc() << std::endl ; 

    return 0 ; 
}


