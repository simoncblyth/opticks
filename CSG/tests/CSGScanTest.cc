// ~/o/CSG/tests/CSGScanTest.sh

#include "OPTICKS_LOG.hh"


#include "scuda.h"
#include "SSim.hh"
#include "ssys.h"
#include "spath.h"

#include "CSGFoundry.h"
#include "CSGMaker.h"
#include "CSGSolid.h"
#include "CSGScan.h"


struct CSGScanTest
{
    const char* geom ; 
    const char* scan ; 
    CSGFoundry* fd ; 
    const CSGSolid* so ; 
    CSGScan*  sc ; 

    CSGScanTest(); 
    void init(); 
    int intersect(); 
}; 

inline CSGScanTest::CSGScanTest()
    :
    geom(ssys::getenvvar("GEOM")),
    scan(ssys::getenvvar("SCAN","axis,rectangle,circle")), 
    fd(nullptr),
    so(nullptr),
    sc(nullptr)
{
    init(); 
}; 

inline void CSGScanTest::init()
{
    SSim::Create(); 

    if(CSGMaker::CanMake(geom))
    {
        fd = CSGMaker::MakeGeom(geom);   
        if(ssys::getenvbool("CSGScanTest__init_SAVEFOLD"))
        {  
            fd->save("$CSGScanTest__init_SAVEFOLD");
        }
    }
    else
    {
        fd = CSGFoundry::Load(); 
    }
    fd->upload(); 
    so = fd->getSolid(0);
    // TODO: makes more sense to pick a CSGPrim (or root CSGNode) not a solid

    sc = new CSGScan( fd, so, scan ); 
}

inline int CSGScanTest::intersect()
{
    sc->intersect_h(); 
    sc->intersect_d();
    std::cout << sc->brief() ; 
    sc->save("$FOLD"); 

    // TODO: compare intersects to define rc 
    return 0 ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);    
    CSGScanTest t ; 
    return t.intersect(); 
}

