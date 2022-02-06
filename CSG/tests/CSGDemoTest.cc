/**
CSGDemoTest.cc
==================

This executable creates and persists simple demo geometries, using the tests/DemoGeo.h struct. 
Invoke the executable with::

    cd ~/opticks/CSG
    ./CSGDemoTest.sh    

To apply the above script to all the demo geometries use ``./make_demos.sh`` 
Render the geometries with::

    cd ~/opticks/CSGOptiX
    ./cxr_demo.sh 

**/

#include "SSys.hh"
#include "OPTICKS_LOG.hh"

#include "scuda.h"
#include "CSGFoundry.h"
#include "DemoGeo.h"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* geom = SSys::getenvvar("GEOM", "sphere" ); 
    CSGFoundry fd ;
    DemoGeo dg(&fd, geom) ;  

    LOG(info) << fd.desc(); 
    LOG(info) << fd.descSolids(); 

    const char* cfbase = SSys::getenvvar("CFBASE", "$TMP/CSGDemoTest/default" );
    const char* rel = "CSGFoundry" ; 

    fd.write(cfbase, rel );    // expects existing directory $CFBASE/CSGFoundry 

    CSGFoundry* fdl = CSGFoundry::Load(cfbase, rel);  // load foundary and check identical bytes
    assert( 0 == CSGFoundry::Compare(&fd, fdl ) );  


    unsigned ias_idx = 0u ; 
    unsigned long long emm = 0ull ;
    LOG(info) << "descInst" << std::endl << fdl->descInst(ias_idx, emm);  

    AABB bb = fdl->iasBB(ias_idx, emm ); 
    float4 ce = bb.center_extent() ;  

    LOG(info) << "bb:" << bb.desc() ; 
    LOG(info) << "ce:" << ce ; 

    std::vector<float3> corners ; 
    AABB::cube_corners(corners, ce ); 
    for(int i=0 ; i < int(corners.size()) ; i++) LOG(info) << corners[i] ;  

    return 0 ; 
}
