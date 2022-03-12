/**
CSGGeometryFromGeocacheTest
=============================

This is for loading geometries converted from GGeo Geocache by CSG_GGeo/run.sh 
For simpler loading of test geometries see CSGGeometryTest.cc


**/

#include "OPTICKS_LOG.hh"

#ifdef WITH_KITCHEN_SINK
#include "Opticks.hh"
#else
#include "SOpticksResource.hh"
#endif

#include "CSGGeometry.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 


#ifdef WITH_KITCHEN_SINK
    Opticks ok(argc, argv);
    ok.configure();
    const char* cfbase = ok.getFoundryBase("CFBASE") ;
#else
    const char* cfbase = SOpticksResource::CFBase("CFBASE") ;  // sensitive to OPTICKS_KEY 
#endif
    LOG(info) << "cfbase " << cfbase ;

    CSGGeometry geom(cfbase) ;
    geom.dump(); 
    geom.draw(); 

    return 0 ; 

}




