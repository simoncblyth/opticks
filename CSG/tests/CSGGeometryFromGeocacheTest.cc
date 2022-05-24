/**
CSGGeometryFromGeocacheTest
=============================

This is for loading geometries converted from GGeo Geocache by CSG_GGeo/run.sh 
For simpler loading of test geometries see CSGGeometryTest.cc


**/

#include "OPTICKS_LOG.hh"

#include "SOpticksResource.hh"

#include "CSGGeometry.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* cfbase = SOpticksResource::CFBase() ;  // sensitive to OPTICKS_KEY 
    LOG(info) << "cfbase " << cfbase ;

    CSGGeometry geom(cfbase) ;
    geom.dump(); 
    geom.draw(); 

    return 0 ; 

}




