/**
CSGGeometryFromGeocacheTest
=============================

This is for loading geometries converted from GGeo Geocache by CSG_GGeo/run.sh 
For simpler loading of test geometries see CSGGeometryTest.cc


**/

#include "OPTICKS_LOG.hh"

#include "SOpticksResource.hh"
#include "spath.h"

#include "CSGGeometry.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* cfbase0 = SOpticksResource::CFBase() ;  
    const char* cfbase1 = spath::Resolve("$HOME/.opticks/GEOM/$GEOM") ; 
    LOG(info) 
        << "cfbase0 " << cfbase0 
        << "cfbase1 " << cfbase1 
        ;

    CSGGeometry geom(cfbase1) ;
    geom.dump(); 
    LOG(info) << geom.desc(); 

    return 0 ; 

}




