/**
CSGIntersectSolidTest
=======================

Used scripts such as CSG/csg_geochain.sh 

With GEOM envvar : Typically small test solids
------------------------------------------------

When this executable is run with the GEOM envvar defined 
small test geometries are created with CSGFoundry::MakeGeom.

These geometries are created by conversion of G4VSolid with 
GeoChain/translate.sh or directly in CSGSolid/Prim/Node with CSGMaker 
(this is similar to CSGOptiX/cxs_geochain.sh)

Without GEOM envvar : Typically full geometries (can be small too) 
--------------------------------------------------------------------

"Standard" CSGFoundry geometries are loaded from the OPTICKS_KEY 
identified geocache. NB the GGeo geocache must first be converted into 
CSGFoundry by CSG_GGeo/run.sh

* TODO: change to doing the cg translation automatically ? 

Single Solid Intersect Test
------------------------------

Single solid testing of CSG intersects obtained using 
code intended for GPU but running on CPU in order to facilitate 
convenient debugging.

**/

#include "OPTICKS_LOG.hh"
#include "SSys.hh"

#ifdef OLD
#include "Opticks.hh"
#else
#include "SOpticksResource.hh"
#include "SOpticks.hh"
#endif

#include "CSGGeometry.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

#ifdef OLD
    Opticks ok(argc, argv);
    ok.configure();
    const char* cfbase = ok.getFoundryBase("CFBASE") ; 
#else
    const char* cfbase = SOpticksResource::CFBase(); 
#endif

    bool with_geom = SSys::getenvbool("GEOM") ; 
    LOG(info) << " with_geom " << with_geom << " cfbase " << cfbase ;

    CSGGeometry geom(with_geom ? nullptr : cfbase) ;
    geom() ; 

    const char* msg = "CSG/tests/CSGIntersectSolidTest.cc" ; 

#ifdef OLD
    if(cfbase) Opticks::WriteCFBaseScript(cfbase, msg) ; 
#else
    SOpticks::WriteCFBaseScript(cfbase, msg); 
#endif

    return 0 ; 
}


