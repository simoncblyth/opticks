// TEST=OKX4Test om-t

#include "Opticks.hh"     
#include "OpticksQuery.hh"
#include "OpticksCfg.hh"

#include "OpticksHub.hh" 
#include "OpticksIdx.hh" 
#include "OpticksViz.hh"
#include "OKPropagator.hh"

#include "OKMgr.hh"     

#include "GBndLib.hh"
#include "GGeo.hh"

#include "CGDMLDetector.hh"

#include "X4PhysicalVolume.hh"

class G4VPhysicalVolume ;

#include "OPTICKS_LOG.hh"

/**
OKX4Test : checking direct from G4 conversion, starting from a GDML loaded geometry
=======================================================================================

The first Opticks is there just to work with CGDMLDetector
to load the GDML and apply fixups for missing material property tables
to provide the G4VPhysicalVolume world volume for checking 
the direct from G4.

It would be cleaner to do this with pure G4. Perhaps 
new Geant4 can avoid the fixups ?  Maybe not I think even
current G4 misses optical properties.

See :doc:`../../notes/issues/OKX4Test`

**/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);
    OpticksHub hub(&ok);
    OpticksQuery* query = ok.getQuery();
    CGDMLDetector* detector  = new CGDMLDetector(&hub, query) ; 
    bool valid = detector->isValid();
    assert(valid);
    G4VPhysicalVolume* top = detector->Construct();
    assert(top);

    LOG(info) << "///////////////////////////////// " ; 

    const char* key = X4PhysicalVolume::Key(top) ; 

    Opticks::SetKey(key);

    LOG(error) << " SetKey " << key  ;   

    const char* argforce = "--tracer" ;

    Opticks* ok2 = new Opticks(argc, argv, argforce);  // Opticks instanciation must be after Opticks::SetKey

    ok2->configure();

    GGeo* gg2 = new GGeo(ok2) ;

    X4PhysicalVolume xtop(gg2, top) ;    // populates gg2 
    //xtop.saveAsGLTF(); 

    gg2->prepare();   // merging meshes, closing libs

/*
    // this is done inside OKMgr 

    OpticksHub hub2(ok2);   // <-- this should pick up gg2, not create/load a new one 

    assert( hub2.getGGeo() == gg2 ); 

    OpticksIdx idx2(&hub2) ;

    OpticksViz viz2(&hub2, &idx2, true);    // true: load/create Bookmarks, setup shaders, upload geometry immediately 

    OKPropagator pro2(&hub2, &idx2, &viz2) ;

    viz2.visualize();
*/


   // not OKG4Mgr as no need for CG4 

    OKMgr mgr(argc, argv);  // OpticksHub inside here picks up the gg2 (last GGeo instanciated) via GGeo::GetInstance 
    mgr.propagate();
    mgr.visualize();   

    assert( GGeo::GetInstance() == gg2 );
    gg2->reportMeshUsage();



    return mgr.rc() ;
}
