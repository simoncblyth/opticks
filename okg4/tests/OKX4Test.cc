// TEST=OKX4Test om-t

#include "SSys.hh"

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
#include "GGeoGLTF.hh"

#include "CGDMLDetector.hh"

#include "X4PhysicalVolume.hh"
#include "X4Sample.hh"

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

G4VPhysicalVolume* make_top(int argc, char** argv)
{
    Opticks* ok = new Opticks(argc, argv); 
    OpticksHub* hub = new OpticksHub(ok);
    OpticksQuery* query = ok->getQuery();
    CGDMLDetector* detector  = new CGDMLDetector(hub, query) ; 
    bool valid = detector->isValid();
    assert(valid);
    G4VPhysicalVolume* top = detector->Construct();
    return top ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    G4VPhysicalVolume* top = make_top(argc, argv); 
    //char c = 's' ; 
    //G4VPhysicalVolume* top = X4Sample::Sample(c) ; 

    assert(top);
    LOG(info) << "///////////////////////////////// " ; 



    const char* key = X4PhysicalVolume::Key(top) ; 

    Opticks::SetKey(key);

    LOG(error) << " SetKey " << key  ;   

    const char* argforce = "--tracer --nogeocache --xanalytic" ;
    // --nogeoache to prevent GGeo booting from cache 

    Opticks* ok2 = new Opticks(argc, argv, argforce);  // Opticks instanciation must be after Opticks::SetKey
    ok2->configure();

    GGeo* gg2 = new GGeo(ok2) ;
    assert(gg2->getMaterialLib());

    LOG(info) << " gg2 " << gg2 
              << " gg2.mlib " << gg2->getMaterialLib()
              ;


    X4PhysicalVolume xtop(gg2, top) ;    // populates gg2 


    xtop.dumpSolidRec(); 
    xtop.writeSolidRec(); 

    int root = SSys::getenvint( "GLTF_ROOT", 3147 ); 
    const char* path = NULL ;  
    xtop.saveAsGLTF(root, path); 

    GGeoGLTF::Save(gg2, "$TMP/ggeo.gltf", root ); 


    gg2->prepare();   // merging meshes, closing libs

   // not OKG4Mgr as no need for CG4 

    OKMgr mgr(argc, argv);  // OpticksHub inside here picks up the gg2 (last GGeo instanciated) via GGeo::GetInstance 
    //mgr.propagate();
    mgr.visualize();   

    assert( GGeo::GetInstance() == gg2 );
    gg2->reportMeshUsage();
    gg2->save();

    Opticks* oki = Opticks::GetInstance() ; 

    assert( oki == ok2 ) ; 

    std::cout << " oki " << oki
              << " ok2 " << ok2 
              << std::endl ; 

    LOG(info) << " oki.idpath " << oki->getIdPath() ; 
    LOG(info) << " ok2.idpath " << ok2->getIdPath() ; 

 

    return mgr.rc() ;

}
