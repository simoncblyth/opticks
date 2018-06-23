// TEST=OKX4Test om-t

#include "Opticks.hh"     
#include "OpticksQuery.hh"
#include "OpticksCfg.hh"

#include "OpticksHub.hh" 
#include "OpticksIdx.hh" 
#include "OpticksViz.hh"

#include "GBndLib.hh"
#include "GGeo.hh"

#include "CGDMLDetector.hh"

#include "X4PhysicalVolume.hh"

class G4VPhysicalVolume ;


#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //// the point of the first Opticks etc.. is just to provide top 
    //// it would be better to do that with pure Geant4 
    //// but CGDMLDetector does some fixup for the lacking GDML
    ////
    //// Hmm maybe by exporting GDML from a newer Geant4, 
    //// can avoid the need for the fixup ? And these complications.

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

    const char* argforce = "--target 3199" ;

    Opticks* ok2 = new Opticks(0,0, argforce);  // Opticks instanciation must be after Opticks::SetKey

    ok2->configure();


    GGeo* gg2 = new GGeo(ok2) ;

    X4PhysicalVolume xtop(gg2, top) ;    // populates gg2 

    //xtop.saveAsGLTF(); 

    //GBndLib* blib = gg2->getBndLib();
    //blib->dump(); 


    gg2->prepare();   // merging meshes


    OpticksHub hub2(ok2, gg2 );

    OpticksIdx idx2(&hub2) ;

    OpticksViz viz2(&hub2, &idx2, true);    // true: load/create Bookmarks, setup shaders, upload geometry immediately 
 
    viz2.visualize();


    return 0 ;
}
