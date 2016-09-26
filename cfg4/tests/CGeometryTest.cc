#include <cassert>

#include "CFG4_BODY.hh"

// okc-
#include "Opticks.hh"
#include "OpticksHub.hh"

// cfg4-
#include "CG4.hh"
#include "CGeometry.hh"
#include "CDetector.hh"
#include "CMaterialTable.hh"
#include "CBorderSurfaceTable.hh"

// g4-
#include "G4VPhysicalVolume.hh"


#include "GGEO_LOG.hh"
#include "CFG4_LOG.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    LOG(info) << argv[0] ;

    CFG4_LOG__ ; 
    GGEO_LOG__ ; 

    Opticks ok(argc, argv);
    OpticksHub hub(&ok) ;
    CGeometry cg(&hub);

    CDetector* detector = cg.getDetector();

    G4VPhysicalVolume* world_pv = detector->Construct();
    assert(world_pv);


    CMaterialTable mt ; 
    mt.dump("CGeometryTest CMaterialTable");

    mt.dumpMaterial("GdDopedLS");


    //CBorderSurfaceTable bst ; 
    //bst.dump("CGeometryTest CBorderSurfaceTable");


    return 0 ; 
}
