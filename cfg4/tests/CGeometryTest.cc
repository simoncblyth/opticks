#include <cassert>

#include "CFG4_BODY.hh"

// okc-
#include "Opticks.hh"
#include "OpticksHub.hh"

#include "GGeo.hh"
#include "GMaterialLib.hh"

// cfg4-
#include "CG4.hh"
#include "CGeometry.hh"
#include "CDetector.hh"
#include "CMaterialTable.hh"

//#include "CBorderSurfaceTable.hh"
//#include "CSkinSurfaceTable.hh"

#include "CMaterialBridge.hh"
#include "CSurfaceBridge.hh"

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

    //detector->dumpLV();


/*
    CMaterialTable mt ; 
    mt.dump("CGeometryTest CMaterialTable");
    mt.dumpMaterial("GdDopedLS");
*/

    // these now moved inside CSurfaceBridge
    //CBorderSurfaceTable bst ; 
    //bst.dump("CGeometryTest CBorderSurfaceTable");
    //CSkinSurfaceTable sst ; 
    //sst.dump("CGeometryTest CSkinSurfaceTable");

   
    GGeo* gg = hub.getGGeo();
    GMaterialLib* mlib = gg->getMaterialLib(); 
    GSurfaceLib*  slib = gg->getSurfaceLib(); 

    CMaterialBridge mbr(mlib) ;
    mbr.dump("CGeometryTest"); 

    CSurfaceBridge sbr(slib);
    sbr.dump("CGeometryTest CSurfaceBridge");

    return 0 ; 
}
