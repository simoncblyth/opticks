#include <cassert>

#include "CFG4_BODY.hh"

// okc-
#include "Opticks.hh"
#include "OpticksHub.hh"

//#include "GGeo.hh"
#include "GGeoBase.hh"
#include "GBndLib.hh"
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

    CFG4_LOG__ ; 
    GGEO_LOG__ ; 

    LOG(info) << argv[0] ;
    Opticks ok(argc, argv);
    OpticksHub hub(&ok) ;

    CSensitiveDetector* sd = NULL ; 
    CGeometry cg(&hub, sd);

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

   
    GGeoBase* ggb = hub.getGGeoBase();
    GBndLib* blib = ggb->getBndLib(); 
    GMaterialLib* mlib = blib->getMaterialLib(); 
    GSurfaceLib*  slib = blib->getSurfaceLib(); 

    CMaterialBridge mbr(mlib) ;
    mbr.dump("CGeometryTest"); 

    CSurfaceBridge sbr(slib);
    sbr.dump("CGeometryTest CSurfaceBridge");

    return 0 ; 
}
