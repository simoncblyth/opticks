#include "CFG4_BODY.hh"
#include <map>

// brap-
#include "BStr.hh"

// okc-
#include "OpticksHub.hh"
#include "Opticks.hh"

// npy-
#include "NGLM.hpp"
#include "NCSG.hpp"
#include "NCSGList.hpp"
#include "GLMFormat.hpp"
#include "NGeoTestConfig.hpp"

// ggeo-
#include "GMaker.hh"
#include "GMaterial.hh"
#include "GGeoTest.hh"
#include "GGeo.hh"
#include "GMergedMesh.hh"
#include "GNodeLib.hh"
#include "GVolume.hh"
#include "GMesh.hh"

// g4-
#include "CFG4_PUSH.hh"


#include "G4Material.hh"
#include "G4LogicalVolume.hh"
#include "G4ThreeVector.hh"
#include "G4PVPlacement.hh"

#include "globals.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include "CFG4_POP.hh"

// cfg4-
#include "CMaker.hh"
#include "CBndLib.hh"
#include "CMaterialLib.hh"
#include "CTestDetector.hh"
#include "CVis.hh"

#include "PLOG.hh"


CTestDetector::CTestDetector(OpticksHub* hub, OpticksQuery* query, CSensitiveDetector* sd)
    : 
    CDetector(hub, query, sd),
    m_geotest(hub->getGGeoTest()),
    m_config(m_geotest->getConfig())
{
    init();
}


void CTestDetector::init()
{
    LOG(verbose) << "CTestDetector::init" ; 

    if(m_ok->hasOpt("dbgtestgeo"))
    {
        LOG(info) << "CTestDetector::init --dbgtestgeo upping verbosity" ; 
        setVerbosity(1);
    }

    LOG(verbose) << "CTestDetector::init CMaker created" ; 

    G4VPhysicalVolume* top = makeDetector();

    LOG(verbose) << "CTestDetector::init makeDetector DONE" ; 

    setTop(top) ;  // <-- kicks off CTraverser

    // no addMPT() ? 
    attachSurfaces();

    hookupSD(); 

}


G4VPhysicalVolume* CTestDetector::makeDetector()
{
    assert( m_config->isNCSG() );
    //return m_config->isNCSG() ? makeDetector_NCSG() : makeDetector_OLD() ;
    return makeDetector_NCSG() ;
}


/**
CTestDetector::makeChildVolume
-------------------------------

Convert an NCSG tree into G4VSolid within LV, PV structure, using CMaker::MakeSolid for shape specifics.

1. converts GMaterial into G4Material
2. makes G4VSolid from NCSG using CMaker
3. wraps the solid into Geant4 LV, PV structure


m_blib 
    CBndLib instance from CDetector base that contains a GBndLib instance

**/

G4VPhysicalVolume* CTestDetector::makeChildVolume(const NCSG* csg, const char* lvn, const char* pvn, G4LogicalVolume* mother )
{
    assert( csg );
    assert( lvn );
    assert( pvn );
  
    const char* spec = csg->getBoundary();
    assert( spec );  

    unsigned boundary = m_blib->addBoundary(spec); 
    // this should not actually add any new boundaries, that all happened earlier in GGeoTest? 

    GMaterial* imat = m_blib->getInnerMaterial(boundary); 

    const G4Material* material = m_mlib->convertMaterial(imat);

    G4VSolid* solid = CMaker::MakeSolid( csg ); 

    G4LogicalVolume* lv = new G4LogicalVolume(solid, const_cast<G4Material*>(material), strdup(lvn), 0,0,0);

    G4VPhysicalVolume* pv = new G4PVPlacement(0,G4ThreeVector(), lv, strdup(pvn) ,mother,false,0);

    LOG(fatal) 
          << " csg.spec " << spec 
          << " boundary " << boundary 
          << " mother " << ( mother ? mother->GetName() : "-" )
          << " lv " << ( lv ? lv->GetName() : "-" )
          << " pv " << ( pv ? pv->GetName() : "-" )
          << " mat " << ( material ? material->GetName() : "-" )
          ;

    return pv ; 
}

G4VPhysicalVolume* CTestDetector::makeVolumeUniverse(const NCSG* csg)
{
    const char* lvn = GGeoTest::UNIVERSE_LV ; 
    const char* pvn = GGeoTest::UNIVERSE_PV ; 
    G4LogicalVolume* mother = NULL ; 
    return makeChildVolume(csg, lvn, pvn, mother  );
}



/**
CTestDetector::makeDetector_NCSG
---------------------------------

Converts the list of GVolumes obtained from GNodeLib, 
which are assumed to have a simple Russian-doll geometry into a Geant4
volume "tree" structure. 

**/

G4VPhysicalVolume* CTestDetector::makeDetector_NCSG()
{
    GNodeLib* nolib = m_geotest->getNodeLib();
    assert( nolib );
    unsigned numVolumes = nolib->getNumVolumes();

    LOG(info) 
        << " numVolumes " << numVolumes 
        ;

    NCSG* universe = m_geotest->getUniverse();
    assert(universe);
    G4VPhysicalVolume* top = universe ? makeVolumeUniverse(universe) : NULL ; 
    G4LogicalVolume* mother = top ? top->GetLogicalVolume() : NULL ; 

    if(mother)
    {
        mother->SetVisAttributes (CVis::MakeInvisible());
    }
    
    G4VPhysicalVolume* ppv = NULL ; 



    for(unsigned i=0 ; i < numVolumes ; i++) 
    {
        GVolume* kso = nolib->getVolume(i); 
        const char* lvn = kso->getLVName();
        const char* pvn = kso->getPVName();
        const GMesh* mesh = kso->getMesh();
        const NCSG* csg = mesh->getCSG();
        const char* spec = csg->getBoundary(); 
        LOG(info) << std::setw(4) << i << " spec " << ( spec ? spec : "NULL" ) ;   
        assert( spec );  

        G4VPhysicalVolume* pv = makeChildVolume( csg , lvn , pvn, mother );

        G4LogicalVolume* lv = pv->GetLogicalVolume() ;

        lv->SetVisAttributes( CVis::MakeAtt(1,0,0, false) ) ; 

        if(top == NULL) top = pv ; 
        if(ppv == NULL) ppv = pv ; 

        mother = lv ;  
    }
    return top ; 
}


