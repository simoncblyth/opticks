#include "CFG4_BODY.hh"
#include <map>

// brap-
#include "BStr.hh"

// okc-
#include "OpticksHub.hh"
#include "Opticks.hh"

// npy-
#include "NGLM.hpp"
#include "NNode.hpp"
#include "NBox.hpp"
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


const plog::Severity CTestDetector::LEVEL = error ; 


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
    LOG(LEVEL) << "[" ; 

    if(m_ok->hasOpt("dbgtestgeo"))
    {
        LOG(info) << "CTestDetector::init --dbgtestgeo upping verbosity" ; 
        setVerbosity(1);
    }

    G4VPhysicalVolume* top = makeDetector();

    setTop(top) ;  // <-- kicks off CTraverser

    // no addMPT() ? 
    attachSurfaces();

    hookupSD(); 

    LOG(LEVEL) << "]" ; 
}


G4VPhysicalVolume* CTestDetector::makeDetector()
{
    assert( m_config->isNCSG() );
    //return m_config->isNCSG() ? makeDetector_NCSG() : makeDetector_OLD() ;
    return makeDetector_NCSG() ;
}


/**
CTestDetector::boxCenteringFix
--------------------------------

See notes/issues/tboolean-proxy-g4evt-immediate-absorption.rst

**/

void CTestDetector::boxCenteringFix( glm::vec3& placement, nnode* root  )
{
    assert( 0 && "not using this way " ); 
    assert( root->type == CSG_BOX ) ;  
    nbox* box = (nbox*)root ; 
    if( !box->is_centered() )
    {
        glm::vec3 center = box->center(); 
        LOG(fatal) << " box.center " << gformat(center) ; 
        placement = center ;  
        box->set_centered() ; 
    }   
    assert( box->is_centered() ); 
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

G4VPhysicalVolume* CTestDetector::makeChildVolume(const NCSG* csg, const char* lvn, const char* pvn, G4LogicalVolume* mother, const NCSG* altcsg )
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


    LOG(LEVEL) << " lvn " << lvn ; 

    bool have_unbalanced_alt = csg->is_balanced() && altcsg && !altcsg->is_balanced() ; 
    if(have_unbalanced_alt) 
    { 
        LOG(LEVEL) << " have_unbalanced_alt " ; 
    }

    G4VSolid* solid = CMaker::MakeSolid( have_unbalanced_alt ? altcsg : csg ); 

    G4Transform3D* transform = NULL ;  
    G4ThreeVector placement(0,0,0); 
  
    if(csg->has_placement_translation())
    {
        glm::vec3 tlate = csg->get_placement_translation(); 
        LOG(LEVEL) << " csg.has_placement_translation " << gformat(tlate) ; 
        placement.set( tlate.x, tlate.y, tlate.z ); 
    }
    else if(csg->has_placement_transform())
    {
        glm::mat4 txf = csg->get_placement_transform(); 
        LOG(LEVEL) << " csg.has_placement_transform " << gformat(txf) ; 
    }
    else if(csg->has_root_transform())
    {
        glm::mat4 txf = csg->get_root_transform(); 
        LOG(LEVEL) << " csg.has_root_transform " << gformat(txf) ; 
        transform = CMaker::ConvertTransform(txf); 
    }
    else
    {
        LOG(LEVEL) << " csg no translate or transform " ; 
    }

    G4LogicalVolume* lv = new G4LogicalVolume(solid, const_cast<G4Material*>(material), strdup(lvn), 0,0,0);

    G4VPhysicalVolume* pv = transform ? 
                                        new G4PVPlacement( *transform, lv, strdup(pvn) ,mother,false,0)
                                      :
                                        new G4PVPlacement(0, placement, lv, strdup(pvn) ,mother,false,0)
                                      ;

    LOG(LEVEL) 
          << " csg.spec " << spec 
          << " csg.getRootCSGName " << csg->getRootCSGName() 
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
    return makeChildVolume(csg, lvn, pvn, mother, NULL  );
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

    LOG(LEVEL) 
        << " numVolumes " << numVolumes 
        ;

    NCSG* universe = m_geotest->getUniverse(); // slightly enlarged distinct clone of outer volume   
    assert(universe);
    G4VPhysicalVolume* top = makeVolumeUniverse(universe) ; 
    assert(top); 
    G4LogicalVolume* mother = top->GetLogicalVolume() ; 

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

        const GMesh* altmesh = mesh->getAlt(); 
        const NCSG* altcsg = altmesh ? altmesh->getCSG() : NULL ; 

        LOG(LEVEL) << std::setw(4) << i << " spec " << ( spec ? spec : "NULL" ) ;   
        assert( spec );  

        G4VPhysicalVolume* pv = makeChildVolume( csg , lvn , pvn, mother, altcsg );

        G4LogicalVolume* lv = pv->GetLogicalVolume() ;

        lv->SetVisAttributes( CVis::MakeAtt(1,0,0, false) ) ; 

        if(top == NULL) top = pv ; 
        if(ppv == NULL) ppv = pv ; 

        mother = lv ;  
    }
    return top ; 
}


