#include "CFG4_BODY.hh"
// cfg4-

//
//  ggv-;ggv-pmt-test --cfg4
//  ggv-;ggv-pmt-test --cfg4 --load 1
//

#include <map>

// okc-
#include "OpticksHub.hh"
#include "Opticks.hh"

// npy-
#include "NGLM.hpp"
#include "NCSG.hpp"
#include "NCSGList.hpp"
#include "GLMFormat.hpp"

// ggeo-
#include "GMaker.hh"
#include "GPmtLib.hh"
#include "GPmt.hh"
#include "GCSG.hh"
#include "GMaterial.hh"
#include "GGeoTest.hh"
#include "GGeoTestConfig.hh"
#include "GSur.hh"
#include "GGeo.hh"
#include "GMergedMesh.hh"

#include "GNodeLib.hh"

#include "GSolid.hh"
#include "GMesh.hh"

// g4-
#include "CFG4_PUSH.hh"

#include "G4RunManager.hh"
#include "G4NistManager.hh"
#include "G4MaterialTable.hh"
#include "G4Material.hh"
#include "G4LogicalVolume.hh"
#include "G4ThreeVector.hh"
#include "G4PVPlacement.hh"
#include "G4UImanager.hh"

#include "globals.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include "CFG4_POP.hh"



// cfg4-
#include "CMaker.hh"
#include "CBndLib.hh"
#include "CMaterialLib.hh"
#include "CTestDetector.hh"

#include "PLOG.hh"



CTestDetector::CTestDetector(OpticksHub* hub, OpticksQuery* query)
    : 
    CDetector(hub, query),
    m_geotest(hub->getGGeoTest()),
    m_config(m_geotest->getConfig()),
    m_maker(new CMaker(m_ok))
{
    init();
}


void CTestDetector::init()
{
    LOG(trace) << "CTestDetector::init" ; 

    if(m_ok->hasOpt("dbgtestgeo"))
    {
        LOG(info) << "CTestDetector::init --dbgtestgeo upping verbosity" ; 
        setVerbosity(1);
    }

    LOG(trace) << "CTestDetector::init CMaker created" ; 

    G4VPhysicalVolume* top = makeDetector();

    LOG(trace) << "CTestDetector::init makeDetector DONE" ; 

    setTop(top) ;  // <-- kicks off CTraverser
}


G4VPhysicalVolume* CTestDetector::makeDetector()
{
    return m_config->isNCSG() ? makeDetector_NCSG() : makeDetector_OLD() ;
}


G4VPhysicalVolume* CTestDetector::makeDetector_NCSG()
{
    NCSGList* csglist = m_geotest->getCSGList();
    GNodeLib* nolib = m_geotest->getNodeLib();

    GMergedMesh* tmm = m_geotest->getMergedMesh(0) ;

    assert( csglist );
    assert( nolib );

    unsigned numTrees = csglist->getNumTrees();

    //unsigned numSolids = solist->getNumSolids();
    unsigned numSolids = nolib->getNumSolids();

    LOG(info) << "CTestDetector::makeDetector_NCSG"
              << " numTrees " << numTrees 
              << " numSolids " << numSolids 
              << " tmm " << tmm
              ;

    assert( numSolids == numTrees );

    // contrast with GGeoTest::loadCSG
    std::vector<G4VSolid*> g4solids ; 
   

    G4LogicalVolume* mother = NULL ; 
    G4VPhysicalVolume* ppv = NULL ; 
    G4VPhysicalVolume* top = NULL ; 
     
    for(unsigned i=0 ; i < numTrees ; i++) 
    {
        //unsigned tree = numTrees-1-i ; // now switching order in NCSG::Deserialize to original outermost first
        unsigned tree = i ;
        GSolid* kso = nolib->getSolid(tree); 

        const GMesh* mesh = kso->getMesh();
        const NCSG* csg = mesh->getCSG();
        { 
            const NCSG* csg2 = csglist->getTree(tree);
            assert( csg == csg2 );
        }

        const char* spec = csg->getBoundary();
        unsigned boundary0 = kso->getBoundary();

        // m_blib is CBndLib instance from CDetector base
        //        that contains a GBndLib instance
        //
        // trying to just use boundary0 : find the index invalid 
        // test geometry often uses dynamic omat/osur/isur/imat 
        // combinations that are not persisted to the bndlib
        // so must add the spec : in order to make the boundary index valid
        //

        unsigned boundary = m_blib->addBoundary(spec);

        LOG(info) 
             << " i " << i 
             << " tree " << tree 
             << " boundary0 " << boundary0
             << " boundary " << boundary
             << " csg.bnd " << spec
             ;

        GMaterial* imat = m_blib->getInnerMaterial(boundary); 
        GSur* isur      = m_blib->getInnerSurface(boundary); 
        GSur* osur      = m_blib->getOuterSurface(boundary); 

        //WHY?  just copying _OLD
        if(isur) isur->setBorder();
        if(osur) osur->setBorder();

        if(isur) isur->dump("isur");
        if(osur) osur->dump("osur");

        const G4Material* material = m_mlib->convertMaterial(imat);
        G4VSolid* solid = m_maker->makeSolid( csg ); 

        OpticksCSG_t type = csg->getRootType() ;
        const char* shapename = CSGName(type);


        const char* lvn = kso->getLVName();
        const char* pvn = kso->getPVName();

        assert( lvn );
        assert( pvn );


        //std::string lvn = CMaker::LVName(shapename, i);
        //std::string pvn = CMaker::PVName(shapename, i);

        LOG(info) 
             << " i " << i 
             << " boundary " << boundary
             << " imat " << imat 
             << " isur " << isur 
             << " osur " << osur 
             << " shapename " << shapename
             << " lvn " << lvn 
             << " pvn " << pvn 
             << " mat " << material->GetName()
             ;


        G4LogicalVolume* lv = new G4LogicalVolume(solid, const_cast<G4Material*>(material), strdup(lvn), 0,0,0);
        G4VPhysicalVolume* pv = new G4PVPlacement(0,G4ThreeVector(), lv, strdup(pvn) ,mother,false,0);
 
        if(top == NULL) top = pv ; 
        if(ppv == NULL) ppv = pv ; 
        mother = lv ;  
    }
    return top ; 
}


G4VPhysicalVolume* CTestDetector::makeDetector_OLD()
{
   // analagous to ggeo-/GGeoTest::CreateBoxInBox
   // but need to translate from a surface based geometry spec into a volume based one
   //
   // creates Russian doll geometry layer by layer, starting from the outermost 
   // hooking up mother volume to prior 
   //
    GMergedMesh* mm = m_ggb->getMergedMesh(0);

    unsigned numSolidsMesh = mm->getNumSolids();
    unsigned numSolidsConfig = m_config->getNumElements();

    bool is_pib  = m_config->isPmtInBox() ;
    bool is_bib  = m_config->isBoxInBox() ;

    LOG(info)  << "CTestDetector::makeDetector_OLD "
               << " PmtInBox " << is_pib
               << " BoxInBox " << is_bib
               << " numSolidsMesh " << numSolidsMesh
               << " numSolidsConfig " << numSolidsConfig 
              ;

    assert( ( is_pib || is_bib ) && "CTestDetector::makeDetector_OLD mode not recognized");

    if(is_bib)
    {
        if( numSolidsMesh != numSolidsConfig )
        {
             mm->dumpSolids("CTestDetector::makeDetector_OLD (solid count inconsistent)");
        }
        assert( numSolidsMesh == numSolidsConfig ); // bound to fail for PmtInBox
    }
    else if(is_pib)
    {
    }


    if(m_verbosity > 0)
    m_config->dump("CTestDetector::makeDetector_OLD");


    G4VPhysicalVolume* ppv = NULL ;    // parent pv
    G4VPhysicalVolume* top = NULL ;  
    G4LogicalVolume* mother = NULL ; 


    for(unsigned int i=0 ; i < numSolidsConfig ; i++)
    {   
        const char* spec = m_config->getBoundary(i);

        guint4* ni = mm->getNodeInfo() + i ;
        guint4* id = mm->getIdentity() + i ;

        if(i > 0) ni->w = i - 1  ;  // set parent in test mesh node info, assuming simple Russian doll geometry 

        LOG(info) 
                  << "ni " << ni->description()  
                  << "id " << id->description() 
                  ;

        unsigned boundary0 = id->z ; 
        unsigned boundary = m_blib->addBoundary(spec);

        // nasty mm fixups for GSurLib consumption

        unsigned node0 = ni->z ; 
        if(node0 != i) ni->z = i ; 

        unsigned node2 = id->x ; 
        if(node2 != i) id->x = i ; 

        if(boundary != boundary0)
        {
           LOG(fatal) << "CTestDetector::makeDetector changing boundary "
                      << std::setw(3) << i 
                      << " spec " << spec 
                      << " from boundary0 (from mesh->getNodeInfo()->z ) " << boundary0 
                      << " to boundary (from blib) " << boundary
                      ;
 
           id->z = boundary ;  
        }
        //assert(boundary == boundary0);  

        GMaterial* imat = m_blib->getInnerMaterial(boundary); 
        GSur* isur      = m_blib->getInnerSurface(boundary); 
        GSur* osur      = m_blib->getOuterSurface(boundary); 

        if(isur) isur->setBorder();
        if(osur) osur->setBorder();

        if(isur) isur->dump("isur");
        if(osur) osur->dump("osur");

        LOG(info) 
                  << " spec " << std::setw(50) << spec
                  << " bnd " << std::setw(3) << boundary 
                  << " imat " << std::setw(10) << imat
                  << " isur " << std::setw(10) << isur
                  << " osur " << std::setw(10) << osur
                  ;

       // TODO:
       //    access corresponding GSur and add lvnames and pv1 pv2 indices 
       //    otherwise the surfaces fail to hookup
       //    as the lvn/pv1/pv2 dont match with the modified geometry

        const G4Material* material = m_mlib->convertMaterial(imat);

        glm::vec4 param = m_config->getParameters(i);

        //char nodecode = m_config->getNode(i) ;
        //const char* nodename = CSGChar2Name(nodecode);

        OpticksCSG_t type = m_config->getTypeCode(i) ;
        const char* nodename = CSGName(type);
        std::string lvn = CMaker::LVName(nodename);
        std::string pvn = CMaker::PVName(nodename);

        if(m_verbosity > 0)
        LOG(info) << "CTestDetector::Construct" 
                  << std::setw(2) << i 
                  << std::setw(2) << type
                  << std::setw(15) << nodename
                  << std::setw(30) << spec
                  << std::setw(20) << gformat(param)
                  ;   

        G4VSolid* solid = m_maker->makeSolid_OLD(type, param);  

        G4LogicalVolume* lv = new G4LogicalVolume(solid, const_cast<G4Material*>(material), lvn.c_str(), 0,0,0);

        G4VPhysicalVolume* pv = new G4PVPlacement(0,G4ThreeVector(), lv, pvn.c_str(),mother,false,0);
        
        m_pvm[pvn] = pv ;  

 
        if(top == NULL) top = pv ; 
        if(ppv == NULL) ppv = pv ; 
        mother = lv ; 
    }   

    if(is_pib)
    {
        makePMT_OLD(mother);
    }

    //m_mlib->dumpMaterials("CTestDetector::Construct CPropLib::dumpMaterials");

    
    return top ;  
}




void CTestDetector::makePMT_OLD(G4LogicalVolume* container)
{
    // try without creating an explicit node tree 

    LOG(trace) << "CTestDetector::makePMT_OLD" ; 

    GPmtLib* pmtlib = m_ggb->getPmtLib();
    GPmt* pmt = pmtlib->getLoadedAnalyticPmt();
  
    GCSG* csg = pmt ? pmt->getCSG() : NULL ;

    if(csg == NULL)
    {
        LOG(fatal) << " CTestDetector::makePMT_OLD NULL csg from CPropLib " ;
        setValid(false);
        return ; 
    }   
    
    //if(m_verbosity > 1)
    csg->dump("CTestDetector::makePMT_OLD");

    unsigned int ni = csg->getNumItems();

    //if(m_verbosity > 0)
    LOG(info) << "CTestDetector::makePMT_OLD" 
              << " csg items " << ni 
              ; 

    G4LogicalVolume* mother = container ; 

    std::map<unsigned int, G4LogicalVolume*> lvm ; 

    for(unsigned int index=0 ; index < ni ; index++)
    {
        unsigned int nix = csg->getNodeIndex(index); 
        if(nix == 0) continue ;
        // skip non-lv with nix:0, as those are constituents of the lv that get recursed over

        unsigned int pix = csg->getParentIndex(index); 
        const char* pvn = csg->getPVName(nix-1);

        if(m_verbosity > 0)
        LOG(info) << "CTestDetector::makePMT_OLD" 
                  << " csg items " << ni 
                  << " index " << std::setw(3) << index 
                  << " nix " << std::setw(3) << nix 
                  << " pix " << std::setw(3) << pix 
                  << " pvn " << pvn 
                  ; 

        G4LogicalVolume* logvol = makeLV_OLD(csg, index );

        lvm[nix-1] = logvol ;

        mother = NULL ; 

        if(nix - 1 == 0)
        { 
            mother = container ;
        }
        else
        {
            assert( pix > 0 && lvm.count(pix-1) == 1  );
            mother = lvm[pix-1] ;
        }
              
        G4RotationMatrix* rot = 0 ; 
        G4ThreeVector tlate(csg->getX(index), csg->getY(index), csg->getZ(index));  

        // pv translation, for DYB PMT only non-zero for pvPmtHemiBottom, pvPmtHemiDynode
        // suspect that G4DAE COLLADA export omits/messes this up somehow, for the Bottom at least

        if(m_verbosity > 0)
        LOG(info) << "CTestDetector::makePMT_OLD"
                  << " index " << index 
                  << " x " << tlate.x()
                  << " y " << tlate.y()
                  << " z " << tlate.z()
                  ;

        G4bool many = false ; 
        G4int copyNo = 0 ; 

        G4VPhysicalVolume* physvol = new G4PVPlacement(rot, tlate, logvol, pvn, mother, many, copyNo);

        m_pvm[pvn] = physvol ;  

    }

    kludgePhotoCathode_OLD();
}


void CTestDetector::kludgePhotoCathode_OLD()
{
   // HMM THIS IS A DIRTY KLUDGE ..
   //
   // Over in CGDMLDetector have implemented the GSurLib/CSurLib machinery
   // for revivifying optical surfaces "properly" ...
   //
   // Can that machinery be used here with test geometry ?
   //
   // Possibly but it would be quite involved as border surfaces need pv volume indices 
   // that change for test geometry.
   // Decided too much effort to do this in general when the only surface needed in test geometry for now is the cathode.
   //
   // See :doc:`notes/issues/geant4_opticks_integration/surlib_with_test_geometry` 
   //

    LOG(info) << "CTestDetector::kludgePhotoCathode_OLD" ;

    float effi = 1.f ; 
    float refl = 0.f ; 
    {
        const char* name = "kludgePhotoCathode_PyrexBialkali" ; 
        G4VPhysicalVolume* pv1 = getLocalPV("pvPmtHemi") ; 
        G4VPhysicalVolume* pv2 = getLocalPV("pvPmtHemiCathode") ; 
        assert(pv1 && pv2);
        //G4LogicalBorderSurface* lbs = m_mlib->makeCathodeSurface(name, pv1, pv2, effi, refl );
        G4LogicalBorderSurface* lbs = m_mlib->makeCathodeSurface(name, pv1, pv2);
        assert(lbs);
    }
    {
        const char* name = "kludgePhotoCathode_PyrexVacuum" ; 
        G4VPhysicalVolume* pv1 = getLocalPV("pvPmtHemi") ; 
        G4VPhysicalVolume* pv2 = getLocalPV("pvPmtHemiVacuum") ; 
        assert(pv1 && pv2);
        G4LogicalBorderSurface* lbs = m_mlib->makeConstantSurface(name, pv1, pv2, effi, refl );
        assert(lbs);
    }
}


G4LogicalVolume* CTestDetector::makeLV_OLD(GCSG* csg, unsigned int i)
{
    unsigned int ix = csg->getNodeIndex(i); 

    assert(ix > 0);

    const char* matname = csg->getMaterialName(ix - 1) ;

    const char* lvn = csg->getLVName(ix - 1)  ;  

    const G4Material* material = m_mlib->makeG4Material(matname) ;

    G4VSolid* solid = m_maker->makeSolid_OLD(csg, i );

    G4LogicalVolume* logvol = new G4LogicalVolume(solid, const_cast<G4Material*>(material), lvn);

    if(m_verbosity > 0)
    LOG(info) 
           << "CTestDetector::makeLV_OLD "
           << "  i " << std::setw(2) << i  
           << " ix " << std::setw(2) << ix  
           << " lvn " << std::setw(2) << lvn
           << " matname " << matname  
           ;

    return logvol ; 
}


