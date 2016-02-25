// cfg4-

//
//  ggv-;ggv-pmt-test --cfg4
//  ggv-;ggv-pmt-test --cfg4 --load 1
//


#include "Detector.hh"
#include <map>

// npy-
#include "NLog.hpp"
#include "GLMFormat.hpp"

// cfg4-
#include "CMaker.hh"
#include "CPropLib.hh"

// ggeo-
#include "GMaker.hh"
#include "GCache.hh"

#include "GPmt.hh"
#include "GCSG.hh"

#include "GMaterial.hh"
#include "GGeoTestConfig.hh"

// g4-
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


void Detector::init()
{
    m_lib = new CPropLib(m_cache);
    m_maker = new CMaker(m_cache);
}

bool Detector::isPmtInBox()
{
    const char* mode = m_config->getMode();
    return strcmp(mode, "PmtInBox") == 0 ;
}
bool Detector::isBoxInBox()
{
    const char* mode = m_config->getMode();
    return strcmp(mode, "BoxInBox") == 0 ;
}


G4VPhysicalVolume* Detector::Construct()
{
   // analagous to ggeo-/GGeoTest::CreateBoxInBox
   // but need to translate from a surface based geometry spec into a volume based one
   //
   // creates Russian doll geometry layer by layer, starting from the outermost 
   // hooking up mother volume to prior 
   //

    bool is_pib = isPmtInBox() ;
    bool is_bib = isBoxInBox() ;

    LOG(info) << "Detector::Construct"
              << " pib " << is_pib
              << " bib " << is_bib
              ;

    assert( is_pib || is_bib && "Detector::Construct mode not recognized");

    m_config->dump("Detector::Construct");

    unsigned int n = m_config->getNumElements();

    G4VPhysicalVolume* top = NULL ;  
    G4LogicalVolume* mother = NULL ; 

    for(unsigned int i=0 ; i < n ; i++)
    {   
        const char* spec = m_config->getBoundary(i);
        G4Material* material = m_lib->makeInnerMaterial(spec);

        glm::vec4 param = m_config->getParameters(i);
        char shapecode = m_config->getShape(i) ;
        const char* shapename = GMaker::ShapeName(shapecode);

        std::string lvn = CMaker::LVName(shapename);
        std::string pvn = CMaker::PVName(shapename);

        LOG(info) << "Detector::Construct" 
                  << std::setw(2) << i 
                  << std::setw(2) << shapecode 
                  << std::setw(15) << shapename
                  << std::setw(30) << spec
                  << std::setw(20) << gformat(param)
                  ;   

        G4VSolid* solid = m_maker->makeSolid(shapecode, param);  

        G4LogicalVolume* lv = new G4LogicalVolume(solid, material, lvn.c_str(), 0,0,0);
        G4VPhysicalVolume* pv = new G4PVPlacement(0,G4ThreeVector(), lv, pvn.c_str(),mother,false,0);
 
        if(top == NULL)
        {
            top = pv ; 
            setCenterExtent(param.x, param.y, param.z, param.w );
        }
        mother = lv ; 
    }   

    if(is_pib)
    {
        makePMT(mother);
    }

    return top ;  
}


void Detector::makePMT(G4LogicalVolume* container)
{
    // try without creating an explicit node tree 

    NSlice* slice = m_config->getSlice();

    GCSG* csg = m_lib->getPmtCSG(slice);
    
    csg->dump();

    unsigned int ni = csg->getNumItems();
    // ni = 6 ; // just the Pyrex

    LOG(info) << "Detector::makePMT" 
              << " csg items " << ni 
              ; 

    G4LogicalVolume* mother = container ; 

    std::map<unsigned int, G4LogicalVolume*> lvm ; 

    for(unsigned int index=0 ; index < ni ; index++)
    {
        unsigned int nix = csg->getNodeIndex(index); 
        unsigned int pix = csg->getParentIndex(index); 

        LOG(info) << "Detector::makePMT" 
                  << " csg items " << ni 
                  << " index " << std::setw(3) << index 
                  << " nix " << std::setw(3) << nix 
                  << " pix " << std::setw(3) << pix 
                  ; 

        if(nix > 0)  // just the lv, as others are constituents of those that are recursed over
        {
             const char* pvn = csg->getPVName(nix-1);

             G4LogicalVolume* logvol = makeLV(csg, index );

             lvm[nix-1] = logvol ;

             G4LogicalVolume* mother = NULL ; 

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

             LOG(info) << "Detector::makePMT"
                       << " index " << index 
                       << " x " << tlate.x()
                       << " y " << tlate.y()
                       << " z " << tlate.z()
                       ;

             G4bool many = false ; 
             G4int copyNo = 0 ; 

             G4VPhysicalVolume* physvol = new G4PVPlacement(rot, tlate, logvol, pvn, mother, many, copyNo);
        }
    }
}





G4LogicalVolume* Detector::makeLV(GCSG* csg, unsigned int i)
{
    unsigned int ix = csg->getNodeIndex(i); 

    assert(ix > 0);

    const char* matname = csg->getMaterialName(ix - 1) ;

    const char* lvn = csg->getLVName(ix - 1)  ;  

    G4Material* material = m_lib->makeMaterial(matname) ;

    G4VSolid* solid = m_maker->makeSolid(csg, i );

    G4LogicalVolume* logvol = new G4LogicalVolume(solid, material, lvn);

    LOG(info) 
           << "Detector::makeLV "
           << "  i " << std::setw(2) << i  
           << " ix " << std::setw(2) << ix  
           << " lvn " << std::setw(2) << lvn
           << " matname " << matname  
           ;

    return logvol ; 
}

