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

// ggeo-
#include "GMaker.hh"
#include "GCache.hh"
#include "GPmt.hh"
#include "GCSG.hh"

#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GMaterial.hh"
#include "GGeoTestConfig.hh"

// g4-
#include "G4RunManager.hh"
#include "G4NistManager.hh"

#include "G4MaterialTable.hh"
#include "G4Material.hh"

#include "G4Box.hh"
#include "G4Sphere.hh"
#include "G4Tubs.hh"

#include "G4LogicalVolume.hh"
#include "G4ThreeVector.hh"
#include "G4PVPlacement.hh"
#include "globals.hh"
#include "G4UImanager.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"


void Detector::init()
{
    bool constituents ; 
    m_bndlib = GBndLib::load(m_cache, constituents=true);
    m_mlib = m_bndlib->getMaterialLib();
    m_slib = m_bndlib->getSurfaceLib();

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

std::string Detector::LVName(const char* shapename)
{
    std::stringstream ss ; 
    ss << shapename << "_log" ; 
    return ss.str();
}

std::string Detector::PVName(const char* shapename)
{
    std::stringstream ss ; 
    ss << shapename << "_phys" ; 
    return ss.str();
}

 

G4MaterialPropertiesTable* Detector::makeMaterialPropertiesTable(GMaterial* kmat)
{
    unsigned int nprop = kmat->getNumProperties();

    LOG(info) << "Detector::makeMaterialPropertiesTable" 
              << " nprop " << nprop 
              ;   

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();

    for(unsigned int i=0 ; i<nprop ; i++)
    {
        const char* key = kmat->getPropertyNameByIndex(i); // refractive_index absorption_length scattering_length reemission_prob
        const char* lkey = m_mlib->getLocalKey(key) ;      // RINDEX ABSLENGTH RAYLEIGH REEMISSIONPROB
        bool length = strcmp(lkey, "ABSLENGTH") == 0 || strcmp(lkey, "RAYLEIGH") == 0  ;

        GProperty<float>* prop = kmat->getPropertyByIndex(i);
        //prop->Summary(lkey);   

        unsigned int nval  = prop->getLength();

        G4double* ddom = new G4double[nval] ;
        G4double* dval = new G4double[nval] ;

        for(unsigned int j=0 ; j < nval ; j++)
        {
            float fnm = prop->getDomain(j) ;
            float fval = prop->getValue(j) ; 

            G4double wavelength = G4double(fnm)*nm ; 
            G4double energy = h_Planck*c_light/wavelength ;

            G4double value = G4double(fval) ;
            if(length) value *= mm ;    // TODO: check unit consistency, also check absolute-wise

            ddom[nval-1-j] = G4double(energy) ; 
            dval[nval-1-j] = G4double(value) ;
        }

        mpt->AddProperty(lkey, ddom, dval, nval)->SetSpline(true); 

        delete [] ddom ; 
        delete [] dval ; 
    }
    return mpt ;
}



G4Material* Detector::makeWater(const char* name)
{
    G4double z,a;
    G4Element* H  = new G4Element("Hydrogen" ,"H" , z= 1., a=   1.01*g/mole);
    G4Element* O  = new G4Element("Oxygen"   ,"O" , z= 8., a=  16.00*g/mole);

    G4double density;
    G4int ncomponents, natoms;

    G4Material* material = new G4Material(name, density= 1.000*g/cm3, ncomponents=2);
    material->AddElement(H, natoms=2);
    material->AddElement(O, natoms=1);

    return material ; 
}

G4Material* Detector::makeVacuum(const char* name)
{
    G4double z, a, density ;

    // Vacuum standard definition...
    G4Material* material = new G4Material(name, z=1., a=1.01*g/mole, density=universe_mean_density );

    return material ; 
}



G4Material* Detector::convertMaterial(GMaterial* kmat)
{
    const char* name = kmat->getShortName();
    LOG(info) << "Detector::convertMaterial  " << name ;
    G4Material* material(NULL);

    if(strcmp(name,"MainH2OHale")==0)
    {
        material = makeWater(name) ;
    } 
    else if(strcmp(name,"Vacuum")==0)
    {
        material = makeVacuum(name) ;
    }
    else
    {
        G4double z, a, density ;
        // presumably z, a and density are not relevant for optical photons 
        material = new G4Material(name, z=1., a=1.01*g/mole, density=universe_mean_density );
    }

    G4MaterialPropertiesTable* mpt = makeMaterialPropertiesTable(kmat);
    //mpt->DumpTable();

    material->SetMaterialPropertiesTable(mpt);
    return material ;  
}



G4VPhysicalVolume* Detector::Construct()
{
    // analogous to GGeoTest::create

    bool is_pib = isPmtInBox() ;
    bool is_bib = isBoxInBox() ;

    LOG(info) << "Detector::Construct"
              << " pib " << is_pib
              << " bib " << is_bib
              ;
     

    assert( is_pib || is_bib && "Detector::Construct mode not recognized");

   // analagous to ggeo-/GGeoTest::CreateBoxInBox
   // but need to translate from a surface based geometry spec into a volume based one
   //
   // creates Russian doll geometry layer by layer, starting from the outermost 
   // hooking up mother volume to prior 
   //
    m_config->dump("Detector::Construct");

    unsigned int n = m_config->getNumElements();

    G4VPhysicalVolume* top = NULL ;  
    G4LogicalVolume* mother = NULL ; 

    for(unsigned int i=0 ; i < n ; i++)
    {   
        const char* spec = m_config->getBoundary(i);
        unsigned int boundary = m_bndlib->addBoundary(spec);
        unsigned int imat = m_bndlib->getInnerMaterial(boundary);
        GMaterial* kmat = m_mlib->getMaterial(imat);
        G4Material* material = convertMaterial(kmat);

        glm::vec4 param = m_config->getParameters(i);
        char shapecode = m_config->getShape(i) ;
        const char* shapename = GMaker::ShapeName(shapecode);

        std::string lvn = LVName(shapename);
        std::string pvn = PVName(shapename);

        LOG(info) << "Detector::createBoxInBox" 
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
    GPmt* pmt = GPmt::load( m_cache, m_bndlib, 0, slice );    // pmtIndex:0
    GCSG* csg = pmt->getCSG();
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

    GMaterial* kmat = m_mlib->getMaterial(matname) ;

    G4Material* material = convertMaterial(kmat);

    LOG(info) 
           << "Detector::makeLV "
           << "  i " << std::setw(2) << i  
           << " ix " << std::setw(2) << ix  
           << " lvn " << std::setw(2) << lvn
           << " matname " << matname  
           ;

    G4VSolid* solid = m_maker->makeSolid(csg, i );

    G4LogicalVolume* logvol = new G4LogicalVolume(solid, material, lvn);

    return logvol ; 
}

