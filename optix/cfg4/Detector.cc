// cfg4-
#include "Detector.hh"

// npy-
#include "NLog.hpp"
#include "GLMFormat.hpp"

// ggeo-
#include "GMaker.hh"
#include "GPropertyLib.hh"
#include "GCache.hh"
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
#include "G4LogicalVolume.hh"
#include "G4ThreeVector.hh"
#include "G4PVPlacement.hh"
#include "globals.hh"
#include "G4UImanager.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4RotationMatrix.hh"


void Detector::init()
{
    bool constituents ; 
    m_bndlib = GBndLib::load(m_cache, constituents=true);
    m_mlib = m_bndlib->getMaterialLib();
    m_slib = m_bndlib->getSurfaceLib();

    m_boundary_domain = GPropertyLib::getDefaultDomainSpec() ;
}

G4VPhysicalVolume* Detector::Construct()
{
    return CreateBoxInBox();
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

G4VSolid* Detector::makeSphere(const glm::vec4& param)
{
    G4double radius = param.w*mm ; 
    G4Sphere* solid = new G4Sphere("sphere_solid", 0., radius, 0., twopi, 0., pi);  
    return solid ; 
}
G4VSolid* Detector::makeBox(const glm::vec4& param)
{
    G4double extent = param.w*mm ; 
    G4double x = extent;
    G4double y = extent;
    G4double z = extent;
    G4Box* solid = new G4Box("box_solid", x,y,z);
    return solid ; 
}
G4VSolid* Detector::makeSolid(char shapecode, const glm::vec4& param)
{
    G4VSolid* solid = NULL ; 
    switch(shapecode)
    {
        case 'B':solid = makeBox(param);break;
        case 'S':solid = makeSphere(param);break;
    }
    return solid ; 
} 

G4Material* Detector::makeOuterMaterial(const char* spec)
{
    unsigned int boundary = m_bndlib->addBoundary(spec);
    unsigned int om = m_bndlib->getOuterMaterial(boundary);
    return makeMaterial(om);
}

G4Material* Detector::makeInnerMaterial(const char* spec)
{
    unsigned int boundary = m_bndlib->addBoundary(spec);
    unsigned int im = m_bndlib->getInnerMaterial(boundary);
    return makeMaterial(im);
}  

G4MaterialPropertiesTable* Detector::makeMaterialPropertiesTable(unsigned int index)
{
    GMaterial* kmat = m_mlib->getMaterial(index);
    unsigned int nprop = kmat->getNumProperties();

    LOG(info) << "Detector::makeMaterialPropertiesTable" 
              << " index " << index
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

G4Material* Detector::makeMaterial(unsigned int index)
{
    GMaterial* kmat = m_mlib->getMaterial(index);
    //m_mlib->dump(kmat);

    const char* name = kmat->getShortName();

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
        LOG(fatal) << "Detector::makeMaterial not implemented for " << name ;
        assert(0);
    }

    G4MaterialPropertiesTable* mpt = makeMaterialPropertiesTable(index);
    //mpt->DumpTable();
    material->SetMaterialPropertiesTable(mpt);

    return material ; 
}


G4VPhysicalVolume* Detector::CreateBoxInBox()
{
   // analagous to ggeo-/GGeoTest::CreateBoxInBox
   // but need to translate from a surface based geometry spec into a volume based one
   //
    m_config->dump("Detector::CreateBoxInBox");

    unsigned int n = m_config->getNumElements();

    G4VPhysicalVolume* top = NULL ;  
    G4LogicalVolume* mother = NULL ; 

    for(unsigned int i=0 ; i < n ; i++)
    {   
        glm::vec4 param = m_config->getParameters(i);
        const char* spec = m_config->getBoundary(i);
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

        G4VSolid* solid = makeSolid(shapecode, param);  
        G4Material* material = makeInnerMaterial(spec); 

        G4LogicalVolume* lv = new G4LogicalVolume(solid, material, lvn.c_str(), 0,0,0);
        G4VPhysicalVolume* pv = new G4PVPlacement(0,G4ThreeVector(), lv, pvn.c_str(),mother,false,0);
 
        if(top == NULL)
        {
            top = pv ; 
            setCenterExtent(param.x, param.y, param.z, param.w );
        }
        mother = lv ; 
    }   
    return top ;  
}


