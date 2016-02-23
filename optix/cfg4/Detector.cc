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
#include "G4RotationMatrix.hh"


#include "G4UnionSolid.hh"
#include "G4IntersectionSolid.hh"



void Detector::init()
{
    bool constituents ; 
    m_bndlib = GBndLib::load(m_cache, constituents=true);
    m_mlib = m_bndlib->getMaterialLib();
    m_slib = m_bndlib->getSurfaceLib();

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
    return makeMaterialPropertiesTable(kmat);
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

G4Material* Detector::makeMaterial(unsigned int index)
{
    GMaterial* kmat = m_mlib->getMaterial(index);
    return makeMaterial(kmat);
}

G4Material* Detector::convertMaterial(GMaterial* kmat)
{
    G4double z, a, density ;

    const char* name = kmat->getShortName();

    LOG(warning) << "Detector::convertMaterial  " << name ;

    G4Material* material = new G4Material(name, z=1., a=1.01*g/mole, density=universe_mean_density );

    // presumably z, a and density are not relevant for optical photons 

    return material ;  
}


G4Material* Detector::makeMaterial(GMaterial* kmat)
{
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
        material = convertMaterial(kmat) ;
        //assert(0);
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
    bool is_bib = isPmtInBox() ;

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

    LOG(info) << "Detector::makePMT" 
              << " csg items " << ni 
              ; 

    G4LogicalVolume* mother = container ; 

    std::map<unsigned int, G4LogicalVolume*> lvm ; 

    for(unsigned int i=0 ; i < ni ; i++)
    {
        unsigned int ix = csg->getNodeIndex(i); 
        unsigned int px = csg->getParentIndex(i); 

        LOG(info) << "Detector::makePMT" 
                  << " csg items " << ni 
                  << " i " << std::setw(3) << i 
                  << " ix " << std::setw(3) << ix 
                  << " px " << std::setw(3) << px 
                  ; 

        if(ix > 0)
        {
             const char* pvn = csg->getPVName(ix-1);

             G4LogicalVolume* logvol = makeLV(csg, i );

             lvm[ix-1] = logvol ;

             G4LogicalVolume* mother = NULL ; 

             if(ix - 1 == 0)
             { 
                 mother = container ;
             }
             else
             {
                 assert( px > 0 && lvm.count(px-1) == 1  );
                 mother = lvm[px-1] ;
             }

             G4VPhysicalVolume* physvol = new G4PVPlacement(0,G4ThreeVector(), logvol, pvn ,mother,false,0);
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

    LOG(info) 
           << "Detector::makeLV "
           << "  i " << std::setw(2) << i  
           << " ix " << std::setw(2) << ix  
           << " lvn " << std::setw(2) << lvn
           << " matname " << matname  
           ;

    G4VSolid* solid = makeSolid(csg, i );
    G4Material* material = makeMaterial(kmat);
    G4LogicalVolume* logvol = new G4LogicalVolume(solid, material, lvn, 0,0,0);
    return logvol ; 
}


G4VSolid* Detector::makeSolid(GCSG* csg, unsigned int i)
{
    unsigned int nc = csg->getNumChildren(i); 
    unsigned int fc = csg->getFirstChildIndex(i); 
    unsigned int lc = csg->getLastChildIndex(i); 
    unsigned int tc = csg->getTypeCode(i);
    const char* tn = csg->getTypeName(i);

    LOG(info) 
           << "Detector::makeSolid "
           << "  i " << std::setw(2) << i  
           << " nc " << std::setw(2) << nc 
           << " fc " << std::setw(2) << fc 
           << " lc " << std::setw(2) << lc 
           << " tc " << std::setw(2) << tc 
           << " tn " << tn 
           ;

   G4VSolid* solid = NULL ; 

   // hmm this is somewhat tied to known structure of DYB PMT
   if(csg->isUnion(i))
   {
       assert(nc == 2);
       std::stringstream ss ; 
       ss << "union-ab" 
          << "-i-" << i 
          << "-fc-" << fc 
          << "-lc-" << lc 
          ;
       std::string ab_name = ss.str();

       G4VSolid* asol = makeSolid(csg, fc );
       G4VSolid* bsol = makeSolid(csg, lc );

       G4UnionSolid* uso = new G4UnionSolid( ab_name.c_str(), asol, bsol  );
       solid = uso ; 
   }
   else if(csg->isIntersection(i))
   {
       assert(nc == 3 && fc + 2 == lc );

       std::string ij_name ;      
       std::string ijk_name ;      

       {
          std::stringstream ss ; 
          ss << "intersection-ij" 
              << "-i-" << i 
              << "-fc-" << fc 
              << "-lc-" << lc 
              ;
          ij_name = ss.str();
       }
  
       {
          std::stringstream ss ; 
          ss << "intersection-ijk" 
              << "-i-" << i 
              << "-fc-" << fc 
              << "-lc-" << lc 
              ;
          ijk_name = ss.str();
       }

       G4VSolid* isol = makeSolid(csg, fc+0 );
       G4VSolid* jsol = makeSolid(csg, fc+1 );
       G4VSolid* ksol = makeSolid(csg, fc+2 );

       G4IntersectionSolid* ij_sol = new G4IntersectionSolid( ij_name.c_str(), isol, jsol  );
       G4IntersectionSolid* ijk_sol = new G4IntersectionSolid( ijk_name.c_str(), ij_sol, ksol  );

       solid = ijk_sol ; 
   } 
   else if(csg->isSphere(i))
   {
        std::stringstream ss ; 
        ss << "sphere" 
              << "-i-" << i 
              ; 

       std::string sp_name = ss.str();

       float inner = csg->getInnerRadius(i);
       float outer = csg->getOuterRadius(i);

       assert(outer > 0 ) ; 

       float startPhi = 0.f ; 
       float deltaPhi = 2.f*pi ; 

       float startTheta = 0.f ; 
       float deltaTheta = 1.f*pi ; 

       solid = new G4Sphere( sp_name.c_str(), inner > 0 ? inner : 0.f , outer, startPhi, deltaPhi, startTheta, deltaTheta  );

   }
   else if(csg->isTubs(i))
   {
        std::stringstream ss ; 
        ss << "tubs" 
              << "-i-" << i 
              ; 

       std::string tb_name = ss.str();
       float inner = 0.f ; // csg->getInnerRadius(i); kludge to avoid rejig as sizeZ occupies innerRadius spot
       float outer = csg->getOuterRadius(i);
       float sizeZ = csg->getSizeZ(i);   // half length
       assert(sizeZ > 0 ) ; 

       float startPhi = 0.f ; 
       float deltaPhi = 2.f*pi ; 

       solid = new G4Tubs( tb_name.c_str(), inner > 0 ? inner : 0.f , outer, sizeZ, startPhi, deltaPhi );

   }
   else
   {
       LOG(warning) << "Detector::makeSolid implementation missing " ; 
   }

   assert(solid) ; 
   return solid ; 
}




