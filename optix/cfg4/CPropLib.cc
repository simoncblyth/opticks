#include "CPropLib.hh"


// ggeo-
#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"

#include "GMaterial.hh"
#include "GPmt.hh"
#include "GCSG.hh"


// npy-
#include "NLog.hpp"


// g4-
#include "G4MaterialTable.hh"
#include "G4Material.hh"
#include "globals.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"



void CPropLib::init()
{
    bool constituents ; 
    m_bndlib = GBndLib::load(m_cache, constituents=true);
    m_mlib = m_bndlib->getMaterialLib();
    m_slib = m_bndlib->getSurfaceLib();
}

G4Material* CPropLib::makeInnerMaterial(const char* spec)
{
    unsigned int boundary = m_bndlib->addBoundary(spec);
    unsigned int imat = m_bndlib->getInnerMaterial(boundary);
    GMaterial* kmat = m_mlib->getMaterial(imat);
    G4Material* material = convertMaterial(kmat);
    return material ; 
}

G4Material* CPropLib::makeMaterial(const char* matname)
{
    GMaterial* kmat = m_mlib->getMaterial(matname) ;
    G4Material* material = convertMaterial(kmat);
    return material ; 
}

GCSG* CPropLib::getPmtCSG(NSlice* slice)
{
    GPmt* pmt = GPmt::load( m_cache, m_bndlib, 0, slice );    // pmtIndex:0
    GCSG* csg = pmt->getCSG();
    return csg ;
}


G4MaterialPropertiesTable* CPropLib::makeMaterialPropertiesTable(GMaterial* kmat)
{
    const char* name = kmat->getShortName();
    unsigned int nprop = kmat->getNumProperties();

    LOG(info) << "CPropLib::makeMaterialPropertiesTable" 
              << " name " << name
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

        LOG(debug) << "CPropLib::makeMaterialPropertiesTable" 
                  << " i " << std::setw(3) << i
                  << " key " << std::setw(40) << key
                  << " lkey " << std::setw(40) << lkey
                  << " nval " << nval 
                 ;   

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



G4Material* CPropLib::makeWater(const char* name)
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

G4Material* CPropLib::makeVacuum(const char* name)
{
    G4double z, a, density ;

    // Vacuum standard definition...
    G4Material* material = new G4Material(name, z=1., a=1.01*g/mole, density=universe_mean_density );

    return material ; 
}



G4Material* CPropLib::convertMaterial(GMaterial* kmat)
{
    const char* name = kmat->getShortName();
    if(m_ggtog4.count(kmat) == 1)
    {
        LOG(info) << "CPropLib::convertMaterial" 
                  << " return preexisting " << name 
                  ;
        return m_ggtog4[kmat] ;
    }

    unsigned int materialIndex = m_mlib->getMaterialIndex(kmat);

    LOG(info) << "CPropLib::convertMaterial  " 
              << " name " << name
              << " materialIndex " << materialIndex
              ;

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

    m_ggtog4[kmat] = material ; 
    m_g4toix[material] = materialIndex ; 

    return material ;  
}


unsigned int CPropLib::getMaterialIndex(G4Material* material)
{
    return m_g4toix[material] ;
}


void CPropLib::dumpMaterials(const char* msg)
{
    typedef std::map<G4Material*, unsigned int> MMU ; 
    LOG(info) << msg << " g4toix" ; 

    for(MMU::const_iterator it=m_g4toix.begin() ; it != m_g4toix.end() ; it++)
    {
        G4Material* mat = it->first ; 
        unsigned int idx = it->second ; 

        const G4String& name = mat->GetName();

        std::cout << std::setw(40) << name 
                  << std::setw(5) << idx 
                  << std::endl ; 
    }

    LOG(info) << msg  << " ggtog4" ; 
    typedef std::map<GMaterial*, G4Material*> MMM ; 
    for(MMM::const_iterator it=m_ggtog4.begin() ; it != m_ggtog4.end() ; it++)
    {
        GMaterial* ggmat = it->first ; 
        G4Material* g4mat = it->second ; 

        const G4String& name = g4mat->GetName();
        const char* ggname = ggmat->getShortName();


        std::cout << std::setw(40) << name 
                  << std::setw(40) << ggname 
                  << std::endl ; 
    }

}
