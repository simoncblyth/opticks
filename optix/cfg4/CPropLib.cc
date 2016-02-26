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

#include "G4OpticalSurface.hh"
#include "G4LogicalBorderSurface.hh"


const char* CPropLib::SENSOR_MATERIAL = "Bialkali" ;

void CPropLib::init()
{
    bool constituents ; 
    m_bndlib = GBndLib::load(m_cache, constituents=true);
    m_mlib = m_bndlib->getMaterialLib();
    m_slib = m_bndlib->getSurfaceLib();

    m_sensor_surface = m_slib->getSensorSurface(0) ;

    if(m_verbosity>2)
    m_sensor_surface->Summary("CPropLib::init cathode_surface");

}

const G4Material* CPropLib::makeInnerMaterial(const char* spec)
{
    unsigned int boundary = m_bndlib->addBoundary(spec);
    unsigned int imat = m_bndlib->getInnerMaterial(boundary);
    GMaterial* kmat = m_mlib->getMaterial(imat);
    const G4Material* material = convertMaterial(kmat);
    return material ; 
}

const G4Material* CPropLib::makeMaterial(const char* matname)
{
    GMaterial* kmat = m_mlib->getMaterial(matname) ;
    const G4Material* material = convertMaterial(kmat);
    return material ; 
}

GCSG* CPropLib::getPmtCSG(NSlice* slice)
{
    GPmt* pmt = GPmt::load( m_cache, m_bndlib, 0, slice );    // pmtIndex:0
    GCSG* csg = pmt->getCSG();
    return csg ;
}


G4OpticalSurface* CPropLib::makeOpticalSurface(const char* name)
{
    G4OpticalSurface* os = new G4OpticalSurface(name);
    os->SetModel(glisur);
    os->SetType(dielectric_dielectric);
    os->SetFinish(polished);

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    os->SetMaterialPropertiesTable(mpt);

    return os ; 
}

G4LogicalBorderSurface* CPropLib::makeConstantSurface(const char* name, G4VPhysicalVolume* pv1, G4VPhysicalVolume* pv2, float effi, float refl)
{
    G4OpticalSurface* os = makeOpticalSurface(name);

    GProperty<float>* efficiency = m_mlib->makeConstantProperty(effi);
    GProperty<float>* reflectivity = m_mlib->makeConstantProperty(refl);
 
    G4MaterialPropertiesTable* mpt = os->GetMaterialPropertiesTable() ;
    addProperty(mpt, "EFFICIENCY" , efficiency );
    addProperty(mpt, "REFLECTIVITY" , reflectivity );

    G4LogicalBorderSurface* lbs = new G4LogicalBorderSurface(name,pv1,pv2,os);
    return lbs ; 
}

G4LogicalBorderSurface* CPropLib::makeCathodeSurface(const char* name, G4VPhysicalVolume* pv1, G4VPhysicalVolume* pv2)
{
    G4OpticalSurface* os = makeOpticalSurface(name);

    GProperty<float>* detect = m_sensor_surface->getProperty("detect"); assert(detect);
    GProperty<float>* reflectivity = m_mlib->makeConstantProperty(0.f);

    G4MaterialPropertiesTable* mpt = os->GetMaterialPropertiesTable() ;
    addProperty(mpt, "EFFICIENCY" , detect );
    addProperty(mpt, "REFLECTIVITY" , reflectivity );

    G4LogicalBorderSurface* lbs = new G4LogicalBorderSurface(name,pv1,pv2,os);
    return lbs ; 
}


G4MaterialPropertiesTable* CPropLib::makeMaterialPropertiesTable(const GMaterial* ggmat)
{
    const char* name = ggmat->getShortName();
    unsigned int mprop = ggmat->getNumProperties();

    LOG(info) << "CPropLib::makeMaterialPropertiesTable" 
              << " name " << name
              << " mprop " << mprop 
              ;   

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();

    GMaterial* ggm = const_cast<GMaterial*>(ggmat) ; // not easily to do this properly 

    for(unsigned int i=0 ; i<mprop ; i++)
    {
        const char* key = ggm->getPropertyNameByIndex(i); // refractive_index absorption_length scattering_length reemission_prob
        const char* lkey = m_mlib->getLocalKey(key) ;      // RINDEX ABSLENGTH RAYLEIGH REEMISSIONPROB
        GProperty<float>* prop = ggm->getPropertyByIndex(i);
        addProperty(mpt, lkey, prop );
    }



    // this was not enough, need optical surface to inject EFFICIENCY for optical photons
    if(strcmp(name, SENSOR_MATERIAL)==0)
    {
        GPropertyMap<float>* surf = m_sensor_surface ; 
        assert(surf);
        unsigned int sprop = surf->getNumProperties() ;

        LOG(info) << "CPropLib::makeMaterialPropertiesTable" 
                  << " material " << name
                  << " adding sensor surface properties "
                  << " from " << surf->getShortName()
                  << " sprop " << sprop 
                  ;   

        for(unsigned int j=0 ; j<sprop ; j++)
        {
            const char* key = surf->getPropertyNameByIndex(j); 
            const char* lkey = m_slib->getLocalKey(key) ; 
            if(strcmp(lkey,"EFFICIENCY")==0)
            {
                GProperty<float>* prop = surf->getPropertyByIndex(j);
                addProperty(mpt, lkey, prop );
            }
        }
    }


    return mpt ;
}





void CPropLib::addProperty(G4MaterialPropertiesTable* mpt, const char* lkey,  GProperty<float>* prop )
{
    bool length = strcmp(lkey, "ABSLENGTH") == 0 || strcmp(lkey, "RAYLEIGH") == 0  ;
    unsigned int nval  = prop->getLength();

    if(m_verbosity>1)
    prop->Summary(lkey);   

    LOG(info) << "CPropLib::addProperty" 
               << " lkey " << std::setw(40) << lkey
               << " nval " << std::setw(10) << nval
               << " length " << std::setw(10) << length
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



const G4Material* CPropLib::convertMaterial(const GMaterial* kmat)
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


unsigned int CPropLib::getMaterialIndex(const G4Material* material)
{
    return m_g4toix[material] ;
}


void CPropLib::dumpMaterials(const char* msg)
{
    typedef std::map<const G4Material*, unsigned int> MMU ; 
    LOG(info) << msg << " g4toix" ; 

    for(MMU::const_iterator it=m_g4toix.begin() ; it != m_g4toix.end() ; it++)
    {
        const G4Material* mat = it->first ; 
        unsigned int idx = it->second ; 

        const G4String& name = mat->GetName();

        std::cout << std::setw(40) << name 
                  << std::setw(5) << idx 
                  << std::endl ; 
    }

    LOG(info) << msg  << " ggtog4" ; 
    typedef std::map<const GMaterial*, const G4Material*> MMM ; 
    for(MMM::const_iterator it=m_ggtog4.begin() ; it != m_ggtog4.end() ; it++)
    {
        const GMaterial* ggmat = it->first ; 
        const G4Material* g4mat = it->second ; 

        const G4String& name = g4mat->GetName();
        const char* ggname = ggmat->getShortName();


        std::cout << std::setw(40) << name 
                  << std::setw(40) << ggname 
                  << std::endl ; 
    }

}
