// op --cproplib

#include "CFG4_BODY.hh"
#include <algorithm>
#include <boost/algorithm/string.hpp>

// okc-
#include "Opticks.hh"

// ggeo-
#include "GDomain.hh"
#include "GAry.hh"
#include "GProperty.hh"
#include "GConstant.hh"
#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GScintillatorLib.hh"

#include "GMaterial.hh"
#include "GPmt.hh"
#include "GCSG.hh"

// g4-
#include "G4MaterialTable.hh"
#include "G4Material.hh"
#include "globals.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include "G4OpticalSurface.hh"
#include "G4LogicalBorderSurface.hh"

// cg4-
#include "CPropLib.hh"

// npy-
#include "PLOG.hh"


const char* CPropLib::SENSOR_MATERIAL = "Bialkali" ;


CPropLib::CPropLib(Opticks* opticks, int verbosity)
  : 
  m_opticks(opticks),
  m_verbosity(verbosity),
  m_bndlib(NULL),
  m_mlib(NULL),
  m_slib(NULL),
  m_sclib(NULL),
  m_domain(NULL),
  m_dscale(1), 
  m_groupvel_kludge(true)
{
    init();
}


void CPropLib::setGroupvelKludge(bool gvk)
{
   m_groupvel_kludge = gvk ; 
}


void CPropLib::init()
{
    bool constituents ; 

    m_bndlib = GBndLib::load(m_opticks, constituents=true);
    m_mlib = m_bndlib->getMaterialLib();
    m_slib = m_bndlib->getSurfaceLib();

    m_sclib = GScintillatorLib::load(m_opticks);
    m_domain = m_mlib->getDefaultDomain();

    m_sensor_surface = m_slib->getSensorSurface(0) ;

    if(m_verbosity>2)
    m_sensor_surface->Summary("CPropLib::init cathode_surface");

    m_dscale = float(GConstant::h_Planck*GConstant::c_light/GConstant::nanometer) ;

    checkConstants(); 

    setupOverrides();
 
    //convert();
}


void CPropLib::setupOverrides()
{
    float yield = 10.f ; 

    std::map<std::string, float>  gdls ; 
    gdls["SCINTILLATIONYIELD"] = yield ;  

    std::map<std::string, float>  ls ; 
    ls["SCINTILLATIONYIELD"] = yield ;  

    m_const_override["GdDopedLS"] = gdls ; 
    m_const_override["LiquidScintillator"] = ls ; 
}


void CPropLib::checkConstants()
{

    LOG(info) << "CPropLib::checkConstants" 
               << " mm " << mm 
               << " MeV " << MeV
               << " nanosecond " << nanosecond
               << " ns " << ns
               << " nm " << nm
               << " GC::nanometer " << GConstant::nanometer
               << " h_Planck " << h_Planck
               << " GC::h_Planck " << GConstant::h_Planck
               << " c_light " << c_light
               << " GC::c_light " << GConstant::c_light
               << " dscale " << m_dscale 
               ;   

}



unsigned int CPropLib::getNumMaterials()
{
   return m_mlib->getNumMaterials();
}
const GMaterial* CPropLib::getMaterial(unsigned int index)
{
   return m_mlib->getMaterial(index); 
}

bool CPropLib::hasMaterial(const char* shortname)
{
   return m_mlib->hasMaterial(shortname); 
}
const GMaterial* CPropLib::getMaterial(const char* shortname)
{
   return m_mlib->getMaterial(shortname); 
}

void CPropLib::convert()
{
    unsigned int ngg = getNumMaterials() ;
    for(unsigned int i=0 ; i < ngg ; i++)
    {
        const GMaterial* ggmat = getMaterial(i);
        const char* name = ggmat->getShortName() ;
        const G4Material* g4mat = convertMaterial(ggmat);
        std::string keys = getMaterialKeys(g4mat);
        m_g4mat[name] = g4mat ; 
        LOG(debug) << "CPropLib::convert : converted ggeo material to G4 material " << name << " with keys " << keys ;  
    }
    LOG(info) << "CPropLib::convert : converted " << ngg << " ggeo materials to G4 materials " ; 
}


const G4Material* CPropLib::getG4Material(const char* shortname)
{
    return m_g4mat.count(shortname) == 1 ? m_g4mat[shortname] : NULL ; 
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
    GPmt* pmt = GPmt::load( m_opticks, m_bndlib, 0, slice );    // pmtIndex:0
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
    addProperty(mpt, name, "EFFICIENCY" , efficiency );
    addProperty(mpt, name, "REFLECTIVITY" , reflectivity );

    G4LogicalBorderSurface* lbs = new G4LogicalBorderSurface(name,pv1,pv2,os);
    return lbs ; 
}

G4LogicalBorderSurface* CPropLib::makeCathodeSurface(const char* name, G4VPhysicalVolume* pv1, G4VPhysicalVolume* pv2)
{
    G4OpticalSurface* os = makeOpticalSurface(name);

    GProperty<float>* detect = m_sensor_surface->getProperty("detect"); assert(detect);
    GProperty<float>* reflectivity = m_mlib->makeConstantProperty(0.f);

    G4MaterialPropertiesTable* mpt = os->GetMaterialPropertiesTable() ;
    addProperty(mpt, name, "EFFICIENCY" , detect );
    addProperty(mpt, name, "REFLECTIVITY" , reflectivity );

    G4LogicalBorderSurface* lbs = new G4LogicalBorderSurface(name,pv1,pv2,os);
    return lbs ; 
}


/*
 GROUPVEL kludge causing "generational" confusion
 as it assumed that no such property already existed

     if(strcmp(lkey,"RINDEX")==0)
     {
         if(m_groupvel_kludge)
         {
             LOG(info) << "CPropLib::makeMaterialPropertiesTable applying GROUPVEL kludge" ; 
             addProperty(mpt, "GROUPVEL", prop );
         }
     }


*/



G4MaterialPropertiesTable* CPropLib::makeMaterialPropertiesTable(const GMaterial* ggmat)
{
    const char* name = ggmat->getShortName();
    GMaterial* _ggmat = const_cast<GMaterial*>(ggmat) ; // wont change it, i promise 

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    addProperties(mpt, _ggmat, "RINDEX,ABSLENGTH,RAYLEIGH,REEMISSIONPROB");

    if(strcmp(name, SENSOR_MATERIAL)==0)
    {
        GPropertyMap<float>* surf = m_sensor_surface ; 
        assert(surf);
        addProperties(mpt, surf, "EFFICIENCY");
    }

    if(_ggmat->hasNonZeroProperty("reemission_prob"))
    {
        GPropertyMap<float>* scintillator = m_sclib->getRaw(name);
        assert(scintillator && "non-zero reemission prob materials should has an associated raw scintillator");
        LOG(debug) << "CPropLib::makeMaterialPropertiesTable found corresponding scintillator from sclib " 
                  << " name " << name 
                  << " keys " << scintillator->getKeysString() 
                   ; 
        bool keylocal, constant ; 
        addProperties(mpt, scintillator, "SLOWCOMPONENT,FASTCOMPONENT", keylocal=false, constant=false);
        addProperties(mpt, scintillator, "SCINTILLATIONYIELD,RESOLUTIONSCALE,YIELDRATIO,FASTTIMECONSTANT,SLOWTIMECONSTANT", keylocal=false, constant=true );

        // NB the above skips prefixed versions of the constants: Alpha, 
        //addProperties(mpt, scintillator, "ALL",          keylocal=false, constant=true );
    }
    return mpt ;
}


void CPropLib::addProperties(G4MaterialPropertiesTable* mpt, GPropertyMap<float>* pmap, const char* _keys, bool keylocal, bool constant)
{
    std::vector<std::string> keys ; 
    boost::split(keys, _keys, boost::is_any_of(","));   

    bool all = keys.size() == 1 && keys[0].compare("ALL") == 0 ;

    const char* matname = pmap->getShortName();
    unsigned int nprop = pmap->getNumProperties();
    std::stringstream ss ; 

    for(unsigned int i=0 ; i<nprop ; i++)
    {
        const char* key =  pmap->getPropertyNameByIndex(i); // refractive_index absorption_length scattering_length reemission_prob
        const char* lkey = m_mlib->getLocalKey(key) ;      // RINDEX ABSLENGTH RAYLEIGH REEMISSIONPROB
        const char* ukey = keylocal ? lkey : key ;

        if(!ukey) LOG(fatal) << "CPropLib::addProperties missing key for prop " << i ; 
        assert(ukey);

        LOG(debug) << "CPropLib::addProperties " << matname << " " << i  << " key " << key << " lkey " << lkey << " ukey " << ukey  ;

        bool select = all ? true : std::find(keys.begin(), keys.end(), ukey) != keys.end() ;
        if(select)
        {
            GProperty<float>* prop = pmap->getPropertyByIndex(i);
            if(constant)
                addConstProperty(mpt, matname, ukey , prop );
            else
                addProperty(mpt, matname, ukey , prop );
 
            ss << ukey << " " ; 
        }
        else
        {
            LOG(debug) << "CPropLib::addProperties " << std::setw(30) << matname << "skipped " << ukey ;
        }
    }
    std::string lka = ss.str(); 
    LOG(debug) << "CPropLib::addProperties MPT of " << std::setw(30) << matname << " keys: " << lka ; ; 
}


void CPropLib::addConstProperty(G4MaterialPropertiesTable* mpt, const char* matname, const char* lkey,  GProperty<float>* prop )
{
    if(!prop->isConstant())
    { 
        LOG(warning) << "CPropLib::addConstProperty " << matname << "." << lkey << " SKIP NON-CONSTANT PROP " ; 
        return  ;
    }

    float value = prop->getConstant();
    float uvalue =  m_const_override.count(matname)==1 && m_const_override[matname].count(lkey) == 1 ? m_const_override[matname][lkey] : value ;  
          
    if( value != uvalue )
    {
        LOG(warning) << "CPropLib::addConstProperty"
                     << " OVERRIDE "  
                     << matname << "." << lkey 
                     << " from " << value
                     << " to " << uvalue 
                     ;
    }

    mpt->AddConstProperty(lkey, uvalue); 
}

void CPropLib::addProperty(G4MaterialPropertiesTable* mpt, const char* matname, const char* lkey,  GProperty<float>* prop )
{

    bool abslength = strcmp(lkey, "ABSLENGTH") == 0 ;
    bool rayleigh = strcmp(lkey, "RAYLEIGH") == 0 ;
    bool length = abslength || rayleigh ;
    bool groupvel = strcmp(lkey, "GROUPVEL") == 0 ; 

    unsigned int nval  = prop->getLength();

    if(m_verbosity>2)
    prop->Summary(lkey);   

    if(m_verbosity>1)
    LOG(info) << "CPropLib::addProperty" 
               << " lkey " << std::setw(40) << lkey
               << " nval " << std::setw(10) << nval
               << " length " << std::setw(10) << length
               << " mm " << std::setw(10) << mm 
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

        if(groupvel && m_groupvel_kludge)
        {        
            // special cased addProperty with the RINDEX property
            value = c_light/value ;  
        }
        else if(length)
        {
            value *= mm ;    // mm=1 anyhow, 
        }

        ddom[nval-1-j] = G4double(energy) ; 
        dval[nval-1-j] = G4double(value) ;
    }


    //LOG(info) << "CPropLib::addProperty lkey " << lkey ; 

    G4MaterialPropertyVector* mpv = mpt->AddProperty(lkey, ddom, dval, nval);

    if(abslength)
    {
       // see issue/optical_local_time_goes_backward.rst
        mpv->SetSpline(false);
    }
    else
    {
        mpv->SetSpline(true);

    } 

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
    unsigned int materialIndex = m_mlib->getMaterialIndex(kmat);

    LOG(debug) << "CPropLib::convertMaterial  " 
              << " name " << name
              << " materialIndex " << materialIndex
              ;

    G4Material* material(NULL);
    if(strcmp(name,"MainH2OHale")==0)
    {
        material = makeWater(name) ;
    } 
    else
    {
        G4double z, a, density ;
        // presumably z, a and density are not relevant for optical photons 
        material = new G4Material(name, z=1., a=1.01*g/mole, density=universe_mean_density );
    }

    G4MaterialPropertiesTable* mpt = makeMaterialPropertiesTable(kmat);
    material->SetMaterialPropertiesTable(mpt);

    m_ggtog4[kmat] = material ; 
    m_g4toix[material] = materialIndex ; 
    m_ixtoname[materialIndex] = name ;  

    return material ;  
}


unsigned int CPropLib::getMaterialIndex(const G4Material* material)
{
    return m_g4toix[material] ;
}

const char* CPropLib::getMaterialName(unsigned int index)
{
    return m_ixtoname[index].c_str() ;
}



std::string CPropLib::MaterialSequence(unsigned long long seqmat)
{
    std::stringstream ss ;
    assert(sizeof(unsigned long long)*8 == 16*4);
    for(unsigned int i=0 ; i < 16 ; i++)
    {   
        unsigned long long m = (seqmat >> i*4) & 0xF ; 

        unsigned int idx = unsigned(m - 1);  

        ss << ( m > 0 ? getMaterialName(idx) : "-" ) << " " ;
        // using 1-based material indices, so 0 represents None
    }   
    return ss.str();
}


void CPropLib::dump(const char* msg)
{
    unsigned int ni = getNumMaterials() ;
    int index = m_opticks->getLastArgInt();
    const char* lastarg = m_opticks->getLastArg();

    if(index < int(ni))
    {   
        const GMaterial* mat = getMaterial(index);
        dump(mat, msg);
    }   
    else if(hasMaterial(lastarg))
    {   
        const GMaterial* mat = getMaterial(lastarg);
        dump(mat, msg);
    }   
    else
    {
        for(unsigned int i=0 ; i < ni ; i++)
        {
           const GMaterial* mat = getMaterial(i);
           dump(mat, msg);
        }
    }
}


void CPropLib::dump(const GMaterial* mat, const char* msg)
{
    GMaterial* _mat = const_cast<GMaterial*>(mat); 
    const G4Material* g4mat = getG4Material(_mat->getName());
    dumpMaterial(g4mat, msg);
}


void CPropLib::dumpMaterial(const G4Material* mat, const char* msg)
{
    const G4String& name = mat->GetName();
    LOG(info) << msg << " name " << name ; 

    G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable();
    //mpt->DumpTable();

    GPropertyMap<float>* pmap = convertTable( mpt , name );  // back into GGeo language for dumping purpose only

    unsigned int fw = 20 ;  
    bool dreciprocal = true ; 

    std::cout << pmap->make_table(fw, m_dscale, dreciprocal) << std::endl ;
}





void CPropLib::dumpMaterials(const char* msg)
{
    unsigned int ngg = getNumMaterials() ;
    LOG(info) << msg 
              << " numMaterials " << ngg
              ;

    for(unsigned int i=0 ; i < ngg ; i++)
    {
        const GMaterial* ggm = getMaterial(i);
        LOG(info) << "CPropLib::dumpMaterials" 
                  << " ggm (shortName) " << ggm->getShortName() 
                  ;
    }

    typedef std::map<const G4Material*, unsigned int> MMU ; 
    LOG(info) << " g4toix " << m_g4toix.size() << " " ; 

    for(MMU::const_iterator it=m_g4toix.begin() ; it != m_g4toix.end() ; it++)
    {
        const G4Material* mat = it->first ; 
        unsigned int idx = it->second ; 

        const G4String& name = mat->GetName();
        const char* name_2 = getMaterialName(idx) ; 

        std::cout << std::setw(40) << name 
                  << std::setw(5) << idx 
                  << std::setw(40) << name_2
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

        dumpMaterial(g4mat, "g4mat");
    }
}

std::string CPropLib::getMaterialKeys(const G4Material* mat)
{   
    std::stringstream ss ;
    G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable();
    typedef const std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> > MKP ; 
    MKP* kp = mpt->GetPropertiesMap() ;
    for(MKP::const_iterator it=kp->begin() ; it != kp->end() ; it++)
    {
        G4String k = it->first ; 
        ss << k << " " ; 
    } 
    return ss.str(); 
}


GPropertyMap<float>* CPropLib::convertTable(G4MaterialPropertiesTable* mpt, const char* name)
{    
    GPropertyMap<float>* pmap = new GPropertyMap<float>(name);
    
    typedef const std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> > MKP ; 
    MKP* pm = mpt->GetPropertiesMap() ;
    for(MKP::const_iterator it=pm->begin() ; it != pm->end() ; it++)
    {
        G4String k = it->first ; 
        G4MaterialPropertyVector* pvec = it->second ; 
        GProperty<float>* prop = convertVector(pvec);        
        pmap->addProperty( k.c_str(), prop );  
   }

   typedef const std::map< G4String, G4double, std::less<G4String> > MKC ; 
   MKC* cm = mpt->GetPropertiesCMap() ;
   for(MKC::const_iterator it=cm->begin() ; it != cm->end() ; it++)
   {
        G4String k = it->first ; 
        float v = float(it->second) ;

        // express standard Opticks nm range in MeV, and swap order
        float dlow  = m_dscale/m_domain->getHigh() ; 
        float dhigh = m_dscale/m_domain->getLow() ;  

        LOG(info) << "CPropLib::convertTable" 
                  << " domlow (nm) "  << m_domain->getLow()  
                  << " domhigh (nm) " << m_domain->getHigh()
                  << " dscale MeV/nm " << m_dscale 
                  << " dlow  (MeV)  " << dlow 
                  << " dhigh (MeV) " << dhigh
                  ;


        GProperty<float>* prop = GProperty<float>::from_constant(v, dlow, dhigh );        
        pmap->addProperty( k.c_str(), prop );  
   }
   return pmap ;    
}

GProperty<float>* CPropLib::convertVector(G4PhysicsVector* pvec)
{
    unsigned int length = pvec->GetVectorLength() ;
    float* domain = new float[length] ;
    float* values = new float[length] ;
    for(unsigned int i=0 ; i < length ; i++)
    {
         domain[i] = float(pvec->Energy(i)) ;
         values[i] = float((*pvec)[i]) ;
    }
    GProperty<float>* prop = new GProperty<float>(values, domain, length );    

    LOG(debug) << "CPropLib::convertVector" 
              << " raw domain  (MeV) " << domain[0] << " : " << domain[length-1] 
              << " m_dscale*(1/domain) (nm) " << m_dscale*1./domain[0] << " : " << m_dscale*1./domain[length-1] 
              ;  

    delete [] domain ;  
    delete [] values ; 

    return prop ; 
}


