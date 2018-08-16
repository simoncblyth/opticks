// op --cproplib

#include "CFG4_BODY.hh"
#include <algorithm>
#include <boost/algorithm/string.hpp>

#include "Opticks.hh"     // okc-
#include "OpticksHub.hh"  // okg-

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
#include "CMPT.hh"

// npy-
#include "PLOG.hh"


const char* CPropLib::SENSOR_MATERIAL = "Bialkali" ;


CPropLib::CPropLib(OpticksHub* hub, int verbosity)
  : 
  m_hub(hub),
  m_ok(m_hub->getOpticks()),
  m_verbosity(verbosity),
  m_bndlib(m_hub->getBndLib()),
  m_mlib(m_hub->getMaterialLib()),
  m_slib(m_hub->getSurfaceLib()),
  m_sclib(m_hub->getScintillatorLib()),
  m_domain(m_mlib->getDefaultDomain()),
  m_dscale(1),
  m_level(verbose)
{
    init();
}


GSurfaceLib* CPropLib::getSurfaceLib()
{
   return m_slib ; 
}


void CPropLib::init()
{
    pLOG(m_level,-2) << "CPropLib::init" ; 

    LOG(info) << m_slib->desc(); 
    //m_slib->dump(); 

    m_sensor_surface = m_slib->getSensorSurface(0) ;

    if(m_sensor_surface == NULL)
    {
        LOG(fatal) << " surface lib sensor_surface NULL " ;


        //assert(0);   // this happens with test running such as tboolean-box 
    }
    else
    {
        if(m_verbosity>2)
        m_sensor_surface->Summary("CPropLib::init cathode_surface");
    }

    m_dscale = float(GConstant::h_Planck*GConstant::c_light/GConstant::nanometer) ;

    initCheckConstants(); 

    initSetupOverrides();
 
    //convert();
}


void CPropLib::initSetupOverrides()
{
    float yield = 10.f ; 

    std::map<std::string, float>  gdls ; 
    gdls["SCINTILLATIONYIELD"] = yield ;  

    std::map<std::string, float>  ls ; 
    ls["SCINTILLATIONYIELD"] = yield ;  

    m_const_override["GdDopedLS"] = gdls ; 
    m_const_override["LiquidScintillator"] = ls ; 
}


void CPropLib::initCheckConstants()
{
    LOG(debug) << "CPropLib::initCheckConstants" 
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
   LOG(verbose) << "." ; 
   if(m_mlib == NULL)
   {
       LOG(error) << "mlib NULL" ;
       return 0 ;    
   } 

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
    os->SetType(dielectric_metal); 
    // dielectric_metal is required so DsG4OpBoundaryProcess can doAbsorption 
    // where EFFICIENCY is consulted and detect can happen

    GProperty<float>* detect = m_sensor_surface->getProperty("detect"); assert(detect);
    GProperty<float>* reflectivity = m_mlib->makeConstantProperty(0.f);

    G4MaterialPropertiesTable* mpt = os->GetMaterialPropertiesTable() ;
    addProperty(mpt, name, "EFFICIENCY" , detect );
    addProperty(mpt, name, "REFLECTIVITY" , reflectivity );

    G4LogicalBorderSurface* lbs = new G4LogicalBorderSurface(name,pv1,pv2,os);
    return lbs ; 
}



G4MaterialPropertiesTable* CPropLib::makeMaterialPropertiesTable(const GMaterial* ggmat)
{
    const char* name = ggmat->getShortName();
    GMaterial* _ggmat = const_cast<GMaterial*>(ggmat) ; // wont change it, i promise 

    LOG(verbose) << " name " << name ; 

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    addProperties(mpt, _ggmat, "RINDEX,ABSLENGTH,RAYLEIGH,REEMISSIONPROB,GROUPVEL");

    if(strcmp(name, SENSOR_MATERIAL)==0)
    {
        GPropertyMap<float>* surf = m_sensor_surface ; 

        if(!surf)
        {
            LOG(fatal) << "CPropLib::makeMaterialPropertiesTable"  
                       << " material with SENSOR_MATERIAL name " << name 
                       << " but no sensor_surface "
                       ; 
            LOG(fatal) << "m_sensor_surface is obtained from slib at CPropLib::init " 
                       << " when Bialkai material is in the mlib " 
                       << " it is required for a sensor surface (with EFFICIENCY/detect) property "
                       << " to be in the slib " 
                       ;
        }
        assert(surf);

        LOG(error) 
             << " name " << name 
             << " adding EFFICIENCY " << surf->brief() ;  

        addProperties(mpt, surf, "EFFICIENCY");

        //CMPT::Dump(mpt);

        assert( CMPT::HasProperty(mpt, "EFFICIENCY")) ; 

        // REFLECTIVITY ?
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

    //if(m_verbosity > 1)
    pLOG(m_level,0) << "CPropLib::addProperties"
                   << " keys " << _keys
                   << " matname " << matname 
                   << " nprop " << nprop
                    ;

    //pmap->dump("CPropLib::addProperties"); 


    std::stringstream ss ; 

    for(unsigned int i=0 ; i<nprop ; i++)
    {
        const char* key =  pmap->getPropertyNameByIndex(i); // refractive_index absorption_length scattering_length reemission_prob
        const char* lkey = m_mlib->getLocalKey(key) ;      // RINDEX ABSLENGTH RAYLEIGH REEMISSIONPROB
        const char* ukey = keylocal ? lkey : key ;

        if(!ukey) LOG(fatal) << "CPropLib::addProperties missing key for prop " << i ; 
        assert(ukey);

        pLOG(m_level,+1) << "CPropLib::addProperties " << matname << " " << i  << " key " << key << " lkey " << lkey << " ukey " << ukey  ;

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
            pLOG(m_level,0) << "CPropLib::addProperties " << std::setw(30) << matname << " skipped " << ukey ;
        }
    }
    std::string lka = ss.str(); 
    pLOG(m_level,-1) << "CPropLib::addProperties MPT of " << std::setw(30) << matname << " keys: " << lka ; ; 
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
    //bool groupvel = strcmp(lkey, "GROUPVEL") == 0 ; 

    unsigned int nval  = prop->getLength();

    if(m_verbosity>2)
    prop->Summary(lkey);   

    if(m_verbosity>1)
    LOG(info) << "CPropLib::addProperty" 
               << " lkey " << std::setw(40) << lkey
               << " nval " << std::setw(10) << nval
               << " length " << std::setw(10) << length
               << " mm " << std::setw(10) << mm 
               << " matname " << matname
               ;   


    // TODO: adopt CMPT::addProperty by moving the special casing elsewhere 

    G4double* ddom = new G4double[nval] ;
    G4double* dval = new G4double[nval] ;

    for(unsigned int j=0 ; j < nval ; j++)
    {
        float fnm = prop->getDomain(j) ;
        float fval = prop->getValue(j) ; 

        G4double wavelength = G4double(fnm)*nm ; 
        G4double energy = h_Planck*c_light/wavelength ;

        G4double value = G4double(fval) ;

        if(length)
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
        //mpv->SetSpline(true);
       // see issue/geant4_ok_integration/interpol_mismatch.rst
        mpv->SetSpline(false);

    } 

    delete [] ddom ; 
    delete [] dval ; 
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
   MKC* cmap = mpt->GetPropertiesCMap() ;
   for(MKC::const_iterator it=cmap->begin() ; it != cmap->end() ; it++)
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


