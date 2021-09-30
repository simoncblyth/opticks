/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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
#include "G4Version.hh"
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



const plog::Severity CPropLib::LEVEL = PLOG::EnvLevel("CPropLib", "DEBUG") ; 

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
    LOG(LEVEL) << "[" ; 

    LOG(LEVEL) << m_slib->desc(); 
    //m_slib->dump(); 

    m_sensor_surface = m_slib->getSensorSurface(0) ;

    if(m_sensor_surface == NULL)
    {
        LOG(LEVEL) << " surface lib sensor_surface NULL " ;
        //assert(0);   // this happens with test running such as tboolean-box 
    }
    else
    {
        if(m_verbosity>2)
        m_sensor_surface->Summary("CPropLib::init cathode_surface");
    }

    m_dscale = double(GConstant::h_Planck*GConstant::c_light/GConstant::nanometer) ;

    initCheckConstants(); 

    //initSetupOverrides();
 
    //convert();
    LOG(LEVEL) << "]" ; 
}


void CPropLib::initSetupOverrides()
{
    double yield = 10.f ; 

    std::map<std::string, double>  gdls ; 
    gdls["SCINTILLATIONYIELD"] = yield ;  

    std::map<std::string, double>  ls ; 
    ls["SCINTILLATIONYIELD"] = yield ;  

    m_const_override["GdDopedLS"] = gdls ; 
    m_const_override["LiquidScintillator"] = ls ; 
}


void CPropLib::initCheckConstants()
{
    LOG(debug) 
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

G4LogicalBorderSurface* CPropLib::makeConstantSurface(const char* name, G4VPhysicalVolume* pv1, G4VPhysicalVolume* pv2, double effi, double refl)
{
    G4OpticalSurface* os = makeOpticalSurface(name);

    GProperty<double>* efficiency = m_mlib->makeConstantProperty(effi);
    GProperty<double>* reflectivity = m_mlib->makeConstantProperty(refl);
 
    G4MaterialPropertiesTable* mpt = os->GetMaterialPropertiesTable() ;
    addProperty(mpt, name, "EFFICIENCY" , efficiency );
    addProperty(mpt, name, "REFLECTIVITY" , reflectivity );

    G4LogicalBorderSurface* lbs = new G4LogicalBorderSurface(name,pv1,pv2,os);
    return lbs ; 
}

/**
CPropLib::makeCathodeSurface
------------------------------

dielectric_metal is required so DsG4OpBoundaryProcess can doAbsorption 
where EFFICIENCY is consulted and detect can happen

**/

G4LogicalBorderSurface* CPropLib::makeCathodeSurface(const char* name, G4VPhysicalVolume* pv1, G4VPhysicalVolume* pv2)
{
    G4OpticalSurface* os = makeOpticalSurface(name);
    os->SetType(dielectric_metal);
 
    GProperty<double>* detect = m_sensor_surface->getProperty("detect"); assert(detect);
    GProperty<double>* reflectivity = m_mlib->makeConstantProperty(0.f);

    G4MaterialPropertiesTable* mpt = os->GetMaterialPropertiesTable() ;
    addProperty(mpt, name, "EFFICIENCY" , detect );
    addProperty(mpt, name, "REFLECTIVITY" , reflectivity );

    G4LogicalBorderSurface* lbs = new G4LogicalBorderSurface(name,pv1,pv2,os);
    return lbs ; 
}


/**
CPropLib::makeMaterialPropertiesTable
---------------------------------------

Converts Opticks GGeo GMaterial as originally obtained from G4DAE 
into Geant4 G4MaterialPropertiesTable.

This gets invoked from CGDMLDetector::addMPTLegacyGDML 
bringing together the geometry from ancient GDML 
with the material properties from ancient G4DAE.

GMaterial which are identified as scintillators by having non-zero "reemission_probability" 
have extra properties added via CPropLib::addScintillatorMaterialProperties

**/

G4MaterialPropertiesTable* CPropLib::makeMaterialPropertiesTable(const GMaterial* ggmat)
{
    const char* name = ggmat->getShortName();
    GMaterial* _ggmat = const_cast<GMaterial*>(ggmat) ; // not changed 


    //bool is_sensor_material = strcmp(name, SENSOR_MATERIAL) == 0 ; // HUH: still checking for Bialkaki : shows this code very old
    bool is_scintillator = _ggmat->hasNonZeroProperty("reemission_prob") ;

    LOG(LEVEL) 
         << " name " << name
     //    << " " << ( is_sensor_material ? "is_sensor_material" : "" ) 
         << " " << ( is_scintillator ? "is_scintillator" : "" ) 
         ; 
        
    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    addProperties(mpt, _ggmat, "RINDEX,ABSLENGTH,RAYLEIGH,REEMISSIONPROB,GROUPVEL");

/*
    bool legacy = Opticks::IsLegacyGeometryEnabled() ; 
    if(is_sensor_material && legacy)
    {
        addSensorMaterialProperties(mpt, name ) ; 
    }
*/

    if(is_scintillator)
    {
        addScintillatorMaterialProperties(mpt, name ) ; 
    }
    return mpt ;
}


/**
CPropLib::addSensorMaterialProperties
----------------------------------------

For sensor materials, currently those with name "Bialkali" 
it is required that a surface surface has been collected 
into the surface lib.  That sensor surface must have a
"detect" property that is taken from the surface and
added as an EFFICIENCY property into the G4MaterialPropertiesTable.

Hmm suspect (eg due to hardcoded Bialkali name) this may be a remnant 
of fixup for the peculiar mixture of GDML and G4DAE geometry information 
that was used in the legacy workflow in the era before Geant4 
persisted material properties in GDML. 

It might have been needed only for that first generation ?

Relevant discussion in: 

* :doc:`notes/issues/direct_route_needs_AssimpGGeo_convertSensors_equivalent`

  
**/

void CPropLib::addSensorMaterialProperties( G4MaterialPropertiesTable* mpt, const char* name )
{
    GPropertyMap<double>* surf = m_sensor_surface ; 

    if(!surf)
    {
        LOG(fatal) 
            << " material with SENSOR_MATERIAL name " << name 
            << " but no sensor_surface "
            ; 
        LOG(fatal) 
            << "m_sensor_surface is obtained from slib at CPropLib::init " 
            << " when Bialkali material is in the mlib " 
            << " it is required for a sensor surface (with EFFICIENCY/detect) property "
            << " to be in the slib " 
            ;
    }
    assert(surf);

    LOG(error) 
         << " name " << name 
         << " adding EFFICIENCY : START " << surf->brief() ;  

    //surf->dump("transfer detect to EFFICIENCY");

    GProperty<double>* detect = surf->getProperty("detect"); 
    assert( detect ); 

    //addProperties(mpt, surf, "EFFICIENCY");
    addProperty( mpt, name, "EFFICIENCY", detect );   // name arg is for debug only 

    //CMPT::Dump(mpt);

    assert( CMPT::HasProperty(mpt, "EFFICIENCY")) ; 
    // REFLECTIVITY ?
}



/**
CPropLib::addScintillatorMaterialProperties
--------------------------------------------

**/

void CPropLib::addScintillatorMaterialProperties( G4MaterialPropertiesTable* mpt, const char* name )
{
    GPropertyMap<double>* scintillator = m_sclib->getRaw(name);

    if( scintillator == nullptr )
    {
        LOG(fatal) << " FAILED to find material in m_sclib (GScintillatorLib) with name " << name ; 
        m_sclib->dump("CPropLib::addScintillatorMaterialProperties");  
    }
    assert(scintillator && "non-zero reemission prob materials should has an associated raw scintillator");
    LOG(LEVEL) 
        << " found corresponding scintillator from sclib " 
        << " name " << name 
        << " keys " << scintillator->getKeysString() 
        ; 

    bool keylocal = false ; 
    bool constant = false ; 
    addProperties(mpt, scintillator, "SLOWCOMPONENT,FASTCOMPONENT", keylocal, constant);
    addProperties(mpt, scintillator, "SCINTILLATIONYIELD,RESOLUTIONSCALE,YIELDRATIO,FASTTIMECONSTANT,SLOWTIMECONSTANT", keylocal, constant ); // this used constant=true formerly

    // NB the above skips prefixed versions of the constants: Alpha, 
    //addProperties(mpt, scintillator, "ALL",          keylocal=false, constant=true );
}



/**
CPropLib::addProperties
-------------------------

Add properties from GPropertyMap.pmap to the G4MaterialPropertiesTable.mpt 
identified by the _keys comma delimited list of local keys (eg EFFICIENCY)

**/

void CPropLib::addProperties(G4MaterialPropertiesTable* mpt, GPropertyMap<double>* pmap, const char* _keys, bool keylocal, bool constant)
{
   
    std::vector<std::string> keys ; 
    boost::split(keys, _keys, boost::is_any_of(","));   
    bool all = keys.size() == 1 && keys[0].compare("ALL") == 0 ;


    const char* matname = pmap->getShortName();
    unsigned int nprop = pmap->getNumProperties();

    //if(m_verbosity > 1)
    pLOG(m_level,0) 
        << " keys " << _keys
        << " matname " << matname 
        << " nprop " << nprop
        ;

    //pmap->dump("CPropLib::addProperties"); 

    std::stringstream ss ; 

    for(unsigned int i=0 ; i<nprop ; i++)
    {
        const char* key =  pmap->getPropertyNameByIndex(i); // refractive_index absorption_length scattering_length reemission_prob
        const char* lkey = m_mlib->getLocalKey(key) ;       // ->lkey RINDEX ABSLENGTH RAYLEIGH REEMISSIONPROB EFFICIENCY  or NULL if not found
        const char* ukey = keylocal ? lkey : key ;

        if(!ukey) 
           LOG(fatal) 
               << "missing key for prop"
               << " i " << i
               << " nprop " << nprop
               << " matname " << matname
               << " key " << key 
               << " lkey " << lkey 
               << " ukey " << ukey 
               << " keylocal " << keylocal
               ; 
        assert(ukey);

        pLOG(m_level,+1)
            << matname << " " << i  << " key " << key << " lkey " << lkey << " ukey " << ukey  ;

        bool select = all ? true : std::find(keys.begin(), keys.end(), ukey) != keys.end() ;
        if(select)
        {
            GProperty<double>* prop = pmap->getPropertyByIndex(i);
            if(constant)
                addConstProperty(mpt, matname, ukey , prop );
            else
                addProperty(mpt, matname, ukey , prop );
 
            ss << ukey << " " ; 
        }
        else
        {
            pLOG(m_level,0) 
                << std::setw(30) << matname << " skipped " << ukey ;
        }
    }
    std::string lka = ss.str(); 
    pLOG(m_level,-1) 
         << "MPT of " << std::setw(30) << matname << " keys: " << lka ; ; 
}


void CPropLib::addConstProperty(G4MaterialPropertiesTable* mpt, const char* matname, const char* lkey,  GProperty<double>* prop )
{
    if(!prop->isConstant())
    { 
        LOG(error) << matname << "." << lkey << " SKIP NON-CONSTANT PROP " ; 
        return  ;
    }

    double value = prop->getConstant();

    bool has_const_override = m_const_override.count(matname)==1 && m_const_override[matname].count(lkey) == 1 ; 

    double uvalue =  has_const_override ? m_const_override[matname][lkey] : value ;  
          
    if( value != uvalue )
    {
        LOG(error)
                     << " OVERRIDE "  
                     << matname << "." << lkey 
                     << " from " << value
                     << " to " << uvalue 
                     ;
    }

    mpt->AddConstProperty(lkey, uvalue); 
}

void CPropLib::addProperty(G4MaterialPropertiesTable* mpt, const char* matname, const char* lkey,  GProperty<double>* prop )
{
    // matname only used for debug dumping 

    bool abslength = strcmp(lkey, "ABSLENGTH") == 0 ;
    bool rayleigh = strcmp(lkey, "RAYLEIGH") == 0 ;
    bool length = abslength || rayleigh ;

    unsigned int nval  = prop->getLength();

    if(m_verbosity>2)
    prop->Summary(lkey);   

    LOG(LEVEL) 
        << " nval " << std::setw(3) << nval  
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
        double fnm = prop->getDomain(j) ;
        double fval = prop->getValue(j) ; 

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
#if G4VERSION_NUMBER < 1100
    G4MaterialPropertyVector* mpv = mpt->AddProperty(lkey, ddom, dval, nval);
#else
    G4String skey(lkey); 
    G4int keyIdx = mpt->GetPropertyIndex(skey, false); 
    G4bool createNewKey = keyIdx == -1  ; 
    G4MaterialPropertyVector* mpv = mpt->AddProperty(lkey, ddom, dval, nval, createNewKey );
#endif
    assert( mpv ); 
  
    // NB spline interpolation is expected to not be used
    // formerly explicitly Set_Spline(false) but Geant4 API changes 
    // make that problematic : so currently assuming
    // a sensible default of false for spline interpolation
    //
    // see notes/issue/optical_local_time_goes_backward.rst
    // see notes/issue/geant4_ok_integration/interpol_mismatch.rst

    delete [] ddom ; 
    delete [] dval ; 
}










std::string CPropLib::getMaterialKeys(const G4Material* mat)
{   
    G4bool warning ; 
    typedef G4MaterialPropertyVector MPV ; 

    std::stringstream ss ;
    G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable();

    std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ;
    LOG(debug) << " pns " << pns.size() ; 
    for( unsigned i=0 ; i < pns.size() ; i++)
    {   
        const std::string& pname = pns[i]; 
        G4int pidx = mpt->GetPropertyIndex(pname, warning=true); 
        assert( pidx > -1 );  
        MPV* pvec = const_cast<G4MaterialPropertiesTable*>(mpt)->GetProperty(pidx, warning=false );  
        if(pvec == NULL) continue ; 

        ss << pname << " " ; 
    }   
    return ss.str(); 
}







GPropertyMap<double>* CPropLib::convertTable(G4MaterialPropertiesTable* mpt, const char* name)
{    
    GPropertyMap<double>* pmap = new GPropertyMap<double>(name);

    typedef G4MaterialPropertyVector MPV ; 
    bool warning ; 

    std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ;
    for( unsigned i=0 ; i < pns.size() ; i++)
    {   
        const std::string& pname = pns[i]; 
        G4int pidx = mpt->GetPropertyIndex(pname, warning=true); 
        assert( pidx > -1 );  
        MPV* pvec = const_cast<G4MaterialPropertiesTable*>(mpt)->GetProperty(pidx, warning=false );  
        if(pvec == NULL) continue ; 

        GProperty<double>* prop = convertVector(pvec);        
        pmap->addPropertyAsis( pname.c_str(), prop );  
    }   


    std::vector<G4String> cpns = mpt->GetMaterialConstPropertyNames() ;
    LOG(debug) << " cpns " << cpns.size() ; 

    for( unsigned i=0 ; i < cpns.size() ; i++)
    {   
        const std::string& n = cpns[i]; 
        G4bool exists = mpt->ConstPropertyExists( n.c_str() ) ; 
        if(!exists) continue ; 

        G4int pidx = mpt->GetConstPropertyIndex(n, warning=true); 
        assert( pidx > -1 );  
        G4double pvalue = mpt->GetConstProperty(pidx);  

        // express standard Opticks nm range in MeV, and swap order
        double dlow  = m_dscale/m_domain->getHigh() ; 
        double dhigh = m_dscale/m_domain->getLow() ;  
    
        LOG(info) << "CPropLib::convertTable" 
                  << " domlow (nm) "  << m_domain->getLow()  
                  << " domhigh (nm) " << m_domain->getHigh()
                  << " dscale MeV/nm " << m_dscale 
                  << " dlow  (MeV)  " << dlow 
                  << " dhigh (MeV) " << dhigh
                  ;


        GProperty<double>* prop = GProperty<double>::from_constant(pvalue, dlow, dhigh );        
        pmap->addPropertyAsis( n.c_str(), prop );  
   }
   return pmap ;    
}



GProperty<double>* CPropLib::convertVector(G4PhysicsVector* pvec)
{
    unsigned int length = pvec->GetVectorLength() ;
    double* domain = new double[length] ;
    double* values = new double[length] ;
    for(unsigned int i=0 ; i < length ; i++)
    {
         domain[i] = double(pvec->Energy(i)) ;
         values[i] = double((*pvec)[i]) ;
    }
    GProperty<double>* prop = new GProperty<double>(values, domain, length );    

    LOG(debug) << "CPropLib::convertVector" 
              << " raw domain  (MeV) " << domain[0] << " : " << domain[length-1] 
              << " m_dscale*(1/domain) (nm) " << m_dscale*1./domain[0] << " : " << m_dscale*1./domain[length-1] 
              ;  

    delete [] domain ;  
    delete [] values ; 

    return prop ; 
}


