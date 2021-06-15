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

#include <iomanip>

#include "SGDML.hh"
#include "SStr.hh"

#include "BStr.hh"
#include "BMeta.hh"

#include "NGLM.hpp"
#include "NPY.hpp"

#include "Opticks.hh"

#include "GVector.hh"
#include "GDomain.hh"
#include "GAry.hh"
#include "GProperty.hh"

#include "GOpticalSurface.hh"
#include "GSkinSurface.hh"
#include "GBorderSurface.hh"
#include "GItemList.hh"

#include "GSurfaceLib.hh"

#include "PLOG.hh"
// trace/debug/info/warning/error/fatal


const plog::Severity GSurfaceLib::LEVEL = PLOG::EnvLevel("GSurfaceLib", "DEBUG") ; 


// surface
const char* GSurfaceLib::detect            = "detect" ;
const char* GSurfaceLib::absorb            = "absorb" ;
const char* GSurfaceLib::reflect_specular  = "reflect_specular" ;
const char* GSurfaceLib::reflect_diffuse   = "reflect_diffuse" ;

const char* GSurfaceLib::extra_x          = "extra_x" ;
const char* GSurfaceLib::extra_y          = "extra_y" ;
const char* GSurfaceLib::extra_z          = "extra_z" ;
const char* GSurfaceLib::extra_w          = "extra_w" ;

const char* GSurfaceLib::BPV1             = "bpv1" ;
const char* GSurfaceLib::BPV2             = "bpv2" ;
const char* GSurfaceLib::SSLV             = "sslv" ;

const char* GSurfaceLib::SKINSURFACE      = "skinsurface" ;
const char* GSurfaceLib::BORDERSURFACE    = "bordersurface" ;
const char* GSurfaceLib::TESTSURFACE      = "testsurface" ;


const char* GSurfaceLib::REFLECTIVITY = "REFLECTIVITY" ;
const char* GSurfaceLib::EFFICIENCY   = "EFFICIENCY" ;
const char* GSurfaceLib::SENSOR_SURFACE = "SensorSurface" ;


bool GSurfaceLib::NameEndsWithSensorSurface(const char* name) // static
{
    return BStr::EndsWith(name, SENSOR_SURFACE) ; 
}
const char* GSurfaceLib::NameWithoutSensorSurface(const char* name) // static
{
    return BStr::WithoutEnding(name, SENSOR_SURFACE) ; 
} 




const char* GSurfaceLib::keyspec = 
"detect:EFFICIENCY," 
"absorb:DUMMY," 
"reflect_specular:REFLECTIVITY," 
"reflect_diffuse:REFLECTIVITY," 
;

float GSurfaceLib::SURFACE_UNSET = -1.f ; 

void GSurfaceLib::save()
{
    LOG(LEVEL); 
    saveToCache();
    saveOpticalBuffer();
}
GSurfaceLib* GSurfaceLib::load(Opticks* ok)
{
    GSurfaceLib* lib = new GSurfaceLib(ok);
    lib->loadFromCache();
    lib->loadOpticalBuffer();
    return lib ; 
}

void GSurfaceLib::loadOpticalBuffer()
{
    std::string dir = getCacheDir(); 
    std::string name = getBufferName("Optical");

    LOG(LEVEL) 
         << " dir " << dir 
         << " name " << name 
         ;

    NPY<unsigned int>* ibuf = NPY<unsigned int>::load(dir.c_str(), name.c_str()); 

    importOpticalBuffer(ibuf);
    setOpticalBuffer(ibuf); 
}
void GSurfaceLib::saveOpticalBuffer()
{
    NPY<unsigned int>* ibuf = createOpticalBuffer();
    saveToCache(ibuf, "Optical") ; 
    setOpticalBuffer(ibuf);
}

void GSurfaceLib::setBasis(GSurfaceLib* basis)
{
    m_basis = basis ; 
}
GSurfaceLib* GSurfaceLib::getBasis() const 
{
    return m_basis ; 
}

unsigned GSurfaceLib::getNumSurfaces() const
{
    return m_surfaces.size();
}


std::string GSurfaceLib::desc() const 
{
    std::stringstream ss ; 

    ss << "GSurfaceLib"
       << " numSurfaces " << getNumSurfaces()
       << " this " << this 
       << " basis " << m_basis
       << " isClosed " << isClosed() 
       << " hasDomain " << hasDomain()
       ;

    return ss.str();
}


GSurfaceLib::GSurfaceLib(Opticks* ok, GSurfaceLib* basis) 
    :
    GPropertyLib(ok, "GSurfaceLib"),
    m_fake_efficiency(-1.f),
    m_optical_buffer(NULL),
    m_basis(basis),
    m_dbgsurf(ok->isDbgSurf())
{
    LOG(LEVEL) ; 
    init();
}

GSurfaceLib::GSurfaceLib(GSurfaceLib* src, GDomain<float>* domain, GSurfaceLib* basis) 
    :
    GPropertyLib(src, domain),
    m_fake_efficiency(-1.f),
    m_optical_buffer(NULL),
    m_basis(basis),
    m_dbgsurf(m_ok->isDbgSurf())
{
    LOG(LEVEL) ; 
    init();
    initInterpolatingCopy(src, domain);
}
 
void GSurfaceLib::initInterpolatingCopy(GSurfaceLib* src, GDomain<float>* domain)
{
    unsigned nsur = src->getNumSurfaces();

    for(unsigned i=0 ; i < nsur ; i++)
    {
        GPropertyMap<float>* ssur = src->getSurface(i);

        if(!ssur->hasStandardDomain())
        {
             LOG(verbose) << "GSurfaceLib::GSurfaceLib set ssur standard domain for " << i << " out of " << nsur ;
             ssur->setStandardDomain(src->getStandardDomain());
        }

        GPropertyMap<float>* dsur = new GPropertyMap<float>(ssur, domain );   // interpolating "copy" ctor

        addDirect(dsur);
    }
}
 

void GSurfaceLib::setFakeEfficiency(float fake_efficiency)
{
    m_fake_efficiency = fake_efficiency ; 
}
void GSurfaceLib::setOpticalBuffer(NPY<unsigned int>* ibuf)
{
    m_optical_buffer = ibuf ; 
}
NPY<unsigned int>* GSurfaceLib::getOpticalBuffer()
{
    return m_optical_buffer ;
}







const char* GSurfaceLib::propertyName(unsigned int k)
{
    assert(k < 4*NUM_FLOAT4);
    if(k == 0) return detect ;
    if(k == 1) return absorb ;
    if(k == 2) return reflect_specular;
    if(k == 3) return reflect_diffuse ;

    if(NUM_FLOAT4 > 1)
    {
        if(k == 4) return extra_x ;
        if(k == 5) return extra_y ;
        if(k == 6) return extra_z ;
        if(k == 7) return extra_w ;
    }

    return "?" ;
}

void GSurfaceLib::Summary(const char* msg)
{
    LOG(info) << msg  
              << " NumSurfaces " << getNumSurfaces() 
              << " NumFloat4 " << NUM_FLOAT4
              ;
}

void GSurfaceLib::defineDefaults(GPropertyMap<float>* defaults)
{
    defaults->addConstantProperty( detect          ,      SURFACE_UNSET );
    defaults->addConstantProperty( absorb          ,      SURFACE_UNSET );
    defaults->addConstantProperty( reflect_specular,      SURFACE_UNSET );
    defaults->addConstantProperty( reflect_diffuse ,      SURFACE_UNSET );

    if(NUM_FLOAT4 > 1)
    {
        defaults->addConstantProperty( extra_x,     SURFACE_UNSET  );
        defaults->addConstantProperty( extra_y,     SURFACE_UNSET  );
        defaults->addConstantProperty( extra_z,     SURFACE_UNSET  );
        defaults->addConstantProperty( extra_w,     SURFACE_UNSET  );
    }

}

void GSurfaceLib::init()
{
    setKeyMap(keyspec);
    defineDefaults(getDefaults());
}




/**
const char* GSurfaceLib::AssignSurfaceType(BMeta* surfmeta )
--------------------------------------------------------------

Return SKINSURFACE, BORDERSURFACE or TESTSURFACE depending on the
SSLV, BPV1, BPV2, name keys present in the surfmeta.


**/

const char* GSurfaceLib::AssignSurfaceType( BMeta* surfmeta ) // static 
{
     assert( surfmeta );
     const char* surftype = NULL ; 

     if( surfmeta->hasItem(SSLV)) 
     { 
         surftype = SKINSURFACE ; 
     }
     else if( surfmeta->hasItem(BPV1) && surfmeta->hasItem(BPV2) ) 
     {
         surftype = BORDERSURFACE ; 
     }
     else if( surfmeta->hasItem("name") )
     {
         // admit perfect surfaces prior to them being "located" via metadata keys
         std::string name = surfmeta->get<std::string>("name","no-name"); 
         if(BStr::StartsWith( name.c_str(), "perfect" )) surftype = TESTSURFACE ;
     }

     return surftype ; 
}



/**
GSurfaceLib::add(GBorderSurface* raw)
-------------------------------------

Adds the input border surface to m_border_surfaces and 
a standardized version of it to m_surfaces. 
The metadata keys "bpv1" and "bpv2" are used to retain border 
surface specific infomation within the common structure.

**/

void GSurfaceLib::add(GBorderSurface* raw, bool implicit, bool direct)
{
    bool has_EFFICIENCY = raw->hasProperty("EFFICIENCY"); 
    LOG(LEVEL) 
        << " GBorderSurface " 
        << " name " << raw->getName() 
        << " pv1 " << raw->getPV1()
        << " pv2 " << raw->getPV2()
        << " keys " << raw->getKeysString()   
        << " has_EFFICIENCY " << has_EFFICIENCY
        ; 

    m_border_surfaces.push_back(raw);

    GPropertyMap<float>* surf = dynamic_cast<GPropertyMap<float>* >(raw);
    if(implicit) surf->setImplicit(true); 

    addBorderSurface(surf, raw->getPV1(), raw->getPV2(), direct );
}

/**
GSurfaceLib::addBorderSurface
------------------------------

Invoked by `GSurfaceLib::add(GBorderSurface* raw)`
sets MetaKV keys "bpv1" "bpv2" onto the GPropertyMap 

**/

void GSurfaceLib::addBorderSurface(GPropertyMap<float>* surf, const char* pv1, const char* pv2, bool direct)
{
   // method to help with de-conflation of surface props and location
    std::string bpv1 = pv1 ;
    std::string bpv2 = pv2 ;
    surf->setMetaKV(BPV1, bpv1 );
    surf->setMetaKV(BPV2, bpv2 );
    surf->setBorderSurface();

    if(direct)
    {
        addDirect(surf);
    }
    else
    {
        addStandardized(surf); 
    }
}


/**
GSurfaceLib::add(GSkinSurface* raw)
-------------------------------------

Adds the input skin surface to m_skin_surfaces and 
a standardized version of it to m_surfaces. 
The metadata key "sslv" is used to retain skin surface specific 
information within the common structure.

**/

void GSurfaceLib::add(GSkinSurface* raw)
{
    bool has_EFFICIENCY = raw->hasProperty("EFFICIENCY"); 

    LOG(LEVEL) 
        << " GSkinSurface " 
        << " name " << raw->getName()
        << " keys " << raw->getKeysString()     
        << " has_EFFICIENCY " << has_EFFICIENCY
        ;

    m_skin_surfaces.push_back(raw);

    LOG(verbose) << ( raw ? raw->getName() : "NULL" ) ;

    GPropertyMap<float>* surf = dynamic_cast<GPropertyMap<float>* >(raw);
    bool direct = false  ;  // not standardized 
    addSkinSurface( surf, raw->getSkinSurfaceVol(), direct );
}
void GSurfaceLib::addSkinSurface(GPropertyMap<float>* surf, const char* sslv_, bool direct )
{
   // method to help with de-conflation of surface props and location
    std::string sslv = sslv_ ;
    surf->setMetaKV(SSLV, sslv );
    surf->setSkinSurface();

    if(direct)
    {
        addDirect(surf);
    }
    else
    {
        addStandardized(surf); 
    }
}



/**
GSurfaceLib::addStandardized(GPropertyMap<float>* surf)
------------------------------------------------------------

Common to both surface types:

1. standardize
2. collect into m_surfaces

**/

void GSurfaceLib::addStandardized(GPropertyMap<float>* surf)
{
    assert(!isClosed());
    GPropertyMap<float>* ssurf = createStandardSurface(surf) ;
    addDirect(ssurf);
}

void GSurfaceLib::addDirect(GPropertyMap<float>* surf)
{
    assert(!isClosed());
    m_surfaces.push_back(surf); 
}



/**
GSurfaceLib::createStandardSurface
------------------------------------

Invoked by GSurfaceLib::addStandardized arranges that all collected surfaces
have the standard set of properties.

1. detect
2. absorb 
3. reflect_specular
4. reflect_diffuse

**/

GPropertyMap<float>* GSurfaceLib::createStandardSurface(GPropertyMap<float>* src)
{
    GProperty<float>* _detect           = NULL ; 
    GProperty<float>* _absorb           = NULL ; 
    GProperty<float>* _reflect_specular = NULL ; 
    GProperty<float>* _reflect_diffuse  = NULL ; 

    bool is_sensor = src->isSensor() ; 

    if(!src)
    {
        _detect           = getDefaultProperty(detect); 
        _absorb           = getDefaultProperty(absorb); 
        _reflect_specular = getDefaultProperty(reflect_specular); 
        _reflect_diffuse  = getDefaultProperty(reflect_diffuse); 
    }
    else
    {
        assert( getStandardDomain() );
        assert( src->getStandardDomain() );
        assert( src->isSurface() );

        GOpticalSurface* os = src->getOpticalSurface() ;  // GSkinSurface and GBorderSurface ctor plant the OpticalSurface into the PropertyMap
        assert( os && " all surfaces must have associated OpticalSurface " );

        if(is_sensor)  // this means it has non-zero EFFICIENCY or detect property
        {
            GProperty<float>* _EFFICIENCY = src->getProperty(EFFICIENCY); 
            assert(_EFFICIENCY && os && "sensor surfaces must have an efficiency" );

            if(m_fake_efficiency >= 0.f && m_fake_efficiency <= 1.0f)
            {
                _detect           = makeConstantProperty(m_fake_efficiency) ;    
                _absorb           = makeConstantProperty(1.0-m_fake_efficiency);
                _reflect_specular = makeConstantProperty(0.0);
                _reflect_diffuse  = makeConstantProperty(0.0);
            } 
            else
            {
                _detect = _EFFICIENCY ;      
                _absorb = GProperty<float>::make_one_minus( _detect );
                _reflect_specular = makeConstantProperty(0.0);
                _reflect_diffuse  = makeConstantProperty(0.0);
            }
        }
        else
        {
            GProperty<float>* _REFLECTIVITY = src->getProperty(REFLECTIVITY); 
            if(_REFLECTIVITY == NULL)
            {
                LOG(fatal) << "ABORT : non-sensor surfaces must have a reflectivity" ; 
                LOG(fatal) << "This ABORT may be caused by Geant4 bug 2305 https://bugzilla-geant4.kek.jp/show_bug.cgi?id=2305 " ;
                LOG(fatal) << "which is present in Geant4 releases 1060,1061,1062,1063,1070 " ;   
                LOG(fatal) << "See the bash function g4-;g4-bug-2305-fix to change Geant4 or use a different Geant4 release" ; 
            }
            assert(_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity " );

            if(os->isSpecular())
            {
                _detect  = makeConstantProperty(0.0) ;    
                _reflect_specular = _REFLECTIVITY ;
                _reflect_diffuse  = makeConstantProperty(0.0) ;    
                _absorb  = GProperty<float>::make_one_minus(_reflect_specular);
            }
            else
            {
                _detect  = makeConstantProperty(0.0) ;    
                _reflect_specular = makeConstantProperty(0.0) ;    
                _reflect_diffuse  = _REFLECTIVITY ;
                _absorb  = GProperty<float>::make_one_minus(_reflect_diffuse);
            } 
        }
    }

    assert(_detect);
    assert(_absorb);
    assert(_reflect_specular);
    assert(_reflect_diffuse);

    GPropertyMap<float>* dst = new GPropertyMap<float>(src);

    dst->addProperty( detect          , _detect          );
    dst->addProperty( absorb          , _absorb          );
    dst->addProperty( reflect_specular, _reflect_specular);
    dst->addProperty( reflect_diffuse , _reflect_diffuse );


    bool valid = checkSurface(dst);
    assert(valid);

    return dst ; 
}













/**
GSurfaceLib::getBasisSurface
------------------------------

Basis surfaces are used with test geometry, methods to relocate 
surfaces requires changes to the volume names associated with the 
surfaces.

**/


GPropertyMap<float>* GSurfaceLib::getBasisSurface(const char* name) const 
{
    return m_basis ? m_basis->getSurface(name) : NULL ; 
}


void GSurfaceLib::relocateBasisBorderSurface(const char* name, const char* bpv1, const char* bpv2)
{
    GPropertyMap<float>* surf = getBasisSurface(name);
    if(!surf) LOG(fatal) << "relocateBasisBorderSurface requires basis library to be present and to contain the surface  " ; 
    assert( surf );        

    bool direct = true ;  // already standardized 
    addBorderSurface( surf, bpv1, bpv2, direct ); 
}

void GSurfaceLib::relocateBasisSkinSurface(const char* name, const char* sslv)
{
    GPropertyMap<float>* surf = getBasisSurface(name);
    if(!surf) LOG(fatal) << "relocateBasisSkinSurface requires basis library to be present and to contain the surface  " ; 
    assert( surf );           
  
    bool direct = true ;  // already standardized 
    addSkinSurface( surf, sslv, direct ); 
}





bool GSurfaceLib::operator()(const GPropertyMap<float>* a_, const GPropertyMap<float>* b_)
{
    const char* a = a_->getShortName();
    const char* b = b_->getShortName();

    typedef std::map<std::string, unsigned int> MSU ;
    MSU& order = getOrder();   
    MSU::const_iterator end = order.end() ; 
 
    unsigned int ia = order.find(a) == end ? UINT_MAX :  order[a] ; 
    unsigned int ib = order.find(b) == end ? UINT_MAX :  order[b] ; 
    return ia < ib ; 
}

/**
GSurfaceLib::sort
-------------------

Sorts m_surfaces


**/

void GSurfaceLib::sort()
{
    bool asis = true ; 
    if(asis)
    {
        LOG(LEVEL) << " not sorting ";
        return  ;
    } 

    typedef std::map<std::string, unsigned int> MSU ;
    MSU& order = getOrder();  

    if(order.size() == 0) return ; 
    std::stable_sort( m_surfaces.begin(), m_surfaces.end(), *this );

}


guint4 GSurfaceLib::createOpticalSurface(GPropertyMap<float>* src)
{
   assert(src->isSkinSurface() || src->isBorderSurface() || src->isTestSurface());
   GOpticalSurface* os = src->getOpticalSurface();
   assert(os && "all skin/boundary surface expected to have associated OpticalSurface");
   guint4 optical = os->getOptical();
   return optical ; 
}

guint4 GSurfaceLib::getOpticalSurface(unsigned int i)
{
    GPropertyMap<float>* surf = getSurface(i);    
    guint4 os = createOpticalSurface(surf);
    os.x = i ;
    return os ; 
}


GPropertyMap<float>* GSurfaceLib::makePerfect(const char* name, float detect_, float absorb_, float reflect_specular_, float reflect_diffuse_)
{
    // placeholders
    const char* type = "1" ; 
    const char* model = "1" ; 
    const char* finish = "1" ; 
    const char* value = "1" ; 
    GOpticalSurface* os = new GOpticalSurface(name, type, model, finish, value);

    unsigned int index = 1000 ;   // does this matter ? 
    GPropertyMap<float>* dst = new GPropertyMap<float>(name, index, TESTSURFACE, os);
    dst->setStandardDomain(getStandardDomain());

    addPerfectProperties(dst, detect_, absorb_, reflect_specular_, reflect_diffuse_ );  
    return dst ;  
}

void GSurfaceLib::addPerfectProperties( GPropertyMap<float>* dst, float detect_, float absorb_, float reflect_specular_, float reflect_diffuse_ )
{
    GProperty<float>* _detect           = makeConstantProperty(detect_) ;    
    GProperty<float>* _absorb           = makeConstantProperty(absorb_) ;    
    GProperty<float>* _reflect_specular = makeConstantProperty(reflect_specular_) ;    
    GProperty<float>* _reflect_diffuse  = makeConstantProperty(reflect_diffuse_) ;    

    dst->addProperty( detect          , _detect          );
    dst->addProperty( absorb          , _absorb          );
    dst->addProperty( reflect_specular, _reflect_specular);
    dst->addProperty( reflect_diffuse , _reflect_diffuse );
}


/**
GSurfaceLib::addImplicitBorderSurface_RINDEX_NoRINDEX
---------------------------------------------------------

Invoked from X4PhysicalVolume::convertImplicitSurfaces_r

This is adding explicit Opticks/GGeo border surfaces to emulate implicit Geant4 
SURFACE_ABSORB behaviour for photons going from material with RINDEX to material without RINDEX.

See notes/issues/GSurfaceLib__addImplicitBorderSurface_RINDEX_NoRINDEX.rst

**/

void GSurfaceLib::addImplicitBorderSurface_RINDEX_NoRINDEX( const char* pv1, const char* pv2 )
{
    GBorderSurface* prior_bs = findBorderSurface(pv1, pv2);
    if( prior_bs != nullptr )
    {
        LOG(fatal) 
            << " pv1 " << pv1 
            << " pv2 " << pv2 
            << " prior_bs " << prior_bs
            << " there is a prior GBorderSurface from pv1->pv2 "
            ;
        assert(0); 
    } 


    std::string spv1 = SGDML::Strip(pv1); 
    std::string spv2 = SGDML::Strip(pv2); 
    std::stringstream ss ; 
    ss << "Implicit_RINDEX_NoRINDEX_" << spv1 << "_" << spv2 ;  
    std::string s = ss.str(); 
    const char* name = s.c_str(); 

    // placeholders
    const char* type = "1" ; 
    const char* model = "1" ; 
    const char* finish = "1" ; 
    const char* value = "1" ; 
    GOpticalSurface* os = new GOpticalSurface(name, type, model, finish, value);

    unsigned index = 2000 ; // TODO: eliminate this index, or automate it 
    GBorderSurface* bs = new GBorderSurface( name, index, os ); 
    bs->setBorderSurface(pv1, pv2);   

    float detect_ = 0.f ; 
    float absorb_ = 1.f ;     // <--- perfect absorber just like G4OpBoundaryProcess RINDEX->NoRINDEX
    float reflect_specular_ = 0.f ; 
    float reflect_diffuse_ = 0.f ; 

    addPerfectProperties(bs, detect_, absorb_, reflect_specular_ , reflect_diffuse_ ); 

    bool implicit = true ; 
    bool direct = true ; 
    add(bs, implicit, direct );     
}



void GSurfaceLib::dumpImplicitBorderSurfaces(const char* msg) const 
{
    LOG(info) << msg << std::endl << descImplicitBorderSurfaces() ; 
}

unsigned GSurfaceLib::getNumImplicitBorderSurface() const
{
    unsigned count = 0 ; 

    for(unsigned i=0 ; i < m_border_surfaces.size() ; i++)
    {
        const  GBorderSurface* bs = m_border_surfaces[i] ; 
        bool implicit = bs->isImplicit(); 
        count += int(implicit); 
    }
    return count ; 
}



std::string GSurfaceLib::descImplicitBorderSurfaces() const 
{
    std::stringstream ss ; 
    ss 
        << " num_implicit_border_surfaces " << getNumImplicitBorderSurface()  << std::endl ; 

    for(unsigned i=0 ; i < m_border_surfaces.size() ; i++)
    {
        const  GBorderSurface* bs = m_border_surfaces[i] ; 
        bool implicit = bs->isImplicit(); 
        if( implicit ) ss << bs->desc() << std::endl ; 
    }
    return ss.str(); 
}




void GSurfaceLib::addPerfectSurfaces()
{
    GPropertyMap<float>* _detect = makePerfect("perfectDetectSurface"    , 1.f, 0.f, 0.f, 0.f );
    GPropertyMap<float>* _absorb = makePerfect("perfectAbsorbSurface"    , 0.f, 1.f, 0.f, 0.f );
    GPropertyMap<float>* _specular = makePerfect("perfectSpecularSurface", 0.f, 0.f, 1.f, 0.f );
    GPropertyMap<float>* _diffuse  = makePerfect("perfectDiffuseSurface" , 0.f, 0.f, 0.f, 1.f );

    addDirect(_detect);
    addDirect(_absorb);
    addDirect(_specular);
    addDirect(_diffuse);
}









bool GSurfaceLib::checkSurface( GPropertyMap<float>* surf)
{
    // After standardization in GBoundaryLib surfaces must have four properties
    // that correspond to the probabilities of what happens at an intersection
    // with the surface.  These need to add to one across the wavelength domain.

    GProperty<float>* _detect = surf->getProperty(detect);
    GProperty<float>* _absorb = surf->getProperty(absorb);
    GProperty<float>* _reflect_specular = surf->getProperty(reflect_specular);
    GProperty<float>* _reflect_diffuse  = surf->getProperty(reflect_diffuse);

    if(!_detect) return false ; 
    if(!_absorb) return false ; 
    if(!_reflect_specular) return false ; 
    if(!_reflect_diffuse) return false ; 


    GProperty<float>* sum = GProperty<float>::make_addition( _detect, _absorb, _reflect_specular, _reflect_diffuse );
    GAry<float>* vals = sum->getValues();
    GAry<float>* ones = GAry<float>::ones(sum->getLength());
    float diff = GAry<float>::maxdiff( vals, ones );
    bool valid = diff < 1e-6 ; 

    if(!valid && surf->hasDefinedName())
    {
        LOG(warning) << "GSurfaceLib::checkSurface " << surf->getName() << " diff " << diff ; 
    }
    return valid ; 
}


GItemList* GSurfaceLib::createNames()
{
    GItemList* names = new GItemList(getType());
    unsigned int ni = getNumSurfaces();
    for(unsigned int i=0 ; i < ni ; i++)
    {
        GPropertyMap<float>* surf = m_surfaces[i] ;
        names->add(surf->getShortName());
    }
    return names ; 
}


/**
GSurfaceLib::collectSensorIndices
----------------------------------

Loops over all surfaces collecting the 
indices of surfaces having non-zero EFFICIENCY or detect
properties.

**/

void GSurfaceLib::collectSensorIndices()
{
    unsigned ni = getNumSurfaces();
    for(unsigned i=0 ; i < ni ; i++)
    {
        GPropertyMap<float>* surf = m_surfaces[i] ;
        bool is_sensor = surf->isSensor() ; 
        if(is_sensor)
        {
            addSensorIndex(i); 
            assert( isSensorIndex(i) == true ) ; 
        }
    }
}


void GSurfaceLib::dumpSurfaces(const char* msg)
{
    unsigned ni = getNumSurfaces();
    LOG(info) << msg << " num_surfaces " << ni ; 

    for(unsigned i=0 ; i < ni ; i++)
    {
        GPropertyMap<float>* surf = m_surfaces[i] ;
        const char* name = surf->getShortName() ;
        const char* type = surf->getType() ; 
        bool is_sensor = surf->isSensor() ; 
        std::cout 
             << " index : " << std::setw(2) << i 
             << " is_sensor : " << ( is_sensor ? "Y" : "N" )
             << " type : " << std::setw(20) << type
             << " name : " << std::setw(50) << name
             ;  

        if(surf->isBorderSurface())
        {
             std::cout 
                  << " bpv1 " << surf->getBPV1()      
                  << " bpv2 " << surf->getBPV2()      
                  ;
        }  
        else if(surf->isSkinSurface())
        {
             std::cout 
                  << " sslv " << surf->getSSLV()      
                  ;
 
        }
        std::cout << " ." << std::endl ; 
    }
}



/**
BMeta* GSurfaceLib::createMeta()
---------------------------------

Compose collective metadata keyed by surface shortnames
with all metadata for each surface.

**/

BMeta* GSurfaceLib::createMeta()
{
    LOG(LEVEL) ; 
    BMeta* libmeta = new BMeta ; 
    unsigned int ni = getNumSurfaces();
    for(unsigned int i=0 ; i < ni ; i++)
    {
        GPropertyMap<float>* surf = getSurface(i) ;
        const char* key = surf->getShortName() ;
        BMeta* surfmeta = surf->getMeta();
        assert( surfmeta );
        libmeta->setObj(key, surfmeta );  
    }
    return libmeta ; 
}




NPY<float>* GSurfaceLib::createBuffer()
{
    return createBufferForTex2d() ; 
}


NPY<float>* GSurfaceLib::createBufferForTex2d()
{
    // memory layout of this buffer is constrained by 
    // need to match the requirements of tex2d<float4>

    unsigned int ni = getNumSurfaces();                 // ~48
    unsigned int nj = NUM_FLOAT4 ;                      // 2 
    unsigned int nk = getStandardDomain()->getLength(); // 39
    unsigned int nl = 4 ;

    if(ni == 0 || nj == 0)
    {
        LOG(error) << "GSurfaceLib::createBufferForTex2d"
                   << " zeros "
                   << " ni " << ni      
                   << " nj " << nj
                   ;
        return NULL ;        
    }  


    NPY<float>* buf = NPY<float>::make(ni, nj, nk, nl);  // surfaces/payload-group/wavelength-samples/4prop
    buf->zero();
    float* data = buf->getValues();

    for(unsigned int i=0 ; i < ni ; i++)
    {
        GPropertyMap<float>* surf = m_surfaces[i] ;
        GProperty<float> *p0,*p1,*p2,*p3 ; 

        for(unsigned int j=0 ; j < nj ; j++)
        {
            p0 = surf->getPropertyByIndex(j*4+0);
            p1 = surf->getPropertyByIndex(j*4+1);
            p2 = surf->getPropertyByIndex(j*4+2);
            p3 = surf->getPropertyByIndex(j*4+3);

            for( unsigned int k = 0; k < nk; k++ )    // over wavelength-samples
            {  
                unsigned int offset = i*nj*nk*nl + j*nk*nl + k*nl ;  

                data[offset+0] = p0 ? p0->getValue(k) : SURFACE_UNSET ;
                data[offset+1] = p1 ? p1->getValue(k) : SURFACE_UNSET ;
                data[offset+2] = p2 ? p2->getValue(k) : SURFACE_UNSET ;
                data[offset+3] = p3 ? p3->getValue(k) : SURFACE_UNSET ;
            }
        }
    }
    return buf ; 
}



NPY<float>* GSurfaceLib::createBufferOld()
{
    unsigned int ni = getNumSurfaces();
    unsigned int nj = getStandardDomain()->getLength();
    unsigned int nk = NUM_PROP  ;  // 4 or 8  
    assert( nk == 4 || nk == 8 ); 
    assert(ni > 0 && nj > 0);

    NPY<float>* buf = NPY<float>::make(ni, nj, nk); 
    buf->zero();

    float* data = buf->getValues();

    GProperty<float> *p0,*p1,*p2,*p3 ; 
    GProperty<float>* p4(NULL);
    GProperty<float>* p5(NULL);
    GProperty<float>* p6(NULL);
    GProperty<float>* p7(NULL);

    for(unsigned int i=0 ; i < ni ; i++)
    {
        GPropertyMap<float>* surf = m_surfaces[i] ;
        p0 = surf->getPropertyByIndex(0);
        p1 = surf->getPropertyByIndex(1);
        p2 = surf->getPropertyByIndex(2);
        p3 = surf->getPropertyByIndex(3);
        if(nk > 4)
        {
            p4 = surf->getPropertyByIndex(4);
            p5 = surf->getPropertyByIndex(5);
            p6 = surf->getPropertyByIndex(6);
            p7 = surf->getPropertyByIndex(7);
        }

        for( unsigned int j = 0; j < nj; j++ ) // interleave 4/8 properties into the buffer
        {   
            unsigned int offset = i*nj*nk + j*nk ;  
            data[offset+0] = p0->getValue(j) ;
            data[offset+1] = p1->getValue(j) ;
            data[offset+2] = p2->getValue(j) ;
            data[offset+3] = p3->getValue(j) ;

            if(nk > 4)
            {
                data[offset+4] = p4 ? p4->getValue(j) : SURFACE_UNSET ;
                data[offset+5] = p5 ? p5->getValue(j) : SURFACE_UNSET ;
                data[offset+6] = p6 ? p6->getValue(j) : SURFACE_UNSET ;
                data[offset+7] = p7 ? p7->getValue(j) : SURFACE_UNSET ;
            }
        } 
    }
    return buf ; 
}

void GSurfaceLib::import()
{
    LOG(LEVEL) ; 
    if(m_buffer == NULL)
    {
        setValid(false);
        LOG(warning) << "GSurfaceLib::import NULL buffer " ; 
        return ;  
    }

    LOG(debug) << "GSurfaceLib::import "    
               << " buffer shape " << m_buffer->getShapeString()
             ;

    assert( m_buffer->getNumItems() == m_names->getNumKeys() );

    importForTex2d();
    //importOld();
}


/**
GSurfaceLib::importForTex2d
------------------------------

1. surfmeta gets pulled out of the collective libmeta and 
   set into the reconstituted GPropertyMap

* observe : GOpticalSurface not reconstructed ?

**/


void GSurfaceLib::importForTex2d()
{
    unsigned int ni = m_buffer->getShape(0); // surfaces
    unsigned int nj = m_buffer->getShape(1); // payload categories 
    unsigned int nk = m_buffer->getShape(2); // wavelength samples
    unsigned int nl = m_buffer->getShape(3); // 4 props
    LOG(LEVEL)  
        << " shape " << m_buffer->getShapeString()
        ; 

    assert(m_standard_domain->getLength() == nk );

    float* data = m_buffer->getValues();

    for(unsigned int i=0 ; i < ni ; i++)
    {
        const char* key = m_names->getKey(i);

        GOpticalSurface* os = NULL ;  // huh : not reconstructed ?

        BMeta* surfmeta = m_meta ? m_meta->getObj(key) : NULL  ;  

        const char* surftype = surfmeta ? AssignSurfaceType(surfmeta) : NULL ;


        if(surftype == NULL)
        {
            LOG(fatal) << "GSurfaceLib::assignSurfaceType FAILED " ; 
            if(surfmeta)
                surfmeta->dump();

            LOG(fatal) << " GSurfaceLib loaded from geocache lacks required metadata "
                       << " try recreating geocache, with -G option. "
                       << " Example commandline :  op --j1707 -G  "
                       ;

        } 

        assert(surftype); 

        LOG(LEVEL) << " i " << std::setw(3) << i 
                     << " surftype " << std::setw(15) << surftype
                     << " key " << key 
                     ;


        GPropertyMap<float>* surf = new GPropertyMap<float>(key,i, surftype, os, surfmeta );

        for(unsigned int j=0 ; j < nj ; j++)
        {
            import(surf, data + i*nj*nk*nl + j*nk*nl , nk, nl, j );
        }


        // hmm: bit dirty, should persist the domain
        // here just assuming the loaded lib used the standard 
        assert(m_standard_domain->getLength() == nk );
        surf->setStandardDomain( m_standard_domain );
        assert( surf->hasStandardDomain() );

        m_surfaces.push_back(surf);
    }  


    if(m_dbgsurf)
    {
        LOG(info) << " --dbgsurf dumpMeta " ; 
        dumpMeta(); 
    }
}


void GSurfaceLib::dumpMeta(const char* msg) const 
{
    unsigned num_surf = m_surfaces.size() ; 
    LOG(info) << msg << " num_surf " << num_surf ; 
    for( unsigned i=0 ; i < num_surf ; i++)
    {
         GPropertyMap<float>* surf = m_surfaces[i]; 
         std::cout << " i " << std::setw(3) << i
                   << " surf.desc " << surf->desc() 
                   << std::endl 
                   ; 
 
    }
}





void GSurfaceLib::importOld()
{
    unsigned int ni = m_buffer->getShape(0);
    unsigned int nj = m_buffer->getShape(1);
    unsigned int nk = m_buffer->getShape(2);

    checkBufferCompatibility(nk, "GSurfaceLib::import");

   assert(m_standard_domain->getLength() == nj );
   float* data = m_buffer->getValues();

   for(unsigned int i=0 ; i < ni ; i++)
   {
       const char* key = m_names->getKey(i);
       LOG(debug) << std::setw(3) << i 
                 << " " << key ;

       GOpticalSurface* os = NULL ;
       GPropertyMap<float>* surf = new GPropertyMap<float>(key,i,"surface", os);
       import(surf, data + i*nj*nk, nj, nk );

       m_surfaces.push_back(surf);
   }  
}

void GSurfaceLib::import( GPropertyMap<float>* surf, float* data, unsigned int nj, unsigned int nk, unsigned int jcat )
{
    float* domain = m_standard_domain->getValues();

    for(unsigned int k = 0 ; k < nk ; k++)
    {
        float* values = new float[nj] ; 
        for(unsigned int j = 0 ; j < nj ; j++) values[j] = data[j*nk+k]; 
        GProperty<float>* prop = new GProperty<float>( values, domain, nj );

        surf->addProperty(propertyName(k+4*jcat), prop);
    } 
}


void GSurfaceLib::importOpticalBuffer(NPY<unsigned int>* ibuf)
{
    // invoked by load after loadFromCache and import have run 

    std::vector<guint4> optical ; 
    importUint4Buffer(optical, ibuf);  

    // thence can revivify GOpticalSurface and associate them to the GPropertyMap<float>*
    
    unsigned int ni = optical.size();
    assert(ni == getNumSurfaces());
    assert(ni == m_names->getNumKeys() );

    for(unsigned int i=0 ; i < ni ; i++)
    {
       const char* key = m_names->getKey(i);
       GPropertyMap<float>* surf = getSurface(i);
       GOpticalSurface* os = GOpticalSurface::create( key, optical[i] );
       surf->setOpticalSurface(os);
    }
}

NPY<unsigned int>* GSurfaceLib::createOpticalBuffer()
{
    std::vector<guint4> optical ; 
    unsigned int ni = getNumSurfaces();
    for(unsigned int i=0 ; i < ni ; i++) optical.push_back(getOpticalSurface(i));
    return createUint4Buffer(optical);
}






void GSurfaceLib::dump(const char* msg)
{
    Summary(msg);

    unsigned int ni = getNumSurfaces() ; 

    LOG(info) << " (index,type,finish,value) " ;  

    for(unsigned int i=0 ; i < ni ; i++)
    {
        guint4 optical = getOpticalSurface(i);
        GPropertyMap<float>* surf = getSurface(i);

        LOG(warning) << std::setw(35) << surf->getName() 
                  <<  GOpticalSurface::brief(optical) 
                  ;
    } 

    int index = m_ok->getLastArgInt();
    const char* lastarg = m_ok->getLastArg();
    if(hasSurface(index))
        dump(index);
    else if(hasSurface(lastarg))
        dump(getSurface(lastarg));
    else
        for(unsigned int i=0 ; i < ni ; i++) dump(i);
}




void GSurfaceLib::dump( unsigned int index )
{
    guint4 optical = getOpticalSurface(index);
    GPropertyMap<float>* surf = getSurface(index);
    assert(surf->getIndex() == index ); 
    std::string desc = optical.description() + surf->description() ; 
    dump(surf, desc.c_str());

    surf->dumpMeta("GSurfaceLib::dump.index");

}

void GSurfaceLib::dump(GPropertyMap<float>* surf)
{
    unsigned int index = surf->getIndex();
    guint4 optical = getOpticalSurface(index);
    std::string desc = optical.description() + surf->description() ; 
    dump(surf, desc.c_str());
}

void GSurfaceLib::dump( GPropertyMap<float>* surf, const char* msg)
{
    GProperty<float>* _detect = surf->getProperty(detect);
    GProperty<float>* _absorb = surf->getProperty(absorb);
    GProperty<float>* _reflect_specular = surf->getProperty(reflect_specular);
    GProperty<float>* _reflect_diffuse  = surf->getProperty(reflect_diffuse);
    GProperty<float>* _extra_x  = surf->getProperty(extra_x);

    assert(_detect);
    assert(_absorb);
    assert(_reflect_specular);
    assert(_reflect_diffuse);

    float dscale = 1.f ; 
    bool dreciprocal = false ; 

    std::string table = GProperty<float>::make_table( 20, dscale, dreciprocal,
                            _detect, "detect", 
                            _absorb, "absorb",  
                            _reflect_specular, "reflect_specular",
                            _reflect_diffuse , "reflect_diffuse", 
                            _extra_x ,  "extra_x"
                            );
    
    LOG(info) << msg << " " 
              << surf->getName()  
              << "\n" << table 
              ; 
}

/**
GSurfaceLib::getNumSensorSurface
---------------------------------

Based on name ending "SensorSurface"


**/

unsigned GSurfaceLib::getNumSensorSurface() const
{
    unsigned count = 0 ; 
    for(unsigned index=0 ; index < getNumSurfaces() ; index++)
    {
        if(isSensorSurface(index)) 
        {
            count += 1 ; 
        } 
    } 
    return count ; 
}


GPropertyMap<float>* GSurfaceLib::getSensorSurface(unsigned int offset)
{
    GPropertyMap<float>* ss = NULL ; 


    unsigned int count = 0 ; 
    for(unsigned int index=0 ; index < getNumSurfaces() ; index++)
    {
        const char* name = getName(index); 
        if(isSensorSurface(index))
        {
             if(count == offset) ss = getSurface(index) ;
             count += 1 ; 

        } 
        LOG(debug) << "GSurfaceLib::getSensorSurface" 
                   << " name " << name
                   << " index " << index
                   << " count " << count 
                  ;

    }
    return ss ; 
}  


/**
GSurfaceLib::isSensorSurface
-----------------------------

Called from AssimpGGeo::convertStructureVisit
**/

bool GSurfaceLib::isSensorSurface(unsigned int qsurface) const 
{
    // "SensorSurface" name suffix based, see AssimpGGeo::convertSensor
   
    const char* name = getName(qsurface); 
    if(!name) return false ; 

    bool iss = NameEndsWithSensorSurface(name) ;     

    if(iss)
    LOG(debug) << "GSurfaceLib::isSensorSurface"
              << " surface " << qsurface  
              << " name " << name 
              << " iss " << iss 
              ;

    return iss ; 
}





bool GSurfaceLib::hasSurface(const char* name) const 
{
    return getSurface(name) != NULL ; 
}

bool GSurfaceLib::hasSurface(unsigned int index) const 
{
    return getSurface(index) != NULL ; 
}

GPropertyMap<float>* GSurfaceLib::getSurface(unsigned int i) const 
{
    return i < m_surfaces.size() ? m_surfaces[i] : NULL ;
}




GPropertyMap<float>* GSurfaceLib::getSurface(const char* name) const 
{
    if(!name) return NULL ; 

    GPropertyMap<float>* surf = NULL ; 
    for(unsigned i=0 ; i < m_surfaces.size() ; i++)
    {
        GPropertyMap<float>* s = m_surfaces[i] ; 
        const char* sname = s->getName() ; 
        assert( sname ) ; 
        if( strcmp( sname, name ) == 0 )
        {
             surf = s ; 
             break ; 
        } 
    }
    return surf ; 
}

GProperty<float>* GSurfaceLib::getSurfaceProperty(const char* name, const char* prop) const
{
    GPropertyMap<float>* surf = getSurface(name) ; 
    assert(surf); 
    GProperty<float>* p = surf->getProperty(prop); 
    return p ; 
} 




/*
// this way only works after closing the lib, or for loaded libs 
// as it uses the names buffer
// 
GPropertyMap<float>* GSurfaceLib::getSurface(const char* name)
{
    unsigned int index = m_names->getIndex(name);
    GPropertyMap<float>* surf = getSurface(index);
    return surf ; 
}
*/











/**
Simple collection relocated from GGeo
**/

unsigned GSurfaceLib::getNumBorderSurfaces() const 
{
    return m_border_surfaces.size();
}
unsigned GSurfaceLib::getNumSkinSurfaces() const 
{
    return m_skin_surfaces.size();
}

GSkinSurface* GSurfaceLib::getSkinSurface(unsigned index) const 
{
    return m_skin_surfaces[index];
}
GBorderSurface* GSurfaceLib::getBorderSurface(unsigned index) const 
{
    return m_border_surfaces[index];
}


/**
GSurfaceLib::findSkinSurface
------------------------------

Returns skin surface associated with the named logical volume or nullptr.

**/

GSkinSurface* GSurfaceLib::findSkinSurface(const char* lv) const 
{
    GSkinSurface* ss = NULL ; 
    for(unsigned int i=0 ; i < m_skin_surfaces.size() ; i++ )
    {
         GSkinSurface* s = m_skin_surfaces[i];
         if(s->matches(lv))   
         {
            ss = s ; 
            break ; 
         } 
    }
    return ss ;
}

void GSurfaceLib::dumpSkinSurface(const char* msg) const
{
    LOG(LEVEL) << msg ; 

    for(unsigned i = 0 ; i < m_skin_surfaces.size() ; i++)
    {
        GSkinSurface* ss = m_skin_surfaces[i];
        LOG(LEVEL)
            << " SS " 
            << std::setw(4) << i 
            << " : " 
            << std::setw(40) << ss->getShortName() 
            << " : " 
            << ss->getSkinSurfaceVol()
            ;
    }
}


/**
GSurfaceLib::findBorderSurface
--------------------------------

Returns border surface between the named physical volumes or nullptr if not found.

**/

GBorderSurface* GSurfaceLib::findBorderSurface(const char* pv1, const char* pv2) const 
{
    GBorderSurface* bs = NULL ; 
    for(unsigned int i=0 ; i < m_border_surfaces.size() ; i++ )
    {
        GBorderSurface* s = m_border_surfaces[i];
        if(s->matches(pv1,pv2))   
        {
            bs = s ; 
            break ; 
        } 
    }
    return bs ;
}








/**
GSurfaceLib::addRaw : USED FOR DEBUG ONLY 
--------------------------------------------

Adds into separate m_border_surfaces_raw m_skin_surfaces_raw vectors

TODO: 

SUSPECT CAN ELIMINATE THE _raw ?  AS SEEMS THAT m_border_surfaces and m_skin_surfaces do the same thing 






**/

void GSurfaceLib::addRaw(GBorderSurface* surface)
{
    m_border_surfaces_raw.push_back(surface);
}
void GSurfaceLib::addRaw(GSkinSurface* surface)
{
    m_skin_surfaces_raw.push_back(surface);
}

unsigned GSurfaceLib::getNumRawBorderSurfaces() const
{
    return m_border_surfaces_raw.size();
}
unsigned GSurfaceLib::getNumRawSkinSurfaces() const 
{
    return m_skin_surfaces_raw.size();
}

void GSurfaceLib::dumpRawSkinSurface(const char* name) const
{
    LOG(info) << name ; 

    GSkinSurface* ss = NULL ; 
    unsigned n = getNumRawSkinSurfaces();
    for(unsigned i = 0 ; i < n ; i++)
    {
        ss = m_skin_surfaces_raw[i];
        ss->Summary("dumpRawSkinSurface", 10); 
    }
}

void GSurfaceLib::dumpRawBorderSurface(const char* name) const 
{
    LOG(info) << name ; 
    GBorderSurface* bs = NULL ; 
    unsigned n = getNumRawBorderSurfaces();
    for(unsigned i = 0 ; i < n ; i++)
    {
        bs = m_border_surfaces_raw[i];
        bs->Summary("dumpRawBorderSurface", 10); 
    }
}




