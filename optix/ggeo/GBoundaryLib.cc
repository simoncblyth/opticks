#include "GBoundaryLib.hh"
#include "GBoundaryLibMetadata.hh"
#include "GItemIndex.hh"

#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
//#include "GItemIndex.hh"

#include "GCache.hh"
#include "GBoundary.hh"
#include "GPropertyMap.hh"
#include "GBuffer.hh"
#include "GEnums.hh"
#include "GOpticalSurface.hh"
#include "md5digest.hpp"
#include "limits.h"



// npy-
#include "Counts.hpp"
#include "jsonutil.hpp"
#include "stringutil.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>

#include <string>
#include <map>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>


#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



#include "assert.h"
#include "stdio.h"
#include "limits.h"
#include "string.h"

//unsigned int GBoundaryLib::NUM_QUAD    = 6  ; 
unsigned int GBoundaryLib::NUM_QUAD    = 4  ; 

float        GBoundaryLib::SURFACE_UNSET = -1.f ; 
float        GBoundaryLib::EXTRA_UNSET = -1.f ; 

const char* GBoundaryLib::inner = "inner_" ;
const char* GBoundaryLib::outer = "outer_" ;

// NB below strings should match names in GEnums.hh
// material
const char* GBoundaryLib::refractive_index  = "refractive_index" ;
const char* GBoundaryLib::absorption_length = "absorption_length" ;
const char* GBoundaryLib::scattering_length = "scattering_length" ;
const char* GBoundaryLib::reemission_prob   = "reemission_prob" ;

// surface
const char* GBoundaryLib::detect            = "detect" ;
const char* GBoundaryLib::absorb            = "absorb" ;
const char* GBoundaryLib::reflect_specular  = "reflect_specular" ;
const char* GBoundaryLib::reflect_diffuse   = "reflect_diffuse" ;

// extra
const char* GBoundaryLib::reemission_cdf    = "reemission_cdf" ; // NOT USED : AS NEEDED MORE BINS FOR inverted CDF SO USED SEPARATE reemission_buffer
const char* GBoundaryLib::extra_x           = "extra_x" ;
const char* GBoundaryLib::extra_y           = "extra_y" ;
const char* GBoundaryLib::extra_z           = "extra_z" ;
const char* GBoundaryLib::extra_w           = "extra_w" ;

// workings for "extra"
//const char* GBoundaryLib::slow_component    = "slow_component" ;
//const char* GBoundaryLib::fast_component    = "fast_component" ;

const char* GBoundaryLib::REFLECTIVITY = "REFLECTIVITY" ;
const char* GBoundaryLib::EFFICIENCY   = "EFFICIENCY" ;

const char* GBoundaryLib::keymap = 
"refractive_index:RINDEX,"
"absorption_length:ABSLENGTH,"
"scattering_length:RAYLEIGH,"
"reemission_prob:REEMISSIONPROB," 
"detect:EFFICIENCY," 
"absorb:DUMMY," 
"reflect_specular:REFLECTIVITY," 
"reflect_diffuse:REFLECTIVITY," 
"slow_component:SLOWCOMPONENT," 
"fast_component:FASTCOMPONENT," 
"reemission_cdf:DUMMY," 
;

//
//  ggv --blib 102 103 104 105 106 107
//

const char* GBoundaryLib::scintillators = "LiquidScintillator,GdDopedLS,LS" ;
const char* GBoundaryLib::reemissionkey = "SLOWCOMPONENT,FASTCOMPONENT" ;

std::vector<std::string>* GBoundaryLib::vscintillators = NULL ;
std::vector<std::string>* GBoundaryLib::vreemissionkey = NULL ;



bool GBoundaryLib::isReemissionKey(std::string& lkey)
{
    if(vreemissionkey == NULL)
    {
        vreemissionkey = new std::vector<std::string> ;
        boost::split(*vreemissionkey, reemissionkey, boost::is_any_of(","));
    }
    assert(vreemissionkey);
    for(unsigned int i=0 ; i < vreemissionkey->size() ; i++)
    {
       if( strcmp(lkey.c_str(), (*vreemissionkey)[i].c_str()) == 0 ) return true ;
    }
    return false ; 
}

bool GBoundaryLib::isScintillator(std::string& matShortName)
{
    if(vscintillators == NULL)
    {
        vscintillators = new std::vector<std::string> ;
        boost::split(*vscintillators, scintillators, boost::is_any_of(","));
    }

    assert(vscintillators);
    for(unsigned int i=0 ; i < vscintillators->size() ; i++)
    {
       if( strcmp(matShortName.c_str(), (*vscintillators)[i].c_str()) == 0 ) return true ;
    }
    return false ; 
}



void GBoundaryLib::init()
{
    setKeyMap(keymap);
    defineDefaults(getDefaults());

    nameConstituents(m_names);
    
    m_ramp = makeRampProperty();

    m_meta = new GBoundaryLibMetadata ; 
    m_materials = new GItemIndex("GMaterialIndex") ; // names for compatibility with prior classes 
    m_surfaces  = new GItemIndex("GSurfaceIndex") ; 
}


unsigned int GBoundaryLib::getNumBoundary()
{
   return m_keys.size();
}

GBoundary* GBoundaryLib::getBoundary(unsigned int index)
{
    GBoundary* boundary = NULL ;
    if(index < m_keys.size())
    {
        std::string key = m_keys[index] ;  
        boundary = m_registry[key];
        if(boundary->getIndex() != index )
        {
            printf("GBoundaryLib::getBoundary WARNING boundary index mismatch request %u boundary %u key %s \n", index, boundary->getIndex(), key.c_str() ); 
        } 
        assert(boundary->getIndex() == index );
    }
    return boundary ; 
}



unsigned int GBoundaryLib::getLine(unsigned int isub, unsigned int ioff)
{
    assert(ioff < NUM_QUAD);
    return isub*NUM_QUAD + ioff ;   
}

unsigned int GBoundaryLib::getLineMin()
{
    unsigned int lineMin = getLine(0, 0);
    return lineMin ; 
}
unsigned int GBoundaryLib::getLineMax()
{
    unsigned int numBoundary = getNumBoundary() ; 
    unsigned int lineMax = getLine(numBoundary - 1, NUM_QUAD-1);
    return lineMax ; 
}


void GBoundaryLib::Summary(const char* msg)
{
    unsigned lmin = getLineMin();
    unsigned lmax = getLineMax();

    LOG(info) << msg 
              << " lineMin " << std::setw(3) << lmin 
              << " lineMax " << std::setw(3) << lmax 
              ;

    for(unsigned int isub=0 ; isub < getNumBoundary() ; isub++)
    {
         GBoundary* boundary = getBoundary(isub);
         unsigned int lineMin = getLine(isub, 0);
         unsigned int lineMax = getLine(isub, NUM_QUAD-1);
         std::cout << "lineMin:Max " 
                   << std::setw(3) << lineMin 
                   << ":"
                   << std::setw(3) << lineMax
                   << boundary->description()
                   << std::endl ; 
    } 
}

GBoundary* GBoundaryLib::getOrCreate(
           GPropertyMap<float>* imaterial, 
           GPropertyMap<float>* omaterial, 
           GPropertyMap<float>* isurface, 
           GPropertyMap<float>* osurface,
           GPropertyMap<float>* iextra,
           GPropertyMap<float>* oextra
      )
{ 
    // this "get" pulls the GBoundary into existance and populates the registry
    //printf("GBoundaryLib::get imaterial %p omaterial %p isurface %p osurface %p \n", imaterial, omaterial, isurface, osurface );

    assert(imaterial);
    assert(omaterial);

    GBoundary raw(imaterial, omaterial, isurface, osurface, iextra, oextra);
    GBoundary* standard = createStandardBoundary(&raw) ;

    bool old = NUM_QUAD == 6 ; 
    std::string key = old ? standard->pdigest(0,4) : standard->digest() ;  
    //
    // old way: standard digest based identity 
    // new way: name based identity, to align with GBndLib
    //

    if(m_registry.count(key) == 0) // not yet registered
    { 
        standard->setIndex(m_keys.size());
        m_keys.push_back(key);  // for simple ordering  
        m_registry[key] = standard ; 
    }
    else
    {
        delete standard ; 
    } 

    GBoundary* boundary = m_registry[key] ;
    //printf("GBoundaryLib::get key %s index %u \n", key.c_str(), boundary->getIndex()); 
    return boundary ; 
}



GBoundary* GBoundaryLib::createStandardBoundary(GBoundary* boundary)
{
    assert(boundary);

    GPropertyMap<float>* imat = boundary->getInnerMaterial();
    GPropertyMap<float>* omat = boundary->getOuterMaterial();
    GPropertyMap<float>* isur = boundary->getInnerSurface();
    GPropertyMap<float>* osur = boundary->getOuterSurface();
    GPropertyMap<float>* iext = boundary->getInnerExtra();
    GPropertyMap<float>* oext = boundary->getOuterExtra();

    assert(imat);
    assert(omat);

    GPropertyMap<float>* s_imat = new GPropertyMap<float>(imat);
    GPropertyMap<float>* s_omat = new GPropertyMap<float>(omat);
    GPropertyMap<float>* s_isur = new GPropertyMap<float>(isur);
    GPropertyMap<float>* s_osur = new GPropertyMap<float>(osur);
    GPropertyMap<float>* s_iext = new GPropertyMap<float>(iext);
    GPropertyMap<float>* s_oext = new GPropertyMap<float>(oext);

    assert(s_imat);
    assert(s_omat);

    standardizeMaterialProperties( s_imat, imat, inner );
    standardizeMaterialProperties( s_omat, omat, outer );
    standardizeSurfaceProperties(  s_isur, isur, inner );
    standardizeSurfaceProperties(  s_osur, osur, outer );
    standardizeExtraProperties(    s_iext, iext, inner );
    standardizeExtraProperties(    s_oext, oext, outer );

    s_isur->setValid(checkSurface( s_isur, inner )); 
    s_osur->setValid(checkSurface( s_osur, outer )); 


    if(isur && s_isur->hasDefinedName() && !s_isur->isValid())
    {
        LOG(warning) << "GBoundaryLib::createStandardBoundary INVALID inner surface " << s_isur->getName() ;  
        dumpSurface(s_isur, inner, "GBoundaryLib::createStandardBoundary INVALID inner surface");
    }
    if(osur && s_osur->hasDefinedName() && !s_osur->isValid())
    {
        LOG(warning) << "GBoundaryLib::createStandardBoundary INVALID outer surface " << s_osur->getName() ;  
        dumpSurface(s_osur, outer, "GBoundaryLib::createStandardBoundary INVALID outer surface");
    }

    GBoundary* s_boundary = new GBoundary( s_imat , s_omat, s_isur, s_osur, s_iext, s_oext);

    return s_boundary ; 
}



void GBoundaryLib::dumpSurfaces(const char* msg)
{
    LOG(info) << msg ; 
    unsigned int numBoundary = getNumBoundary() ; 
    for(unsigned int i=0 ; i < numBoundary ; i++)
    {
        GBoundary* boundary = getBoundary(i);

        LOG(info) << boundary->description() ;
        GPropertyMap<float>* s_isur = boundary->getInnerSurface(); 
        GPropertyMap<float>* s_osur = boundary->getOuterSurface(); 
        
        if(s_isur->isValid())
            dumpSurface(s_isur, inner, s_isur->getName() );

        if(s_osur->isValid())
            dumpSurface(s_osur, outer, s_osur->getName() );
    }
}







GPropertyMap<float>* GBoundaryLib::createStandardProperties(const char* pname, GBoundary* boundary)
{
    // combining all 6 sets into one PropertyMap : for insertion into wavelengthBuffer

    GPropertyMap<float>* ptex = new GPropertyMap<float>(pname);
    
    ptex->add(boundary->getInnerMaterial(), inner);
    ptex->add(boundary->getOuterMaterial(), outer);
    ptex->add(boundary->getInnerSurface(),  inner);
    ptex->add(boundary->getOuterSurface(),  outer);
    ptex->add(boundary->getInnerExtra(),    inner);
    ptex->add(boundary->getOuterExtra(),    outer);

    checkMaterialProperties(ptex,  0*4 , inner);
    checkMaterialProperties(ptex,  1*4 , outer);
    checkSurfaceProperties( ptex,  2*4 , inner);
    checkSurfaceProperties( ptex,  3*4 , outer);
    checkExtraProperties(   ptex,  4*4 , inner);
    checkExtraProperties(   ptex,  5*4 , outer);

    return ptex ; 
}


void GBoundaryLib::import()
{
    LOG(info) << "GBoundaryLib::import placeholder " ;
}
void GBoundaryLib::sort()
{
    LOG(info) << "GBoundaryLib::sort placeholder " ;
}



NPY<float>* GBoundaryLib::createBuffer()
{
    return NULL ; 
}

GItemList* GBoundaryLib::createNames()
{
    return NULL ;
}



void GBoundaryLib::defineDefaults(GPropertyMap<float>* defaults)
{
    defaults->addConstantProperty( refractive_index,      1.f  );
    defaults->addConstantProperty( absorption_length,     1e6  );
    defaults->addConstantProperty( scattering_length,     1e6  );
    defaults->addConstantProperty( reemission_prob,       0.f  );

    defaults->addConstantProperty( detect          ,      SURFACE_UNSET );
    defaults->addConstantProperty( absorb          ,      SURFACE_UNSET );
    defaults->addConstantProperty( reflect_specular,      SURFACE_UNSET );
    defaults->addConstantProperty( reflect_diffuse ,      SURFACE_UNSET );

    defaults->addConstantProperty( extra_x         ,      EXTRA_UNSET );
    defaults->addConstantProperty( extra_y         ,      EXTRA_UNSET );
    defaults->addConstantProperty( extra_z         ,      EXTRA_UNSET );
    defaults->addConstantProperty( extra_w         ,      EXTRA_UNSET );

}

std::vector<std::string> GBoundaryLib::splitString(std::string keys)
{
    std::vector<std::string> vkeys;
    boost::split(vkeys, keys, boost::is_any_of(" "));
    return vkeys ; 
}


void GBoundaryLib::standardizeMaterialProperties(GPropertyMap<float>* pstd, GPropertyMap<float>* pmap, const char* prefix)
{
    assert(pmap);  // materials must always be defined
    assert(pmap->isMaterial());
    assert(getStandardDomain()->isEqual(pmap->getStandardDomain()));

    if(pstd->hasStandardDomain())
    {
        assert(pstd->getStandardDomain()->isEqual(pmap->getStandardDomain()));
    } 
    else
    {
        pstd->setStandardDomain(pmap->getStandardDomain());
    }

    pstd->addProperty(refractive_index, getPropertyOrDefault( pmap, refractive_index ), prefix);
    pstd->addProperty(absorption_length,getPropertyOrDefault( pmap, absorption_length ), prefix);
    pstd->addProperty(scattering_length,getPropertyOrDefault( pmap, scattering_length ), prefix);
    pstd->addProperty(reemission_prob  ,getPropertyOrDefault( pmap, reemission_prob ), prefix);
}




void GBoundaryLib::standardizeExtraProperties(GPropertyMap<float>* pstd, GPropertyMap<float>* pmap, const char* prefix)
{
    //pstd->addProperty(reemission_cdf   , constructReemissionCDF( pmap )              , prefix);
    pstd->addProperty(extra_x          , getPropertyOrDefault( pmap, extra_x )       , prefix);
    pstd->addProperty(extra_y          , getPropertyOrDefault( pmap, extra_y )       , prefix);
    pstd->addProperty(extra_z          , getPropertyOrDefault( pmap, extra_z )       , prefix);
    pstd->addProperty(extra_w          , getPropertyOrDefault( pmap, extra_w )       , prefix);
}

void GBoundaryLib::standardizeSurfaceProperties(GPropertyMap<float>* pstd, GPropertyMap<float>* pmap, const char* prefix)
{
    GProperty<float>* _detect           = NULL ; 
    GProperty<float>* _absorb           = NULL ; 
    GProperty<float>* _reflect_specular = NULL ; 
    GProperty<float>* _reflect_diffuse  = NULL ; 

    if(!pmap)
    {
        _detect           = getDefaultProperty(detect); 
        _absorb           = getDefaultProperty(absorb); 
        _reflect_specular = getDefaultProperty(reflect_specular); 
        _reflect_diffuse  = getDefaultProperty(reflect_diffuse); 

    }
    else
    {
        assert(getStandardDomain()->isEqual(pmap->getStandardDomain()));
        assert(pmap->isSkinSurface() || pmap->isBorderSurface());
        GOpticalSurface* os = pmap->getOpticalSurface() ;  // GSkinSurface and GBorderSurface ctor plant the OpticalSurface into the PropertyMap

        if(pmap->isSensor())
        {
            GProperty<float>* _EFFICIENCY = pmap->getProperty(EFFICIENCY); 
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
            GProperty<float>* _REFLECTIVITY = pmap->getProperty(REFLECTIVITY); 
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

    pstd->setSensor( pmap ? pmap->isSensor() : false ); 
    pstd->addProperty( detect          , _detect          , prefix );
    pstd->addProperty( absorb          , _absorb          , prefix );
    pstd->addProperty( reflect_specular, _reflect_specular, prefix );
    pstd->addProperty( reflect_diffuse , _reflect_diffuse , prefix );
}



void GBoundaryLib::dumpSurface( GPropertyMap<float>* surf, const char* prefix, const char* msg)
{
    GProperty<float>* _detect = surf->getProperty(detect, prefix);
    GProperty<float>* _absorb = surf->getProperty(absorb, prefix);
    GProperty<float>* _reflect_specular = surf->getProperty(reflect_specular, prefix );
    GProperty<float>* _reflect_diffuse  = surf->getProperty(reflect_diffuse, prefix );

    assert(_detect);
    assert(_absorb);
    assert(_reflect_specular);
    assert(_reflect_diffuse);
  

    std::string table = GProperty<float>::make_table( 
                            _detect, "detect", 
                            _absorb, "absorb",  
                            _reflect_specular, "reflect_specular",
                            _reflect_diffuse , "reflect_diffuse", 
                            20 );
    
    LOG(info) << msg << " " 
              << surf->getName()  
              << "\n" << table 
              ; 
}


bool GBoundaryLib::checkSurface( GPropertyMap<float>* surf, const char* prefix)
{
    // After standardization in GBoundaryLib surfaces must have four properties
    // that correspond to the probabilities of what happens at an intersection
    // with the surface.  These need to add to one across the wavelength domain.

    GProperty<float>* _detect = surf->getProperty(detect, prefix);
    GProperty<float>* _absorb = surf->getProperty(absorb, prefix);
    GProperty<float>* _reflect_specular = surf->getProperty(reflect_specular, prefix );
    GProperty<float>* _reflect_diffuse  = surf->getProperty(reflect_diffuse, prefix );

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
        LOG(warning) << "GBoundaryLib::checkSurface " << surf->getName() << " diff " << diff ; 
    }
    return valid ; 
}





void GBoundaryLib::checkMaterialProperties(GPropertyMap<float>* ptex, unsigned int offset, const char* _prefix)
{
    std::string prefix = _prefix ; 
    std::string xname, pname ;

    xname = prefix + refractive_index ; 
    pname = ptex->getPropertyNameByIndex(offset+e_refractive_index);
    assert(xname == pname);

    xname = prefix + absorption_length  ; 
    pname = ptex->getPropertyNameByIndex(offset+e_absorption_length);
    assert(xname == pname);

    xname = prefix + scattering_length  ; 
    pname = ptex->getPropertyNameByIndex(offset+e_scattering_length);
    assert(xname == pname);

    xname = prefix + reemission_prob  ; 
    pname = ptex->getPropertyNameByIndex(offset+e_reemission_prob);
    assert(xname == pname);

}
void GBoundaryLib::checkSurfaceProperties(GPropertyMap<float>* ptex, unsigned int offset, const char* _prefix)
{
    std::string prefix = _prefix ; 
    std::string xname, pname ;

    xname = prefix + detect ; 
    pname = ptex->getPropertyNameByIndex(offset+e_detect);
    assert(xname == pname);

    xname = prefix + absorb ; 
    pname = ptex->getPropertyNameByIndex(offset+e_absorb);
    assert(xname == pname);

    xname = prefix + reflect_specular ; 
    pname = ptex->getPropertyNameByIndex(offset+e_reflect_specular);
    assert(xname == pname);

    xname = prefix + reflect_diffuse ; 
    pname = ptex->getPropertyNameByIndex(offset+e_reflect_diffuse);
    assert(xname == pname);
}

void GBoundaryLib::checkExtraProperties(GPropertyMap<float>* ptex, unsigned int offset, const char* _prefix)
{
    std::string prefix = _prefix ; 
    std::string xname, pname ;

    //xname = prefix + reemission_cdf ; 
    xname = prefix + extra_x ; 
    pname = ptex->getPropertyNameByIndex(offset+e_extra_x);
    assert(xname == pname);

    xname = prefix + extra_y ; 
    pname = ptex->getPropertyNameByIndex(offset+e_extra_y);
    assert(xname == pname);

    xname = prefix + extra_z ; 
    pname = ptex->getPropertyNameByIndex(offset+e_extra_z);
    assert(xname == pname);

    xname = prefix + extra_w ; 
    pname = ptex->getPropertyNameByIndex(offset+e_extra_w);
    assert(xname == pname);
}





const char* GBoundaryLib::getDigest(unsigned int index)
{
    return m_keys[index].c_str();
}



static bool second_order( const std::pair<std::string, unsigned int>& a, const std::pair<std::string, unsigned int>& b ){
    return a.second < b.second ;
}

void GBoundaryLib::countMaterials()
{
    LOG(info) << "GBoundaryLib::countMaterials " ;
    Counts<unsigned int> cmat("countMaterials");

    unsigned int numBoundary = getNumBoundary() ;
    for(unsigned int isub=0 ; isub < numBoundary ; isub++)
    {
        GBoundary* boundary = getBoundary(isub);
        for( unsigned int p = 0; p < NUM_QUAD ; ++p )  // over NUM_QUAD different sets (imat,omat,isur,osur,iext,oext)  
        { 
            GPropertyMap<float>* psrc = boundary->getConstituentByIndex(p) ; 
            if(psrc->isMaterial())
            {
                std::string shortname = psrc->getShortName();
                cmat.add(shortname.c_str());
            }
        }
    }

   cmat.sort();
   cmat.dump();
   cmat.sort(false);
   cmat.dump();
}



void GBoundaryLib::createWavelengthAndOpticalBuffers()
{
    //
    //  wavelength_buffer floats (6 sets of 4 props)  
    //  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //
    //  Six sets for: 
    //          (inner material, outer material, inner surface, outer surface, inner extra, outer extra) 
    //
    //  and the props:
    //         material: (refractive_index, absorption_length, scattering_length, reemission_prob)
    //         surface:  (detect, absorb, reflect_specular, reflect_diffuse)    
    //         extra:    (reemission_cdf, extra_y, extra_z, extra_w )
    //
    //
    //   
    //
    //
    //  optical_buffer unsigned ints (6 sets of 4 values)
    //  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //
    //  Same six sets as wavelength_buffer, with a single quad uint for 
    //  each set, with differing meanings
    //
    //
    //                   material        surface          extra
    //
    //          .x         index            index         -
    //          .y         -                type          -
    //          .z         -                finish        -
    //          .w         -                value         -
    //
    //
    //   Optical surface props
    //
    //        type                : 0:dielectric_metal/... 
    //        finish              : 0:polished  3:ground
    //        value(sigma alpha)  : expressed as integer percentage value, used in facet normal calculation 
    //                              for non-polished 
    //
    //   Other optical surface props
    //
    //        model  : assumed to always be "unified" so omitted
    //        name   : skipped as not needed on GPU, can be looked up via index for GUI selections 
    //


    assert(m_wavelength_buffer == NULL && m_optical_buffer == NULL && "not expecting preexisting wavelength/optical buffers");

    GDomain<float>* domain = getStandardDomain();
    const unsigned int domainLength = domain->getLength();
    unsigned int numBoundary = getNumBoundary() ;
    unsigned int numFloat = domainLength*NUM_QUAD*4*numBoundary ; 
    unsigned int numUInt  =              NUM_QUAD*4*numBoundary ; 


    assert(NUM_QUAD == 6 || NUM_QUAD == 4);
    bool extra = NUM_QUAD == 6 ;  
    bool name = !extra ;

    
    // for comparison againt new GBndLib 
    //    * adopt name based identity
    //    * eliminate the extra 
    //


    m_wavelength_buffer = new GBuffer( sizeof(float)*numFloat, new float[numFloat], sizeof(float), 1 );
    float* wavelength_data = (float*)m_wavelength_buffer->getPointer();

    m_optical_buffer = new GBuffer( sizeof(unsigned int)*numUInt, new unsigned int[numUInt], sizeof(unsigned int), 1 );
    unsigned int* optical_data = (unsigned int*)m_optical_buffer->getPointer();


    for(unsigned int isub=0 ; isub < numBoundary ; isub++)
    {
        GBoundary* boundary = getBoundary(isub);
        unsigned int boundaryIndex = boundary->getIndex();

        //printf("GBoundaryLib::createWavelengthBuffer isub %u/%u boundaryIndex  %u \n", isub, numBoundary, boundaryIndex );
        assert(boundaryIndex < numBoundary);
        assert(isub == boundaryIndex);

        const char* dig = getDigest(isub); // m_keys lookup
        {
            char* ckdig = boundary->pdigest(0,4,name);
            assert(strcmp(ckdig, dig) == 0);
        }

        unsigned int opticalOffset =          NUM_QUAD*4*boundaryIndex ; 
        unsigned int subOffset = domainLength*NUM_QUAD*4*boundaryIndex ; 

        const char* kfmt = "lib.boundary.%d.%s.%s" ;
        m_meta->addDigest(kfmt, isub, "boundary", (char*)dig ); 

        std::string ishortname = boundary->getInnerMaterial()->getShortName() ; 

        std::string oshortname = boundary->getOuterMaterial()->getShortName() ; 

        LOG(debug)<<  __func__
                 << " isn " << ishortname 
                 << " osn " << oshortname 
                 ;

        {
            m_meta->add(kfmt, isub, "imat", boundary->getInnerMaterial() );
            m_meta->add(kfmt, isub, "omat", boundary->getOuterMaterial() );
            m_meta->add(kfmt, isub, "isur", boundary->getInnerSurface() );
            m_meta->add(kfmt, isub, "osur", boundary->getOuterSurface() );
            m_meta->add(kfmt, isub, "iext", boundary->getInnerExtra() );
            m_meta->add(kfmt, isub, "oext", boundary->getOuterExtra() );
        }

 
        for( unsigned int p = 0; p < NUM_QUAD ; ++p )  // over NUM_QUAD different sets (imat,omat,isur,osur,iext,oext)  
        { 
            GPropertyMap<float>* psrc = boundary->getConstituentByIndex(p) ; 
            addToIndex(psrc);  // this index includes only materials or surfaces used in boundaries, unlike the GGeo index
            std::string shortname = psrc->getShortName();

            if(psrc->isSkinSurface() || psrc->isBorderSurface())
            {
                GOpticalSurface* os = psrc->getOpticalSurface();
                assert(os && "all skin/boundary surface expected to have associated OpticalSurface");

                m_surfaces->add(shortname.c_str(), psrc->getIndex() );  // registering source indices (aiScene mat index) into GItemIndex
                unsigned int index_local = m_surfaces->getIndexLocal(shortname.c_str());  

                optical_data[opticalOffset + p*4 + optical_index]  =  index_local ; 
                optical_data[opticalOffset + p*4 + optical_type]   =  boost::lexical_cast<unsigned int>(os->getType()); 
                optical_data[opticalOffset + p*4 + optical_finish] =  boost::lexical_cast<unsigned int>(os->getFinish()); 
                optical_data[opticalOffset + p*4 + optical_value]  =  boost::lexical_cast<float>(os->getValue())*100.f ;   // express as integer percentage 
            } 
            else if(psrc->isMaterial())
            {
                m_materials->add(shortname.c_str(), psrc->getIndex() );  // registering source indices (aiScene mat index) into GItemIndex
                unsigned int index_local = m_materials->getIndexLocal(shortname.c_str());  

                optical_data[opticalOffset + p*4 + optical_index]  = index_local ;  
                optical_data[opticalOffset + p*4 + optical_type]   =  0 ;
                optical_data[opticalOffset + p*4 + optical_finish] =  0 ;
                optical_data[opticalOffset + p*4 + optical_value]  =  0 ;
            }
            else
            {
                optical_data[opticalOffset + p*4 + optical_index]  =  0 ;
                optical_data[opticalOffset + p*4 + optical_type]   =  0 ;
                optical_data[opticalOffset + p*4 + optical_finish] =  0 ;
                optical_data[opticalOffset + p*4 + optical_value]  =  0 ;
            }


            //printf("optical_data p %d : ", p); 
            //for(unsigned int i=0 ; i < 4 ; i++) printf(" %4d ", optical_data[opticalOffset + p*4 + i] );
            //printf("\n"); 


            GProperty<float> *p0,*p1,*p2,*p3 ; // 4 properties of the set 
            p0 = psrc->getPropertyByIndex(0);
            p1 = psrc->getPropertyByIndex(1);
            p2 = psrc->getPropertyByIndex(2);
            p3 = psrc->getPropertyByIndex(3);

            for( unsigned int d = 0; d < domainLength; ++d ) // interleave 4 properties into the buffer
            {   
                unsigned int dataOffset = ( p*domainLength + d )*4;  
                wavelength_data[subOffset+dataOffset+0] = p0->getValue(d) ;
                wavelength_data[subOffset+dataOffset+1] = p1->getValue(d) ;
                wavelength_data[subOffset+dataOffset+2] = p2->getValue(d) ;
                wavelength_data[subOffset+dataOffset+3] = p3->getValue(d) ;
            } 

            // record standard 4-property digest into metadata
            std::vector<GPropertyF*> props ; 
            props.push_back(p0);
            props.push_back(p1);
            props.push_back(p2);
            props.push_back(p3);
            std::string pdig = digestString(props);
          
            switch(p)
            {
               case 0:
                      m_meta->addDigest(kfmt, isub, "imat", pdig.c_str() ); 
                      m_meta->addMaterial(isub, "imat", ishortname.c_str(), pdig.c_str() );
                      break ;
               case 1:
                      m_meta->addDigest(kfmt, isub, "omat", pdig.c_str() ); 
                      m_meta->addMaterial(isub, "omat", oshortname.c_str(), pdig.c_str() );
                      break ;
               case 2:
                      m_meta->addDigest(kfmt, isub, "isur", pdig.c_str() ); 
                      break ;
               case 3:
                      m_meta->addDigest(kfmt, isub, "osur", pdig.c_str() ); 
                      break ;
               case 4:
                      m_meta->addDigest(kfmt, isub, "iext", pdig.c_str() ); 
                      break ;
               case 5:
                      m_meta->addDigest(kfmt, isub, "oext", pdig.c_str() ); 
                      break ;
            }
        }   
    }

    m_meta->createMaterialMap();


}


void GBoundaryLib::addToIndex(GPropertyMap<float>* psrc)
{
    unsigned int pindex = psrc->getIndex();
    if(pindex < UINT_MAX)
    {
         if(m_index.count(pindex) == 0) 
               m_index[pindex] = psrc->getShortName(); 
         else
               assert(strcmp(m_index[pindex].c_str(), psrc->getShortName()) == 0);
    }
}

void  GBoundaryLib::dumpIndex(const char* msg)
{
    printf("%s\n", msg);
    for(Index_t::iterator it=m_index.begin() ; it != m_index.end() ; it++)
         printf("  %3u :  %s \n", it->first, it->second.c_str() );
}

void GBoundaryLib::saveIndex(const char* dir, const char* filename)
{
   // HAVE SHIFTED TO GItemIndex for this 
   // saveMap<unsigned int, std::string>(m_index, dir, filename );
}




std::string GBoundaryLib::digestString(std::vector<GProperty<float>*>& props)
{
    return digest(props);
}
char* GBoundaryLib::digest(std::vector<GProperty<float>*>& props)
{
    MD5Digest dig ;            
    for(unsigned int i=0 ; i < props.size() ; ++i)
    {
        GProperty<float>* p = props[i]; 
        char* pdig = p->digest();
        dig.update(pdig, strlen(pdig));
        free(pdig);
    }
    return dig.finalize();
}


const char* GBoundaryLib::extraPropertyName(unsigned int i)
{
    assert(i < 4);
    //if(i == 0) return reemission_cdf ;
    if(i == 0) return extra_x ;
    if(i == 1) return extra_y ;
    if(i == 2) return extra_z ;
    if(i == 3) return extra_w ;
    return "?" ;
}

char* GBoundaryLib::propertyName(unsigned int p, unsigned int i)
{
    assert(p < NUM_QUAD && i < 4);
    char name[64];
    switch(p)
    {
       case 0: snprintf(name, 64, "%s%s", inner, GMaterialLib::propertyName(i) ); break;
       case 1: snprintf(name, 64, "%s%s", outer, GMaterialLib::propertyName(i) ); break;
       case 2: snprintf(name, 64, "%s%s", inner, GSurfaceLib::propertyName(i) ); break;
       case 3: snprintf(name, 64, "%s%s", outer, GSurfaceLib::propertyName(i) ); break;
       case 4: snprintf(name, 64, "%s%s", inner, extraPropertyName(i) ); break;
       case 5: snprintf(name, 64, "%s%s", outer, extraPropertyName(i) ); break;
    }
    return strdup(name);
}

std::string GBoundaryLib::propertyNameString(unsigned int p, unsigned int i)
{
    return propertyName(p,i);
}

void GBoundaryLib::digestDebug(GBoundary* boundary, unsigned int isub)
{
    for(unsigned int i=0 ; i<4 ; i++)
    {
        GPropertyMap<float>* q = boundary->getConstituentByIndex(i);
        std::string qdig   = q->getPDigestString(0,4);
        std::string qdigm  = m_meta->getBoundaryQtyByIndex(isub, i, "digest");

        if(strcmp(qdigm.c_str(), qdig.c_str())!=0)
        {
            switch(i)
            {
                case 0:
                case 1:
                        printf("xmat %20s %s %s \n", q->getShortNameString().c_str(), qdig.c_str(), qdigm.c_str());
                        break ; 
                case 2:
                case 3:
                        printf("xsur %20s %s %s \n", "", qdig.c_str(), qdigm.c_str());
                        break ; 
            }
            GProperty<float>* prop ; 
            for(unsigned int p=0 ; p < 4 ; ++p)
            {
                prop = q->getPropertyByIndex(p);
                std::string pdig = prop->getDigestString();
                printf(" %u prop digest %s \n", p, pdig.c_str());
            }
         }
     } 
}




void GBoundaryLib::dumpWavelengthBuffer(int wline)
{
    dumpWavelengthBuffer(wline, getWavelengthBuffer(), getMetadata(), getNumBoundary(), getStandardDomainLength());  
}
void GBoundaryLib::dumpOpticalBuffer(int wline)
{
    dumpOpticalBuffer(wline, getOpticalBuffer(), getMetadata(), getNumBoundary() );  
}

void GBoundaryLib::dumpOpticalBuffer(int wline, GBuffer* buffer, GBoundaryLibMetadata* meta, unsigned int numBoundary)
{
    if(!buffer) return ;
    unsigned int* data = (unsigned int*)buffer->getPointer();
    unsigned int numElementsTotal = buffer->getNumElementsTotal();
    assert(numElementsTotal == numBoundary*NUM_QUAD*4);

    printf("GBoundaryLib::dumpOpticalBuffer wline %d numBoundary %u numQuad %u \n", wline, numBoundary, NUM_QUAD );

    for(unsigned int isub=0 ; isub < numBoundary ; ++isub )
    {
        unsigned int subOffset = NUM_QUAD*4*isub ;
        for(unsigned int p=0 ; p < NUM_QUAD ; ++p ) 
        {
             if(p==0) printf("\n");
             std::string pname = meta ? meta->getBoundaryQtyByIndex(isub, p, "name") : "" ; 
             unsigned int line = getLine(isub, p) ;

             bool wselect = ( wline == -1 ) ||  (wline == line  ) ;
             if(wselect)
             {
                 printf(" %3u | %3u/%3u | ", line,isub, p );
                 unsigned int offset = subOffset + ( p*4 ) ;
                 for(unsigned int l=0 ; l < 4 ; ++l )
                 {
                     printf(" %5d ", data[offset+l] ); 
                 }
                 printf("   %s \n", pname.c_str() );
             }
        }
    }
}


void GBoundaryLib::dumpWavelengthBuffer(int wline, GBuffer* buffer, GBoundaryLibMetadata* meta, unsigned int numBoundary, unsigned int domainLength)
{
    if(!buffer) return ;

    float* data = (float*)buffer->getPointer();
    unsigned int numElementsTotal = buffer->getNumElementsTotal();
    assert(numElementsTotal == numBoundary*NUM_QUAD*4*domainLength);
    GDomain<float>* domain = GBoundaryLib::getDefaultDomain();
    assert(domain->getLength() == domainLength);

    printf("GBoundaryLib::dumpWavelengthBuffer wline %d numSub %u domainLength %u numQuad %u \n", wline, numBoundary, domainLength, NUM_QUAD );

    for(unsigned int isub=0 ; isub < numBoundary ; ++isub )
    {
        unsigned int subOffset = domainLength*NUM_QUAD*4*isub ;
        for(unsigned int p=0 ; p < NUM_QUAD ; ++p ) 
        {
             std::string pname = meta ? meta->getBoundaryQtyByIndex(isub, p, "name") : "" ; 
             unsigned int line = getLine(isub, p) ;
             bool wselect = ( wline == -1 ) ||  (wline == line  ) ;
             if(wselect)
             {
                 printf("\n %3u | %3u/%3u %s \n", line,isub, p, pname.c_str());
                 unsigned int offset = subOffset + ( p*domainLength*4 ) ;
                 for(unsigned int l=0 ; l < 4 ; ++l )
                 {
                     for(unsigned int d=0 ; d < domainLength ; ++d )
                     {
                         if(d%5 == 0) printf(" %15.3f", data[offset+d*4+l] );  // too many numbers so display one in every 5
                     }
                     printf("\n");
                 }
             }
        }
    }
}












const char* GBoundaryLib::wavelength   = "wavelength" ;
const char* GBoundaryLib::reemission   = "reemission" ;
const char* GBoundaryLib::optical       = "optical" ;


void GBoundaryLib::nameConstituents(std::vector<std::string>& names)
{
    names.push_back(wavelength); 
    names.push_back(reemission); 
    names.push_back(optical); 
}

GBuffer* GBoundaryLib::getBuffer(const char* name)
{
    if(strcmp(name, wavelength) == 0)   return m_wavelength_buffer ; 
    if(strcmp(name, reemission) == 0)   return m_reemission_buffer ; 
    if(strcmp(name, optical) == 0)      return m_optical_buffer ; 
    return NULL ;
}

void GBoundaryLib::setBuffer(const char* name, GBuffer* buffer)
{
    if(strcmp(name, wavelength) == 0)   setWavelengthBuffer(buffer) ; 
    if(strcmp(name, reemission) == 0)   setReemissionBuffer(buffer) ; 
    if(strcmp(name, optical) == 0)      setOpticalBuffer(buffer) ; 
}

bool GBoundaryLib::isFloatBuffer(const char* name)
{
    return ( 
             strcmp( name, wavelength ) == 0  || 
             strcmp( name, reemission ) == 0  
           );
}

bool GBoundaryLib::isIntBuffer(const char* name)
{
    return false ;
}

bool GBoundaryLib::isUIntBuffer(const char* name)
{
    return 
           ( 
              strcmp( name, optical) == 0  
           );
}

void GBoundaryLib::saveBuffer(const char* path, const char* name, GBuffer* buffer)
{
    GCache* cache = GCache::getInstance();
    LOG(info) << "GBoundaryLib::saveBuffer "
              << cache->getRelativePath(path)  
              ;

    if(isFloatBuffer(name))     buffer->save<float>(path);
    else if(isIntBuffer(name))  buffer->save<int>(path);
    else if(isUIntBuffer(name)) buffer->save<unsigned int>(path);
    else 
       printf("GBoundaryLib::saveBuffer WARNING NOT saving uncharacterized buffer %s into %s \n", name, path );
}


void GBoundaryLib::loadBuffer(const char* path, const char* name)
{
    GBuffer* buffer(NULL); 
    if(isFloatBuffer(name))                    buffer = GBuffer::load<float>(path);
    else if(isIntBuffer(name))                 buffer = GBuffer::load<int>(path);
    else if(isUIntBuffer(name))                buffer = GBuffer::load<unsigned int>(path);
    else
        printf("GBoundaryLib::loadBuffer WARNING not loading %s from %s \n", name, path ); 

    if(buffer) setBuffer(name, buffer);
}



void GBoundaryLib::save(const char* dir)
{
    fs::path cachedir(dir);
    if(!fs::exists(cachedir))
    {
        if (fs::create_directory(cachedir))
        {
            printf("GBoundaryLib::save created directory %s \n", dir );
        }
    }

    if(fs::exists(cachedir) && fs::is_directory(cachedir))
    {
        for(unsigned int i=0 ; i<m_names.size() ; i++)
        {
            std::string name = m_names[i];
            fs::path bufpath(dir);
            bufpath /= name + ".npy" ; 
            GBuffer* buffer = getBuffer(name.c_str());
            if(!buffer)
            {
                LOG(warning) << "GBoundaryLib::save skipping NULL buffer " << name ; 
                continue ; 
            }
            saveBuffer(bufpath.string().c_str(), name.c_str(), buffer);  
        } 
    }
    else
    {
        printf("GBoundaryLib::save directory %s DOES NOT EXIST \n", dir);
    }
}


void GBoundaryLib::loadBuffers(const char* dir)
{
    for(unsigned int i=0 ; i<m_names.size() ; i++)
    {
        std::string name = m_names[i];
        fs::path bufpath(dir);
        bufpath /= name + ".npy" ; 

        if(fs::exists(bufpath) && fs::is_regular_file(bufpath))
        { 
            loadBuffer(bufpath.string().c_str(), name.c_str());
        }
    } 


}


GBoundaryLib* GBoundaryLib::load(GCache* cache)
{
    const char* dir = cache->getIdPath() ;

    GBoundaryLib* lib(NULL);
    fs::path cachedir(dir);
    if(!fs::exists(cachedir))
    {
        printf("GBoundaryLib::load directory %s DOES NOT EXIST \n", dir);
    }
    else
    {
        lib = new GBoundaryLib(cache) ;
        GBoundaryLibMetadata* meta = GBoundaryLibMetadata::load(dir);
        lib->setMetadata(meta);  // buffer loading needs meta, so this has to come first
        lib->loadBuffers(dir);
    }
    return lib ; 
}



void GBoundaryLib::setWavelengthBuffer(GBuffer* buffer)
{
    if(!buffer) return ;
    m_wavelength_buffer = buffer ; 

    float* data = (float*)buffer->getPointer();

    unsigned int numElementsTotal = buffer->getNumElementsTotal();

    GDomain<float>* domain = GBoundaryLib::getDefaultDomain();
    unsigned int domainLength = domain->getLength(); 
    unsigned int numProp = getNumProp();
    unsigned int numBoundary = numElementsTotal/(numProp*domainLength);

    LOG(info) << "GBoundaryLib::setWavelengthBuffer  numBoundary: " << numBoundary ;


    for(unsigned int isub=0 ; isub < numBoundary ; ++isub )
    {
        unsigned int subOffset = domainLength*numProp*isub ;
        GBoundary* boundary = loadBoundary(data + subOffset, isub); 
        //boundary->Summary("GBoundaryLib::loadWavelengthBuffer",1);

        //std::string key = boundary->pdigest(0,4);  
        std::string key = boundary->digest();  // name based identity  
        assert(m_registry.count(key) == 0); // there should be no digest duplicates in wavelengthBuffer

        boundary->setIndex(m_keys.size());
        m_keys.push_back(key);  // for simple ordering  
        m_registry[key] = boundary ; 
    }
}

GBoundary* GBoundaryLib::loadBoundary(float* subData, unsigned int isub)
{
    GBoundary* boundary = new GBoundary ; 
    GDomain<float>* domain = GBoundaryLib::getDefaultDomain();
    unsigned int domainLength = domain->getLength(); 

    std::string mdig = m_meta->getBoundaryQty(isub, "boundary", "digest");

    for(unsigned int p=0 ; p < NUM_QUAD ; ++p ) 
    {
         std::string mapName = m_meta->getBoundaryQtyByIndex(isub, p, "name");
         GPropertyMap<float>* pmap = new GPropertyMap<float>(mapName.c_str(), isub, "recon"); 

         float* pdata = subData + p*domainLength*4 ; 

         for(unsigned int l=0 ; l < 4 ; ++l ) // un-interleaving the 4 properties
         {
             float* v = new float[domainLength];
             for(unsigned int d=0 ; d < domainLength ; ++d ) v[d] = pdata[d*4+l];  

             std::string pname = propertyNameString(p, l);
             pmap->addProperty(pname.c_str(), v, domain->getValues(), domainLength );
             delete v;
         }

         switch(p)
         {
            case 0:boundary->setInnerMaterial(pmap);break;
            case 1:boundary->setOuterMaterial(pmap);break;
            case 2:boundary->setInnerSurface(pmap);break;
            case 3:boundary->setOuterSurface(pmap);break;
            case 4:boundary->setInnerExtra(pmap);break;
            case 5:boundary->setOuterExtra(pmap);break;
         }
     }

     std::string sdig = boundary->getPDigestString(0,4);

     if(strcmp(sdig.c_str(), mdig.c_str()) != 0)
     {
         printf("GBoundaryLib::loadBoundary digest mismatch %u : %s %s \n", isub, sdig.c_str(), mdig.c_str());
         digestDebug(boundary, isub);
     }
     //assert(strcmp(sdig.c_str(), mdig.c_str()) == 0); 
     return boundary ; 
}


