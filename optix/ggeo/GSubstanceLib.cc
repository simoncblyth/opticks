#include "GSubstanceLib.hh"
#include "GSubstanceLibMetadata.hh"
#include "GSubstance.hh"
#include "GPropertyMap.hh"
#include "GBuffer.hh"
#include "GEnums.hh"
#include "md5digest.hh"

#include <string>
#include <map>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal




#include <sstream>
#include "assert.h"
#include "stdio.h"
#include "limits.h"
#include "string.h"

unsigned int GSubstanceLib::NUM_QUAD    = 6  ; 
float        GSubstanceLib::DOMAIN_LOW  = 60.f ; 
float        GSubstanceLib::DOMAIN_HIGH = 810.f ; 
float        GSubstanceLib::DOMAIN_STEP = 20.f ; 
unsigned int GSubstanceLib::DOMAIN_LENGTH = 39  ; 
float        GSubstanceLib::SURFACE_UNSET = -1.f ; 
float        GSubstanceLib::EXTRA_UNSET = -1.f ; 

const char* GSubstanceLib::inner = "inner_" ;
const char* GSubstanceLib::outer = "outer_" ;

// NB below strings should match names in GEnums.hh
// material
const char* GSubstanceLib::refractive_index  = "refractive_index" ;
const char* GSubstanceLib::absorption_length = "absorption_length" ;
const char* GSubstanceLib::scattering_length = "scattering_length" ;
const char* GSubstanceLib::reemission_prob   = "reemission_prob" ;

// surface
const char* GSubstanceLib::detect            = "detect" ;
const char* GSubstanceLib::absorb            = "absorb" ;
const char* GSubstanceLib::reflect_specular  = "reflect_specular" ;
const char* GSubstanceLib::reflect_diffuse   = "reflect_diffuse" ;

// extra
const char* GSubstanceLib::reemission_cdf    = "reemission_cdf" ; // NOT USED : AS NEEDED MORE BINS FOR inverted CDF SO USED SEPARATE reemission_buffer
const char* GSubstanceLib::extra_x           = "extra_x" ;
const char* GSubstanceLib::extra_y           = "extra_y" ;
const char* GSubstanceLib::extra_z           = "extra_z" ;
const char* GSubstanceLib::extra_w           = "extra_w" ;

// workings for "extra"
const char* GSubstanceLib::slow_component    = "slow_component" ;
const char* GSubstanceLib::fast_component    = "fast_component" ;

const char* GSubstanceLib::keymap = 
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

//  ggeo-meta 102 103 104 105 106 107 
//
//  need to branch REFLECTIVITY depending on *finish* property from GOpticalSurface
//
//     reflect_specular   finish = 0 (polished)
//     reflect_diffuse    finish = 3 (ground)
//        
//


const char* GSubstanceLib::scintillators = "LiquidScintillator,GdDopedLS" ;
const char* GSubstanceLib::reemissionkey = "SLOWCOMPONENT,FASTCOMPONENT" ;

std::vector<std::string>* GSubstanceLib::vscintillators = NULL ;
std::vector<std::string>* GSubstanceLib::vreemissionkey = NULL ;



bool GSubstanceLib::isReemissionKey(std::string& lkey)
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

bool GSubstanceLib::isScintillator(std::string& matShortName)
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

void GSubstanceLib::setKeyMap(const char* spec)
{
    m_keymap.clear();
    const char* kmap = spec ? spec : keymap ; 

    char delim = ',' ;
    std::istringstream f(kmap);
    std::string s;
    while (getline(f, s, delim)) 
    {
        std::size_t colon = s.find(":");
        if(colon == std::string::npos)
        {
            printf("GSubstanceLib::setKeyMap SKIPPING ENTRY WITHOUT COLON %s\n", s.c_str());
            continue ;
        }
        
        std::string dk = s.substr(0, colon);
        std::string lk = s.substr(colon+1);
        //printf("GSubstanceLib::setKeyMap dk [%s] lk [%s] \n", dk.c_str(), lk.c_str());
        m_keymap[dk] = lk ; 
    }
}



GSubstanceLib::GSubstanceLib() 
          : 
          m_defaults(NULL), 
          m_meta(NULL), 
          m_standard(true), 
          m_num_quad(6), 
          m_wavelength_buffer(NULL),
          m_reemission_buffer(NULL)
{
    setKeyMap(NULL);
    GDomain<float>* domain = new GDomain<float>(DOMAIN_LOW, DOMAIN_HIGH, DOMAIN_STEP ); 
    setStandardDomain( domain );

    GPropertyMap<float>* defaults = new GPropertyMap<float>("defaults", UINT_MAX, "defaults");
    defaults->setStandardDomain(getStandardDomain());

    defineDefaults(defaults);
    setDefaults(defaults);

    m_ramp = GProperty<float>::ramp( domain->getLow(), domain->getStep(), domain->getValues(), domain->getLength() );
}

GSubstanceLib::~GSubstanceLib()
{
}

unsigned int GSubstanceLib::getNumSubstances()
{
   return m_keys.size();
}

GDomain<float>* GSubstanceLib::getDefaultDomain()
{
   return new GDomain<float>(DOMAIN_LOW, DOMAIN_HIGH, DOMAIN_STEP ); 
}

void GSubstanceLib::setStandardDomain(GDomain<float>* standard_domain)
{
    m_standard_domain = standard_domain ; 

    assert(getStandardDomainLength() == DOMAIN_LENGTH );
    assert(m_standard_domain->getLow()  == DOMAIN_LOW );
    assert(m_standard_domain->getHigh() == DOMAIN_HIGH );
    assert(m_standard_domain->getStep() == DOMAIN_STEP );
}

unsigned int GSubstanceLib::getStandardDomainLength()
{
    return m_standard_domain ? m_standard_domain->getLength() : 0 ;
}

GProperty<float>* GSubstanceLib::getDefaultProperty(const char* name)
{
    return m_defaults ? m_defaults->getProperty(name) : NULL ;
}

GSubstance* GSubstanceLib::getSubstance(unsigned int index)
{
    GSubstance* substance = NULL ;
    if(index < m_keys.size())
    {
        std::string key = m_keys[index] ;  
        substance = m_registry[key];
        if(substance->getIndex() != index )
        {
            printf("GSubstanceLib::getSubstance WARNING substance index mismatch request %u substance %u key %s \n", index, substance->getIndex(), key.c_str() ); 
        } 
        assert(substance->getIndex() == index );
    }
    return substance ; 
}


void GSubstanceLib::Summary(const char* msg)
{
    printf("%s\n", msg );
    char buf[128];
    for(unsigned int isub=0 ; isub < getNumSubstances() ; isub++)
    {
         GSubstance* substance = getSubstance(isub);

         unsigned int lineMin = getLine(isub, 0);
         unsigned int lineMax = getLine(isub, NUM_QUAD-1);
         snprintf(buf, 128, "%s lineMin/Max %3u:%3u ", msg, lineMin, lineMax );
         substance->Summary(buf);
    } 
}

GSubstance* GSubstanceLib::get(
           GPropertyMap<float>* imaterial, 
           GPropertyMap<float>* omaterial, 
           GPropertyMap<float>* isurface, 
           GPropertyMap<float>* osurface,
           GPropertyMap<float>* iextra,
           GPropertyMap<float>* oextra,
           GOpticalSurface* inner_optical,
           GOpticalSurface* outer_optical
      )
{ 
    // this "get" pulls the GSubstance into existance and populates the registry
    //printf("GSubstanceLib::get imaterial %p omaterial %p isurface %p osurface %p \n", imaterial, omaterial, isurface, osurface );

    GSubstance raw(imaterial, omaterial, isurface, osurface, iextra, oextra, inner_optical, outer_optical);
    GSubstance* standard = createStandardSubstance(&raw) ;
    std::string key = standard->pdigest(0,4);  // standard digest based identity 

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

    GSubstance* substance = m_registry[key] ;
    //printf("GSubstanceLib::get key %s index %u \n", key.c_str(), substance->getIndex()); 
    return substance ; 
}



GSubstance* GSubstanceLib::createStandardSubstance(GSubstance* substance)
{
    GPropertyMap<float>* imat = substance->getInnerMaterial();
    GPropertyMap<float>* omat = substance->getOuterMaterial();
    GPropertyMap<float>* isur = substance->getInnerSurface();
    GPropertyMap<float>* osur = substance->getOuterSurface();
    GPropertyMap<float>* iext = substance->getInnerExtra();
    GPropertyMap<float>* oext = substance->getOuterExtra();

    // snag extra props from corresponding materials pmaps : HUH thats unhealthy
    // if(iext == NULL) iext = imat ;
    // if(oext == NULL) oext = omat ;

    GPropertyMap<float>* s_imat = new GPropertyMap<float>(imat);
    GPropertyMap<float>* s_omat = new GPropertyMap<float>(omat);
    GPropertyMap<float>* s_isur = new GPropertyMap<float>(isur);
    GPropertyMap<float>* s_osur = new GPropertyMap<float>(osur);
    GPropertyMap<float>* s_iext = new GPropertyMap<float>(iext);
    GPropertyMap<float>* s_oext = new GPropertyMap<float>(oext);

    standardizeMaterialProperties( s_imat, imat, inner );
    standardizeMaterialProperties( s_omat, omat, outer );
    standardizeSurfaceProperties(  s_isur, isur, inner );
    standardizeSurfaceProperties(  s_osur, osur, outer );
    standardizeExtraProperties(    s_iext, iext, inner );
    standardizeExtraProperties(    s_oext, oext, outer );

    GSubstance* s_substance = new GSubstance( s_imat , s_omat, s_isur, s_osur, s_iext, s_oext);

    return s_substance ; 
}

const char* GSubstanceLib::getLocalKey(const char* dkey) // mapping between standard keynames and local key names, eg refractive_index -> RINDEX
{
    return m_keymap[dkey].c_str();
}



GPropertyMap<float>* GSubstanceLib::createStandardProperties(const char* pname, GSubstance* substance)
{
    // combining all 6 sets into one PropertyMap : for insertion into wavelengthBuffer

    GPropertyMap<float>* ptex = new GPropertyMap<float>(pname);
    
    ptex->add(substance->getInnerMaterial(), inner);
    ptex->add(substance->getOuterMaterial(), outer);
    ptex->add(substance->getInnerSurface(),  inner);
    ptex->add(substance->getOuterSurface(),  outer);
    ptex->add(substance->getInnerExtra(),    inner);
    ptex->add(substance->getOuterExtra(),    outer);

    checkMaterialProperties(ptex,  0*4 , inner);
    checkMaterialProperties(ptex,  1*4 , outer);
    checkSurfaceProperties( ptex,  2*4 , inner);
    checkSurfaceProperties( ptex,  3*4 , outer);
    checkExtraProperties(   ptex,  4*4 , inner);
    checkExtraProperties(   ptex,  5*4 , outer);

    return ptex ; 
}


void GSubstanceLib::defineDefaults(GPropertyMap<float>* defaults)
{
    defaults->addConstantProperty( refractive_index,      1.f  );
    defaults->addConstantProperty( absorption_length,     1e6  );
    defaults->addConstantProperty( scattering_length,     1e6  );
    defaults->addConstantProperty( reemission_prob,       0.f  );

    defaults->addConstantProperty( detect          ,      SURFACE_UNSET );
    defaults->addConstantProperty( absorb          ,      SURFACE_UNSET );
    defaults->addConstantProperty( reflect_specular,      SURFACE_UNSET );
    defaults->addConstantProperty( reflect_diffuse ,      SURFACE_UNSET );

    //defaults->addConstantProperty( reemission_cdf  ,      EXTRA_UNSET );
    defaults->addConstantProperty( extra_x         ,      EXTRA_UNSET );
    defaults->addConstantProperty( extra_y         ,      EXTRA_UNSET );
    defaults->addConstantProperty( extra_z         ,      EXTRA_UNSET );
    defaults->addConstantProperty( extra_w         ,      EXTRA_UNSET );

}

std::vector<std::string> GSubstanceLib::splitString(std::string keys)
{
    std::vector<std::string> vkeys;
    boost::split(vkeys, keys, boost::is_any_of(" "));
    return vkeys ; 
}


void GSubstanceLib::standardizeMaterialProperties(GPropertyMap<float>* pstd, GPropertyMap<float>* pmap, const char* prefix)
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




GProperty<float>* GSubstanceLib::constructInvertedReemissionCDF(GPropertyMap<float>* pmap)
{
    std::string name = pmap->getShortNameString();

    if(!isScintillator(name)) return NULL ;

    typedef GProperty<float> P ; 

    P* slow = getProperty(pmap, slow_component);
    P* fast = getProperty(pmap, fast_component);

    if( slow == NULL || fast == NULL)
    {
        LOG(warning) << "GSubstanceLib::constructInvertedReemissionCDF failed to find slow/fast for purported scintillator pmap: " << pmap->description() ;
        return NULL ; 
    }

    float mxdiff = GProperty<float>::maxdiff(slow, fast);
    assert(mxdiff < 1e-6 );

    P* rrd = slow->createReversedReciprocalDomain();    // have to used reciprocal "energywise" domain for G4/NuWa agreement

    P* srrd = rrd->createZeroTrimmed();                 // trim extraneous zero values, leaving at most one zero at either extremity

    assert( srrd->getLength() == rrd->getLength() - 2); // expect to trim 2 values

    P* rcdf = srrd->createCDF();

    unsigned int nicdf = 4096 ; //  minor discrep ~600nm

    //
    // Why does lookup "sampling" require so many more bins to get agreeable 
    // results than standard sampling ?
    //
    // * maybe because "agree" means it matches a prior standard sampling and in
    //   the limit of many bins the techniques converge ?
    //
    // * Nope, its because of the fixed width raster across entire 0:1 in 
    //   lookup compared to "effectively" variable raster when doing value binary search
    //   as opposed to domain jump-to-the-bin : see notes in tests/GPropertyTest.cc
    //

    P* icdf = rcdf->createInverseCDF(nicdf); 

    icdf->getValues()->reciprocate();  // avoid having to reciprocate lookup results : by doing it here 

    return icdf ; 
}

GProperty<float>* GSubstanceLib::constructReemissionCDF(GPropertyMap<float>* pmap)
{
    std::string name = pmap->getShortNameString();
    if(!isScintillator(name)) return getDefaultProperty( reemission_cdf ) ;

    // LiquidScintillator,GdDopedLS
    GProperty<float>* slow = getProperty(pmap, slow_component);
    GProperty<float>* fast = getProperty(pmap, fast_component);

    if(slow == NULL || fast == NULL )
    {
        LOG(warning)<<"GSubstanceLib::constructReemissionCDF failed to find slow/fast for pmap: " << pmap->description() ;
        return getDefaultProperty( reemission_cdf ) ;
    } 


    float mxdiff = GProperty<float>::maxdiff(slow, fast);
    //printf("mxdiff pslow-pfast *1e6 %10.4f \n", mxdiff*1e6 );
    assert(mxdiff < 1e-6 );

    GProperty<float>* rrd = slow->createReversedReciprocalDomain();
    GProperty<float>* cdf = rrd->createCDF();
    delete rrd ; 
    return cdf ;
}


void GSubstanceLib::standardizeExtraProperties(GPropertyMap<float>* pstd, GPropertyMap<float>* pmap, const char* prefix)
{
    //pstd->addProperty(reemission_cdf   , constructReemissionCDF( pmap )              , prefix);
    pstd->addProperty(extra_x          , getPropertyOrDefault( pmap, extra_x )       , prefix);
    pstd->addProperty(extra_y          , getPropertyOrDefault( pmap, extra_y )       , prefix);
    pstd->addProperty(extra_z          , getPropertyOrDefault( pmap, extra_z )       , prefix);
    pstd->addProperty(extra_w          , getPropertyOrDefault( pmap, extra_w )       , prefix);
}



void GSubstanceLib::standardizeSurfaceProperties(GPropertyMap<float>* pstd, GPropertyMap<float>* pmap, const char* prefix)
{
    if(pmap) // surfaces often not defined
    { 
        assert(pmap->isSkinSurface() || pmap->isBorderSurface());
        assert(getStandardDomain()->isEqual(pmap->getStandardDomain()));
    }

    // hmm using UNSET values, means that need to zero others ?

    pstd->addProperty(detect,           getPropertyOrDefault( pmap, detect ), prefix);
    pstd->addProperty(absorb,           getPropertyOrDefault( pmap, absorb ), prefix);
    pstd->addProperty(reflect_specular, getPropertyOrDefault( pmap, reflect_specular ), prefix);
    pstd->addProperty(reflect_diffuse,  getPropertyOrDefault( pmap, reflect_diffuse  ), prefix);
}

GProperty<float>* GSubstanceLib::getProperty(GPropertyMap<float>* pmap, const char* dkey)
{
    assert(pmap);

    const char* lkey = getLocalKey(dkey); assert(lkey);  // missing local key mapping 

    GProperty<float>* prop = pmap->getProperty(lkey) ;

    //assert(prop);
    if(!prop)
    {
        LOG(warning) << "GSubstanceLib::getProperty failed to find property " << dkey << "/" << lkey ;
    }

    return prop ;  
}



GProperty<float>* GSubstanceLib::getPropertyOrDefault(GPropertyMap<float>* pmap, const char* dkey)
{
    const char* lkey = getLocalKey(dkey); assert(lkey);  // missing local key mapping 

    GProperty<float>* fallback = getDefaultProperty(dkey);  assert(fallback);

    GProperty<float>* prop = pmap ? pmap->getProperty(lkey) : NULL ;

    return prop ? prop : fallback ;
}



void GSubstanceLib::checkMaterialProperties(GPropertyMap<float>* ptex, unsigned int offset, const char* _prefix)
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
void GSubstanceLib::checkSurfaceProperties(GPropertyMap<float>* ptex, unsigned int offset, const char* _prefix)
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

void GSubstanceLib::checkExtraProperties(GPropertyMap<float>* ptex, unsigned int offset, const char* _prefix)
{
    std::string prefix = _prefix ; 
    std::string xname, pname ;

    xname = prefix + reemission_cdf ; 
    pname = ptex->getPropertyNameByIndex(offset+e_reemission_cdf);
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





const char* GSubstanceLib::getDigest(unsigned int index)
{
    return m_keys[index].c_str();
}

unsigned int GSubstanceLib::getLine(unsigned int isub, unsigned int ioff)
{
    assert(ioff < NUM_QUAD);
    return isub*NUM_QUAD + ioff ;   
}

GBuffer* GSubstanceLib::createReemissionBuffer(GPropertyMap<float>* scint)
{
    assert(scint);

    // TODO: reposition the .npy inside idpath
    // for comparison only
    GProperty<float>* cdf = constructReemissionCDF(scint);
    if(cdf)
    {
        cdf->save("/tmp/reemissionCDF.npy");
    }


    GProperty<float>* icdf = constructInvertedReemissionCDF(scint);
    if(icdf == NULL)
    {
        LOG(warning)<<"GSubstanceLib::createReemissionBuffer FAILED as no icdf from constructInvertedReemissionCDF for scint " << scint->description() ;
        return NULL ; 
    }

     
    icdf->Summary("GSubstanceLib::createReemissionBuffer icdf ", 256);
    {
        icdf->save("/tmp/invertedReemissionCDF.npy");
        GAry<float>* insitu = icdf->lookupCDF(1e6);
        insitu->Summary("icdf->lookupCDF(1e6)");
        insitu->save("/tmp/insitu.npy");
    }    

    unsigned int numFloat = icdf->getLength();
    LOG(info) << "GSubstanceLib::createReemissionBuffer numFloat " << numFloat ;  

    GBuffer* buffer = new GBuffer( sizeof(float)*numFloat, new float[numFloat], sizeof(float), 1 );
    float* data = (float*)buffer->getPointer();
    for( unsigned int d = 0; d < numFloat ; ++d ) data[d] = icdf->getValue(d);

    return buffer ; 
}



GBuffer* GSubstanceLib::createWavelengthBuffer()
{
    //
    //  6 sets of 4 props  
    //
    //  The 6 sets for: 
    //          (inner material, outer material, inner surface, outer surface, inner extra, outer extra) 
    //
    //  and the props:
    //         material: (refractive_index, absorption_length, scattering_length, reemission_prob)
    //         surface:  (detect, absorb, reflect_specular, reflect_diffuse)    
    //         extra:    (reemission_cdf, extra_y, extra_z, extra_w )
    //

    GBuffer* buffer(NULL) ;
    float* data(NULL) ;

    if(!m_meta) m_meta = new GSubstanceLibMetadata ; 

    GDomain<float>* domain = getStandardDomain();
    const unsigned int domainLength = domain->getLength();
    unsigned int numSubstance = getNumSubstances() ;
    unsigned int numFloat = domainLength*NUM_QUAD*4*numSubstance ; 

    for(unsigned int isub=0 ; isub < numSubstance ; isub++)
    {
        GSubstance* substance = getSubstance(isub);
        unsigned int substanceIndex = substance->getIndex();

        //printf("GSubstanceLib::createWavelengthBuffer isub %u/%u substanceIndex  %u \n", isub, numSubstance, substanceIndex );
        assert(substanceIndex < numSubstance);
        assert(isub == substanceIndex);

        const char* dig = getDigest(isub);
        {
            char* ckdig = substance->pdigest(0,4);
            assert(strcmp(ckdig, dig) == 0);
        }

        unsigned int subOffset = domainLength*NUM_QUAD*4*substanceIndex ; 
        const char* kfmt = "lib.substance.%d.%s.%s" ;
        m_meta->addDigest(kfmt, isub, "substance", (char*)dig ); 

        std::string ishortname = substance->getInnerMaterial()->getShortNameString("__dd__Materials__") ; 
        std::string oshortname = substance->getOuterMaterial()->getShortNameString("__dd__Materials__") ; 
        {
            m_meta->add(kfmt, isub, "imat", substance->getInnerMaterial() );
            m_meta->add(kfmt, isub, "omat", substance->getOuterMaterial() );
            m_meta->add(kfmt, isub, "isur", substance->getInnerSurface() );
            m_meta->add(kfmt, isub, "osur", substance->getOuterSurface() );
            m_meta->add(kfmt, isub, "iext", substance->getInnerExtra() );
            m_meta->add(kfmt, isub, "oext", substance->getOuterExtra() );
        }


        if( buffer == NULL )
        {
            buffer = new GBuffer( sizeof(float)*numFloat, new float[numFloat], sizeof(float), 1 );
            data = (float*)buffer->getPointer();
        }


        //
        // * maybe add a fifth set "iore" for reemission
        // * actually cleaner and to add two extra sets: iext, oext
        //   so reemission_cdf will occupy 1 of the 4 slots leaving 3 spares
        //
 
        for( unsigned int p = 0; p < NUM_QUAD ; ++p )  // over NUM_QUAD different sets (imat,omat,isur,osur,iext,oext)  
        { 
            GPropertyMap<float>* psrc = substance->getConstituentByIndex(p) ; 

            GProperty<float> *p0,*p1,*p2,*p3 ; // 4 properties of the set 
            p0 = psrc->getPropertyByIndex(0);
            p1 = psrc->getPropertyByIndex(1);
            p2 = psrc->getPropertyByIndex(2);
            p3 = psrc->getPropertyByIndex(3);

            for( unsigned int d = 0; d < domainLength; ++d ) // interleave 4 properties into the buffer
            {   
                unsigned int dataOffset = ( p*domainLength + d )*4;  
                data[subOffset+dataOffset+0] = p0->getValue(d) ;
                data[subOffset+dataOffset+1] = p1->getValue(d) ;
                data[subOffset+dataOffset+2] = p2->getValue(d) ;
                data[subOffset+dataOffset+3] = p3->getValue(d) ;
            } 

            // record standard 4-property digest into metadata
            std::vector<GPropertyF*> props ; 
            props.push_back(p0);
            props.push_back(p1);
            props.push_back(p2);
            props.push_back(p3);
            std::string pdig = digestString(props);
            {
               std::string ckdig = psrc->getPDigestString(0,4);
               assert(strcmp(pdig.c_str(), ckdig.c_str())==0);
            }

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

    return buffer ; 
}


std::string GSubstanceLib::digestString(std::vector<GProperty<float>*>& props)
{
    return digest(props);
}
char* GSubstanceLib::digest(std::vector<GProperty<float>*>& props)
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


const char* GSubstanceLib::materialPropertyName(unsigned int i)
{
    assert(i < 4);
    if(i == 0) return refractive_index ;
    if(i == 1) return absorption_length ;
    if(i == 2) return scattering_length ;
    if(i == 3) return reemission_prob ;
    return "?" ;
}

const char* GSubstanceLib::surfacePropertyName(unsigned int i)
{
    assert(i < 4);
    if(i == 0) return detect ;
    if(i == 1) return absorb ;
    if(i == 2) return reflect_specular;
    if(i == 3) return reflect_diffuse ;
    return "?" ;
}
const char* GSubstanceLib::extraPropertyName(unsigned int i)
{
    assert(i < 4);
    if(i == 0) return reemission_cdf ;
    if(i == 1) return extra_y ;
    if(i == 2) return extra_z ;
    if(i == 3) return extra_w ;
    return "?" ;
}

char* GSubstanceLib::propertyName(unsigned int p, unsigned int i)
{
    assert(p < NUM_QUAD && i < 4);
    char name[64];
    switch(p)
    {
       case 0: snprintf(name, 64, "%s%s", inner, materialPropertyName(i) ); break;
       case 1: snprintf(name, 64, "%s%s", outer, materialPropertyName(i) ); break;
       case 2: snprintf(name, 64, "%s%s", inner, surfacePropertyName(i) ); break;
       case 3: snprintf(name, 64, "%s%s", outer, surfacePropertyName(i) ); break;
       case 4: snprintf(name, 64, "%s%s", inner, extraPropertyName(i) ); break;
       case 5: snprintf(name, 64, "%s%s", outer, extraPropertyName(i) ); break;
    }
    return strdup(name);
}

std::string GSubstanceLib::propertyNameString(unsigned int p, unsigned int i)
{
    return propertyName(p,i);
}

GSubstance* GSubstanceLib::loadSubstance(float* subData, unsigned int isub)
{
    GSubstance* substance = new GSubstance ; 
    GDomain<float>* domain = GSubstanceLib::getDefaultDomain();
    unsigned int domainLength = domain->getLength(); 

    std::string mdig = m_meta->getSubstanceQty(isub, "substance", "digest");

    for(unsigned int p=0 ; p < NUM_QUAD ; ++p ) 
    {
         std::string mapName = m_meta->getSubstanceQtyByIndex(isub, p, "name");
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
            case 0:substance->setInnerMaterial(pmap);break;
            case 1:substance->setOuterMaterial(pmap);break;
            case 2:substance->setInnerSurface(pmap);break;
            case 3:substance->setOuterSurface(pmap);break;
            case 4:substance->setInnerExtra(pmap);break;
            case 5:substance->setOuterExtra(pmap);break;
         }
     }

     std::string sdig = substance->getPDigestString(0,4);

     if(strcmp(sdig.c_str(), mdig.c_str()) != 0)
     {
         printf("GSubstanceLib::loadSubstance digest mismatch %u : %s %s \n", isub, sdig.c_str(), mdig.c_str());
         digestDebug(substance, isub);
     }
     //assert(strcmp(sdig.c_str(), mdig.c_str()) == 0); 
     return substance ; 
}

void GSubstanceLib::digestDebug(GSubstance* substance, unsigned int isub)
{
    for(unsigned int i=0 ; i<4 ; i++)
    {
        GPropertyMap<float>* q = substance->getConstituentByIndex(i);
        std::string qdig   = q->getPDigestString(0,4);
        std::string qdigm  = m_meta->getSubstanceQtyByIndex(isub, i, "digest");

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




GSubstanceLib* GSubstanceLib::load(const char* dir)
{
    GSubstanceLib* lib = new GSubstanceLib();

    GSubstanceLibMetadata* meta = GSubstanceLibMetadata::load(dir);
    lib->setMetadata(meta); 

    GBuffer* buffer = GBuffer::load<float>(dir, "wavelength.npy");
    buffer->Summary("wavelength buffer");

    lib->loadWavelengthBuffer(buffer);
    lib->setWavelengthBuffer(buffer);

    return lib ; 
}


void GSubstanceLib::loadWavelengthBuffer(GBuffer* buffer)
{
    if(!buffer) return ;

    float* data = (float*)buffer->getPointer();

    unsigned int numElementsTotal = buffer->getNumElementsTotal();

    GDomain<float>* domain = GSubstanceLib::getDefaultDomain();
    unsigned int domainLength = domain->getLength(); 
    unsigned int numProp = getNumProp();
    unsigned int numSubstance = numElementsTotal/(numProp*domainLength);
    //assert(numSubstance == 54);
    if(numSubstance != 54)
    {
        LOG(warning) << "GSubstanceLib::loadWavelengthBuffer didnt see 54, numSubstance: " << numSubstance ; 
    }

    for(unsigned int isub=0 ; isub < numSubstance ; ++isub )
    {
        unsigned int subOffset = domainLength*numProp*isub ;
        GSubstance* substance = loadSubstance(data + subOffset, isub); 
        //substance->Summary("GSubstanceLib::loadWavelengthBuffer",1);

        std::string key = substance->pdigest(0,4);  
        assert(m_registry.count(key) == 0); // there should be no digest duplicates in wavelengthBuffer

        substance->setIndex(m_keys.size());
        m_keys.push_back(key);  // for simple ordering  
        m_registry[key] = substance ; 
        
    }
}

void GSubstanceLib::dumpWavelengthBuffer(int wline)
{
    dumpWavelengthBuffer(wline, getWavelengthBuffer(), getMetadata(), getNumSubstances(), getStandardDomainLength());  
}

void GSubstanceLib::dumpWavelengthBuffer(int wline, GBuffer* buffer, GSubstanceLibMetadata* meta, unsigned int numSubstance, unsigned int domainLength)
{
    if(!buffer) return ;

    float* data = (float*)buffer->getPointer();
    unsigned int numElementsTotal = buffer->getNumElementsTotal();
    assert(numElementsTotal == numSubstance*NUM_QUAD*4*domainLength);
    GDomain<float>* domain = GSubstanceLib::getDefaultDomain();
    assert(domain->getLength() == domainLength);

    printf("GSubstanceLib::dumpWavelengthBuffer wline %d numSub %u domainLength %u numQuad %u \n", wline, numSubstance, domainLength, NUM_QUAD );

    for(unsigned int isub=0 ; isub < numSubstance ; ++isub )
    {
        unsigned int subOffset = domainLength*NUM_QUAD*4*isub ;
        for(unsigned int p=0 ; p < NUM_QUAD ; ++p ) 
        {
             std::string pname = meta ? meta->getSubstanceQtyByIndex(isub, p, "name") : "" ; 
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




