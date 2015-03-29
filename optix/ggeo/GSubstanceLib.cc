#include "GSubstanceLib.hh"
#include "GSubstance.hh"
#include "GPropertyMap.hh"
#include "GEnums.hh"

#include <sstream>
#include "assert.h"
#include "stdio.h"
#include "limits.h"

GSubstanceLib::GSubstanceLib() : m_defaults(NULL) 
{
    // chroma/chroma/geometry.py
    // standard_wavelengths = np.arange(60, 810, 20).astype(np.float32)

    setKeyMap(NULL);

    GDomain<double>* domain = new GDomain<double>(60.f, 810.f, 20.f );
    setStandardDomain( domain );


    GPropertyMap* defaults = new GPropertyMap("defaults", UINT_MAX, "defaults");
    defaults->setStandardDomain(getStandardDomain());

    // TODO: find way to avoid detector specific keys and default values being hardcoded ? 
    // or at least move up to a higher level of the code 

    defineDefaults(defaults);
    setDefaults(defaults);

    m_ramp = GProperty<double>::ramp( domain->getLow(), domain->getStep(), domain->getValues(), domain->getLength() );
    //m_ramp->Summary("GSubstanceLib::GSubstanceLib ramp", 20 );

}

GSubstanceLib::~GSubstanceLib()
{
}

unsigned int GSubstanceLib::getNumSubstances()
{
   return m_keys.size();
}

GProperty<double>* GSubstanceLib::getRamp()
{
   return m_ramp ;
}

void GSubstanceLib::setDefaults(GPropertyMap* defaults)
{
    m_defaults = defaults ;
}
GPropertyMap* GSubstanceLib::getDefaults()
{
    return m_defaults ;
}

void GSubstanceLib::setStandardDomain(GDomain<double>* standard_domain)
{
    m_standard_domain = standard_domain ; 
}
GDomain<double>* GSubstanceLib::getStandardDomain()
{
    return m_standard_domain ;
}


GProperty<double>* GSubstanceLib::getDefaultProperty(const char* name)
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
        assert(substance->getIndex() == index );
    }
    return substance ; 
}


void GSubstanceLib::Summary(const char* msg)
{
    printf("%s\n", msg );
    char buf[128];
    for(unsigned int i=0 ; i < getNumSubstances() ; i++)
    {
         GSubstance* substance = getSubstance(i);
         snprintf(buf, 128, "%s substance ", msg );
         substance->Summary(buf);
    } 
}



GSubstance* GSubstanceLib::get(GPropertyMap* imaterial, GPropertyMap* omaterial, GPropertyMap* isurface, GPropertyMap* osurface )
{ 
    //printf("GSubstanceLib::get imaterial %p omaterial %p isurface %p osurface %p \n", imaterial, omaterial, isurface, osurface );

    GSubstance* tmp = new GSubstance(imaterial, omaterial, isurface, osurface);
    std::string key = tmp->digest();

    if(m_registry.count(key) == 0) // not yet registered, identity based on the digest 
    { 
        tmp->setIndex(m_keys.size());
        m_keys.push_back(key);  // for simple ordering  
        m_registry[key] = tmp ; 
    }
    else
    {
        delete tmp ; 
    } 

    GSubstance* substance = m_registry[key] ;
    //printf("GSubstanceLib::get key %s index %u \n", key.c_str(), substance->getIndex()); 
    return substance ; 
}




const char* GSubstanceLib::inner = "inner_" ;
const char* GSubstanceLib::outer = "outer_" ;

const char* GSubstanceLib::refractive_index = "refractive_index" ;
const char* GSubstanceLib::absorption_length = "absorption_length" ;
const char* GSubstanceLib::scattering_length = "scattering_length" ;
const char* GSubstanceLib::reemission_prob = "reemission_prob" ;

const char* GSubstanceLib::detect = "detect" ;
const char* GSubstanceLib::absorb = "absorb" ;
const char* GSubstanceLib::reflect_specular = "reflect_specular" ;
const char* GSubstanceLib::reflect_diffuse  = "reflect_diffuse" ;



const char* GSubstanceLib::keymap = 
"refractive_index:RINDEX,"
"absorption_length:ABSLENGTH,"
"scattering_length:RAYLEIGH,"
"reemission_prob:REEMISSIONPROB," 
"detect:EFFICIENCY," 
"absorb:DUMMY," 
"reflect_specular:REFLECTIVITY," 
"reflect_diffuse:REFLECTIVITY," 
;
// hmm how to convey specular/diffuse


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

const char* GSubstanceLib::getLocalKey(const char* dkey) // mapping between standard keynames and local key names
{
    return m_keymap[dkey].c_str();
}




GPropertyMap* GSubstanceLib::createStandardProperties(const char* pname, GSubstance* substance)
{
    GPropertyMap* ptex = new GPropertyMap(pname);

    addMaterialProperties(ptex, substance->getInnerMaterial(), inner);
    addMaterialProperties(ptex, substance->getOuterMaterial(), outer);
    addSurfaceProperties( ptex, substance->getInnerSurface(),  inner);
    addSurfaceProperties( ptex, substance->getOuterSurface(),  outer);

    checkMaterialProperties(ptex,  0 , inner);
    checkMaterialProperties(ptex,  4 , outer);
    checkSurfaceProperties( ptex,  8 , inner);
    checkSurfaceProperties( ptex, 12 , outer);

    return ptex ; 
}


void GSubstanceLib::defineDefaults(GPropertyMap* defaults)
{

    defaults->addConstantProperty( refractive_index,      1.f  );
    defaults->addConstantProperty( absorption_length,     1e6  );
    defaults->addConstantProperty( scattering_length,     1e6  );
    defaults->addConstantProperty( reemission_prob,       0.f  );

    defaults->addConstantProperty( detect          ,       0.f );
    defaults->addConstantProperty( absorb          ,       0.f );
    defaults->addConstantProperty( reflect_specular,       0.f );
    defaults->addConstantProperty( reflect_diffuse ,       0.f );
}

void GSubstanceLib::addMaterialProperties(GPropertyMap* ptex, GPropertyMap* pmap, const char* prefix)
{
    assert(pmap);  // materials must always be defined
    assert(pmap->isMaterial());
    assert(getStandardDomain()->isEqual(pmap->getStandardDomain()));

    if(ptex->hasStandardDomain())
    {
        assert(ptex->getStandardDomain()->isEqual(pmap->getStandardDomain()));
    } 
    else
    {
        ptex->setStandardDomain(pmap->getStandardDomain());
    }

    ptex->addProperty(refractive_index, getPropertyOrDefault( pmap, refractive_index ), prefix);
    ptex->addProperty(absorption_length,getPropertyOrDefault( pmap, absorption_length ), prefix);
    ptex->addProperty(scattering_length,getPropertyOrDefault( pmap, scattering_length ), prefix);
    ptex->addProperty(reemission_prob  ,getPropertyOrDefault( pmap, reemission_prob ), prefix);
}

void GSubstanceLib::addSurfaceProperties(GPropertyMap* ptex, GPropertyMap* pmap, const char* prefix)
{
    if(pmap) // surfaces often not defined
    { 
        assert(pmap->isSkinSurface() || pmap->isBorderSurface());
        assert(getStandardDomain()->isEqual(pmap->getStandardDomain()));
    }

    ptex->addProperty(detect,           getPropertyOrDefault( pmap, detect ), prefix);
    ptex->addProperty(absorb,           getPropertyOrDefault( pmap, absorb ), prefix);
    ptex->addProperty(reflect_specular, getPropertyOrDefault( pmap, reflect_specular ), prefix);
    ptex->addProperty(reflect_diffuse,  getPropertyOrDefault( pmap, reflect_diffuse  ), prefix);
}


GPropertyD* GSubstanceLib::getPropertyOrDefault(GPropertyMap* pmap, const char* dkey)
{
    const char* lkey = getLocalKey(dkey); assert(lkey);  // missing local key mapping 

    GPropertyD* fallback = getDefaultProperty(dkey);  assert(fallback);

    GPropertyD* prop = pmap ? pmap->getProperty(lkey) : NULL ;

    return prop ? prop : fallback ;
}

void GSubstanceLib::checkMaterialProperties(GPropertyMap* ptex, unsigned int offset, const char* _prefix)
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

void GSubstanceLib::checkSurfaceProperties(GPropertyMap* ptex, unsigned int offset, const char* _prefix)
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


