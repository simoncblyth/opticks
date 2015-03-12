#include "GSubstanceLib.hh"
#include "GSubstance.hh"
#include "GPropertyMap.hh"

#include "assert.h"
#include "stdio.h"
#include "limits.h"

GSubstanceLib::GSubstanceLib() : m_defaults(NULL) 
{
    // chroma/chroma/geometry.py
    // standard_wavelengths = np.arange(60, 810, 20).astype(np.float32)

    setStandardDomain( new GDomain<double>(60.f, 810.f, 20.f ));

    GPropertyMap* defaults = new GPropertyMap("defaults", UINT_MAX, "defaults");
    defaults->setStandardDomain(getStandardDomain());

    // TODO: find way to avoid detector specific keys and default values being hardcoded ? 
    // or at least move up to a higher level of the code 

    defaults->addConstantProperty( "RINDEX",      1.f  );
    defaults->addConstantProperty( "ABSLENGTH",     1e6  );
    defaults->addConstantProperty( "RAYLEIGH",      1e6  );
    defaults->addConstantProperty( "REEMISSIONPROB", 0.f );

    setDefaults(defaults);
}

GSubstanceLib::~GSubstanceLib()
{
}

unsigned int GSubstanceLib::getNumSubstances()
{
   return m_keys.size();
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


