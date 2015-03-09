#include "GSubstanceLib.hh"
#include "GSubstance.hh"
#include "GPropertyMap.hh"

#include "assert.h"
#include "stdio.h"
#include "limits.h"

GSubstanceLib::GSubstanceLib()
{
}

GSubstanceLib::~GSubstanceLib()
{
}

void GSubstanceLib::Summary(const char* msg)
{
    printf("%s\n", msg );
    char buf[128];
    for(unsigned int i=0 ; i < m_keys.size() ; i++)
    {
         std::string key = m_keys[i] ;  
         GSubstance* substance = m_registry[key];
         assert(substance->getIndex() == i );
         snprintf(buf, 128, "%s substance ", msg );
         substance->Summary(buf);
    } 
}



GSubstance* GSubstanceLib::get(GPropertyMap* imaterial, GPropertyMap* isurface, GPropertyMap* osurface )
{ 
    //printf("GSubstanceLib::get imaterial %p isurface %p osurface %p \n", imaterial, isurface, osurface );

    GSubstance* tmp = new GSubstance(imaterial, isurface, osurface);
    std::string key = tmp->digest();

    if(m_registry.count(key) == 0) // substance not yet registered, with substance identity based on the digest 
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


