#include "GPropertyMap.hh"
#include "md5digest.hh"
#include "string.h"
#include <sstream>


GPropertyMap::GPropertyMap(const char* name, unsigned int index, const char* type)
{
   m_name = name ;
   m_index = index ;
   m_type = type ;
}

GPropertyMap::~GPropertyMap()
{
}

void GPropertyMap::setStandardDomain( float low, float high, float step)
{
    m_low = low ;
    m_high = high ;
    m_step = step ;
}
float GPropertyMap::getLow()
{
    return m_low ; 
}
float GPropertyMap::getHigh()
{
    return m_high ; 
}
float GPropertyMap::getStep()
{
    return m_step ; 
}



char* GPropertyMap::digest()
{
   // NB digest excludes the name
   MD5Digest dig ;
   for(GPropertyMapF_t::iterator it=m_prop.begin() ; it != m_prop.end() ; it++ )
   {
       std::string key = it->first ;
       char* k = (char*)key.c_str();  
       GPropertyF* prop = it->second ; 
       dig.update(k, strlen(k));

       char* pdig = prop->digest();
       dig.update(pdig, strlen(pdig));
   } 
   return dig.finalize();
}


const char* GPropertyMap::getName()
{
    return m_name.c_str();
}

char* GPropertyMap::getKeys()
{
    std::stringstream ss ; 
    for(GPropertyMapF_t::iterator it=m_prop.begin() ; it != m_prop.end() ; it++ )
    {
        ss << " " ;
        ss << it->first ;
    }
    return strdup(ss.str().c_str());
}



char* GPropertyMap::getShortName(const char* prefix)
{
    char* name = strdup(m_name.c_str());

    const char* ox = "0x" ;

    //  __dd__Materials__ADTableStainlessSteel0xc177178    0x is 9 chars from the end

    char* c = name + strlen(name) - 9 ;

    if(strncmp(c, ox, strlen(ox)) == 0) *c = '\0';

    if(strncmp( name, prefix, strlen(prefix)) == 0 ) name += strlen(prefix) ;

    return strdup(name) ;
}



unsigned int GPropertyMap::getIndex()
{
    return m_index ; 
}
const char* GPropertyMap::getType()
{
    return m_type.c_str();
}

bool GPropertyMap::isSkinSurface()
{
    return m_type == "skinsurface" ;
}
bool GPropertyMap::isBorderSurface()
{
    return m_type == "bordersurface" ;
}
bool GPropertyMap::isMaterial()
{
    return m_type == "material" ;
}


void GPropertyMap::addProperty(const char* pname, float* values, float* domain, size_t length )
{
   GProperty<float>* prop = new GProperty<float>(values, domain, length) ;  
   prop->setStandardDomain( getLow(), getHigh(), getStep() );
   m_prop[pname] = prop ;  
}

GProperty<float>* GPropertyMap::getProperty(const char* pname)
{
   return (m_prop.find(pname) != m_prop.end()) ? m_prop[pname] : NULL ;
}


void GPropertyMap::Summary(const char* msg)
{
   printf("%s %s %d %s %s\n", msg, getType(), getIndex(), digest(), getName()); 
   for(GPropertyMapF_t::iterator it=m_prop.begin() ; it != m_prop.end() ; it++ )
   {
       std::string key = it->first ;
       GPropertyF* prop = it->second ; 
       prop->Summary(key.c_str());
   } 
}


