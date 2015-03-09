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





void GPropertyMap::AddProperty(const char* pname, float* values, float* domain, size_t length )
{
   m_prop[pname] = new GProperty<float>(values, domain, length) ;  
}

GProperty<float>* GPropertyMap::GetProperty(const char* pname)
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


