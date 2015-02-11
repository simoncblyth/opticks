#include "GPropertyMap.hh"


GPropertyMap::GPropertyMap(const char* name, const char* type)
{
   m_name = name ;
   m_type = type ;
}

GPropertyMap::~GPropertyMap()
{
}

const char* GPropertyMap::getName()
{
    return m_name.c_str();
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
   printf("%s %s %s\n", msg, getType(), getName()); 
   for(GPropertyMapF_t::iterator it=m_prop.begin() ; it != m_prop.end() ; it++ )
   {
       std::string key = it->first ;
       GPropertyF* prop = it->second ; 
       prop->Summary(key.c_str());
   } 
}


