#include "GPropertyMap.hh"


GPropertyMap::GPropertyMap(const char* name)
{
   m_name = name ;
}

GPropertyMap::~GPropertyMap()
{
}

void GPropertyMap::AddProperty(const char* pname, float* values, float* domain, size_t length )
{
   m_prop[pname] = new GProperty<float>(values, domain, length) ;  
}

GProperty<float>* GPropertyMap::GetProperty(const char* pname)
{
   return (m_prop.find(pname) != m_prop.end()) ? m_prop[pname] : NULL ;
}

