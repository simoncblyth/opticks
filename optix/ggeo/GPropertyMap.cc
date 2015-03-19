#include "GPropertyMap.hh"
#include "md5digest.hh"
#include "string.h"
#include "limits.h"
#include <sstream>

GPropertyMap::GPropertyMap(const char* name) : m_standard_domain(NULL)
{
   m_name = name ; 
   m_index = UINT_MAX ;
   m_type = "" ;
}

GPropertyMap::GPropertyMap(const char* name, unsigned int index, const char* type) : m_index(index), m_standard_domain(NULL)
{
   // set the std::string
   m_name = name ; 
   m_type = type ; 
}

GPropertyMap::~GPropertyMap()
{
}


void GPropertyMap::setStandardDomain(GDomain<double>* standard_domain)
{
    m_standard_domain = standard_domain ; 
}

GDomain<double>* GPropertyMap::getStandardDomain()
{
    return m_standard_domain ;
}




char* GPropertyMap::digest()
{
   // NB digest excludes the name
   MD5Digest dig ;
   for(GPropertyMapD_t::iterator it=m_prop.begin() ; it != m_prop.end() ; it++ )
   {
       std::string key = it->first ;
       char* k = (char*)key.c_str();  
       GPropertyD* prop = it->second ; 
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


void GPropertyMap::addConstantProperty(const char* pname, double value )
{
   assert(m_standard_domain);
   GProperty<double>* prop = GProperty<double>::from_constant( value, m_standard_domain->getValues(), m_standard_domain->getLength() );
   addProperty(pname, prop);
}

void GPropertyMap::addProperty(const char* pname, double* values, double* domain, unsigned int length)
{
   //printf("GPropertyMap::addProperty name %s pname %s length %u \n", getName(), pname, length );
   assert(length < 1000);

   GAry<double>* vals = new GAry<double>( length, values );
   GAry<double>* doms  = new GAry<double>( length, domain );
   GProperty<double>* orig = new GProperty<double>(vals, doms)  ;  

   if(m_standard_domain)
   {
       GProperty<double>* ipol = orig->createInterpolatedProperty(m_standard_domain); 
   
       //orig->Summary("orig", 10 );
       //ipol->Summary("ipol", 10 );

       addProperty(pname, ipol) ;  
   } 
   else
   {
       addProperty(pname, orig);
   }
}


void GPropertyMap::addProperty(const char* pname,  GProperty<double>* prop)
{
    std::string key(pname);
    m_keys.push_back(key);
    m_prop[key] = prop ;  
}


std::vector<std::string>& GPropertyMap::getKeys()
{
    return m_keys ; 
}

GPropertyD* GPropertyMap::getProperty(const char* pname)
{
   return (m_prop.find(pname) != m_prop.end()) ? m_prop[pname] : NULL ;
}

GPropertyD* GPropertyMap::getPropertyByIndex(unsigned int index)
{
   std::string key = m_keys[index];
   return getProperty(key.c_str()); 
}

const char* GPropertyMap::getPropertyNameByIndex(unsigned int index)
{
   return m_keys[index].c_str(); 
}




void GPropertyMap::Summary(const char* msg, unsigned int nline)
{
   if(nline == 0) return ;
   printf("%s %s %d %s %s\n", msg, getType(), getIndex(), digest(), getName()); 

   for(std::vector<std::string>::iterator it=m_keys.begin() ; it != m_keys.end() ; it++ )
   {
       std::string key = *it ;
       GPropertyD* prop = m_prop[key] ; 
       prop->Summary(key.c_str(), nline);
   } 
}


unsigned int GPropertyMap::getNumProperties()
{
   assert(m_prop.size() == m_keys.size());
   return m_prop.size();
}

std::string GPropertyMap::getKeysString()
{
    std::stringstream ss ; 
    for(std::vector<std::string>::iterator it=m_keys.begin() ; it != m_keys.end() ; it++ )
    {
        ss << " " ;
        ss << *it ;
    }
    return ss.str() ;
}




