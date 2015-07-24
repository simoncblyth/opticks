#include "GPropertyMap.hh"
#include "GOpticalSurface.hh"
#include "md5digest.hpp"
#include "string.h"
#include "limits.h"
#include "assert.h"
#include <sstream>
#include <iomanip>


template <typename T>
const char* GPropertyMap<T>::NOT_DEFINED = "-" ;




template <typename T>
GPropertyMap<T>::GPropertyMap(GPropertyMap<T>* other) 
      : 
      m_name(other ? other->getName() : NOT_DEFINED ),
      m_shortname(NULL),
      m_index(other ? other->getIndex() : UINT_MAX ),
      m_type(other ? other->getType() : "" ),
      m_standard_domain(NULL),
      m_optical_surface(other ? other->getOpticalSurface() : NULL )
{

    findShortName();
}


template <typename T>
GPropertyMap<T>::GPropertyMap(const char* name)
    : 
    m_shortname(NULL),
    m_standard_domain(NULL),
    m_optical_surface(NULL)
{
   m_name = name ; 
   m_index = UINT_MAX ;
   m_type = "" ;
   findShortName();
}

template <typename T>
GPropertyMap<T>::GPropertyMap(const char* name, unsigned int index, const char* type, GOpticalSurface* optical_surface) 
   : 
   m_index(index), 
   m_standard_domain(NULL),
   m_optical_surface(optical_surface)
{
   // set the std::string
   m_name = name ; 
   m_type = type ; 
   findShortName();
}

template <typename T>
GPropertyMap<T>::~GPropertyMap()
{
}


template <typename T>
void GPropertyMap<T>::setStandardDomain(GDomain<T>* standard_domain)
{
    m_standard_domain = standard_domain ; 
}

template <typename T>
GDomain<T>* GPropertyMap<T>::getStandardDomain()
{
    return m_standard_domain ;
}
template <typename T>
bool GPropertyMap<T>::hasStandardDomain()
{
    return m_standard_domain != NULL;
}



/*
template <typename T>
char* GPropertyMap<T>::digest()
{
   // NB digest excludes the name
   MD5Digest dig ;
   for(GPropertyMap_t::iterator it=m_prop.begin() ; it != m_prop.end() ; it++ )
   {
       std::string key = it->first ;
       char* k = (char*)key.c_str();  
       GProperty<T>* prop = it->second ; 
       dig.update(k, strlen(k));

       char* pdig = prop->digest();
       dig.update(pdig, strlen(pdig));
       free(pdig);
   } 
   return dig.finalize();
}
*/


template <typename T>
char* GPropertyMap<T>::pdigest(int ifr, int ito)
{
    MD5Digest dig ;

    if(ito == ifr) printf("GPropertyMap<T>::pdigest unexpected ifr/ito %d/%d \n", ifr, ito); 
    assert(ito >= ifr);

    for(int i=ifr ; i < ito ; ++i )
    {
        GProperty<T>* prop = getPropertyByIndex(i) ; 
        if(!prop) continue ; 

        char* pdig = prop->digest();
        dig.update(pdig, strlen(pdig));
        free(pdig);
        
    }

    if(m_optical_surface)
    {
        char* sdig = m_optical_surface->digest();
        dig.update(sdig, strlen(sdig));
        free(sdig);
    }

    return dig.finalize();
}

template <typename T>
std::string GPropertyMap<T>::getPDigestString(int ifr, int ito)
{
    return pdigest(ifr, ito);
}


template <typename T>
const char* GPropertyMap<T>::getName()
{
    return m_name.c_str();
}

template <typename T>
std::string GPropertyMap<T>::getShortNameString()
{    
    return getShortName(); 
}
template <typename T>
const char* GPropertyMap<T>::getShortName()
{
    return m_shortname ; 
}

template <typename T>
bool GPropertyMap<T>::hasShortName(const char* name)
{
    return strcmp(m_shortname, name) == 0 ;
}

template <typename T>
bool GPropertyMap<T>::hasDefinedName()
{
    return strcmp(m_shortname, NOT_DEFINED) != 0 ;
}

template <typename T>
char* GPropertyMap<T>::trimSuffixPrefix(const char* origname, const char* prefix)
{
    //  __dd__Materials__ADTableStainlessSteel0xc177178    0x is 9 chars from the end
    const char* ox = "0x" ;
    char* name = strdup(origname);
    char* c = name + strlen(name) - 9 ;              
    if(strncmp(c, ox, strlen(ox)) == 0) *c = '\0';   // insert NULL to snip off the 0x tail
    if(prefix) name += strlen(prefix) ;
    return name ;  
}

template <typename T>
void GPropertyMap<T>::findShortName(const char* prefix)
{
    //printf("GPropertyMap<T>::getShortName %s \n", prefix);

    if(m_name.empty() || strcmp(m_name.c_str(), NOT_DEFINED) == 0)
    { 
        m_shortname = NOT_DEFINED ;
    }  
    else if(strncmp( m_name.c_str(), prefix, strlen(prefix)) == 0)
    { 
        m_shortname = trimSuffixPrefix(m_name.c_str(), prefix); 
    }
    else
    {
        // when doesnt match the prefix, eg for surface names
        //     __dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib1Surface
        // just provide chars after the last _

        if( m_name[0] == '_') //  detect dyb names by first char 
        { 
            const char* p = strrchr(m_name.c_str(), '_') ;
            m_shortname = p ? p + 1 :  NOT_DEFINED ; 
        }
        else
        {
            //  JUNO names have no prefix and are short, so just trim the 0x tail
            m_shortname = trimSuffixPrefix(m_name.c_str(), NULL) ; 
        }


    }
}


template <typename T>
std::string GPropertyMap<T>::description()
{
    std::stringstream ss ; 
    ss << "GPropertyMap<T>:: "  
       << std::setw(2)  << getIndex()
       << std::setw(15) << getType()
       ;

    if(m_optical_surface)
    {
        assert(strcmp(m_optical_surface->getShortName(), m_shortname)==0);
        ss << " s:" << m_optical_surface->description() ;
    }
    else
    {
        ss << " m:" << getShortName() ;
    }
    ss << " k:" << getKeysString() ;

    return ss.str();
}
  

template <typename T>
unsigned int GPropertyMap<T>::getIndex()
{
    return m_index ; 
}
template <typename T>
const char* GPropertyMap<T>::getType()
{
    return m_type.c_str();
}

template <typename T>
bool GPropertyMap<T>::isSkinSurface()
{
    return m_type == "skinsurface" ;
}
template <typename T>
bool GPropertyMap<T>::isBorderSurface()
{
    return m_type == "bordersurface" ;
}
template <typename T>
bool GPropertyMap<T>::isMaterial()
{
    return m_type == "material" ;
}


template <typename T>
void GPropertyMap<T>::addConstantProperty(const char* pname, T value, const char* prefix)
{
   assert(m_standard_domain);
   GProperty<T>* prop = GProperty<T>::from_constant( value, m_standard_domain->getValues(), m_standard_domain->getLength() );
   addProperty(pname, prop);
}

template <typename T>
void GPropertyMap<T>::addProperty(const char* pname, T* values, T* domain, unsigned int length, const char* prefix)
{
   //printf("GPropertyMap<T>::addProperty name %s pname %s length %u \n", getName(), pname, length );
   assert(length < 1000);

   GAry<T>* vals = new GAry<T>( length, values );
   GAry<T>* doms  = new GAry<T>( length, domain );
   GProperty<T>* orig = new GProperty<T>(vals, doms)  ;  

   if(m_standard_domain)
   {
       GProperty<T>* ipol = orig->createInterpolatedProperty(m_standard_domain); 
   
       //orig->Summary("orig", 10 );
       //ipol->Summary("ipol", 10 );

       addProperty(pname, ipol, prefix) ;  
   } 
   else
   {
       addProperty(pname, orig, prefix);
   }
}


template <typename T>
void GPropertyMap<T>::addProperty(const char* pname,  GProperty<T>* prop, const char* _prefix)
{
    if(!prop)
    {
        printf("GPropertyMap<T>::addProperty pname %s NULL PROPERTY \n", pname);
    }
    assert(prop); 


    std::string key(pname) ;
    if(_prefix) key = _prefix + key ;

    m_keys.push_back(key);
    m_prop[key] = prop ;  
}


template <typename T>
std::vector<std::string>& GPropertyMap<T>::getKeys()
{
    return m_keys ; 
}

template <typename T>
GProperty<T>* GPropertyMap<T>::getProperty(const char* pname)
{
   return (m_prop.find(pname) != m_prop.end()) ? m_prop[pname] : NULL ;
}

template <typename T>
bool GPropertyMap<T>::hasProperty(const char* pname)
{
   return m_prop.find(pname) != m_prop.end() ;
}

template <typename T>
GProperty<T>* GPropertyMap<T>::getPropertyByIndex(int index)
{
   if(index < 0) index += m_keys.size() ;
   std::string key = m_keys[index];
   return getProperty(key.c_str()); 
}

template <typename T>
const char* GPropertyMap<T>::getPropertyNameByIndex(int index)
{
   if(index < 0) index += m_keys.size() ;
   return m_keys[index].c_str(); 
}




template <typename T>
void GPropertyMap<T>::Summary(const char* msg, unsigned int nline)
{
   if(nline == 0) return ;

   unsigned int n = getNumProperties();
   std::string pdig = getPDigestString(0,n);

   printf("%s %s %d %s %s\n", msg, getType(), getIndex(), pdig.c_str(), getName()); 

   for(std::vector<std::string>::iterator it=m_keys.begin() ; it != m_keys.end() ; it++ )
   {
       std::string key = *it ;
       GProperty<T>* prop = m_prop[key] ; 
       prop->Summary(key.c_str(), nline);
   } 

   if(m_optical_surface) m_optical_surface->Summary(msg, nline);
}


template <typename T>
unsigned int GPropertyMap<T>::getNumProperties()
{

   if(m_prop.size() != m_keys.size())
   {
      printf("GPropertyMap<T>::getNumProperties prop/keys mismatch prop %lu  keys %lu \n", m_prop.size(), m_keys.size()); 
   }

   assert(m_prop.size() == m_keys.size()); // maybe a duplicated key can trigger this
   return m_prop.size();
}

template <typename T>
std::string GPropertyMap<T>::getKeysString()
{
    std::stringstream ss ; 
    unsigned int nkeys = m_keys.size();
    for(unsigned int i=0 ; i < nkeys ; i++)
    {
        ss << m_keys[i] ;
        if( i < nkeys - 1) ss << " " ;
    }
    return ss.str() ;
}

template <typename T>
void GPropertyMap<T>::add(GPropertyMap<T>* other, const char* prefix)
{
    unsigned int n = other->getNumProperties();
    for(unsigned int i=0 ; i<n ; i++)
    {
         const char* name  = other->getPropertyNameByIndex(i); 
         GProperty<T>* prop = other->getPropertyByIndex(i); 

         addProperty( name, prop, prefix );
    }
}





/*
* :google:`move templated class implementation out of header`
* http://www.drdobbs.com/moving-templates-out-of-header-files/184403420

A compiler warning "declaration does not declare anything" was avoided
by putting the explicit template instantiation at the tail rather than the 
head of the implementation.
*/

template class GPropertyMap<float>;
template class GPropertyMap<double>;

