/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <climits>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <cstring>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


// sysrap-
#include "SDigest.hh"
// brap-
#include "BStr.hh"
#include "BDir.hh"
#include "BMeta.hh"

// ggeo-

#include "GAry.hh"
#include "GProperty.hh"
#include "GDomain.hh"
#include "GOpticalSurface.hh"
#include "GPropertyMap.hh"
#include "GSurfaceLib.hh"

#include "PLOG.hh"

template <typename T>
const char* GPropertyMap<T>::NOT_DEFINED = "-" ;


template <typename T>
const plog::Severity GPropertyMap<T>::LEVEL = debug ;



template <typename T>
BMeta* GPropertyMap<T>::getMeta() const 
{
    return m_meta ; 
}

template <typename T>
std::string GPropertyMap<T>::getMetaDesc() const 
{
    return m_meta ? m_meta->desc() : "no-meta"  ; 
}




template <typename T>
template <typename S>
void GPropertyMap<T>::setMetaKV(const char* key, S val)  
{
    return m_meta->set(key, val) ; 
}

template <typename T>
template <typename S>
S GPropertyMap<T>::getMetaKV(const char* key, const char* fallback)  const 
{
    return m_meta->get<S>(key, fallback) ; 
}



template <typename T>
bool GPropertyMap<T>::hasMetaItem(const char* key) const 
{
    return m_meta->hasItem(key) ; 
}


template <typename T>
std::string GPropertyMap<T>::getBPV1() const 
{
    assert( m_type.compare(GSurfaceLib::BORDERSURFACE) == 0 ) ; 
    std::string bpv1 = m_meta->get<std::string>( GSurfaceLib::BPV1 ) ;  
    assert( !bpv1.empty() );
    return bpv1 ; 
}
template <typename T>
std::string GPropertyMap<T>::getBPV2() const 
{
    assert( m_type.compare(GSurfaceLib::BORDERSURFACE) == 0 ) ; 
    std::string bpv2 = m_meta->get<std::string>( GSurfaceLib::BPV2 ) ;  
    assert( !bpv2.empty() );
    return bpv2 ; 
}
template <typename T>
std::string GPropertyMap<T>::getSSLV() const 
{
    assert( m_type.compare(GSurfaceLib::SKINSURFACE) == 0 ) ; 
    std::string sslv = m_meta->get<std::string>( GSurfaceLib::SSLV ) ;  
    assert( !sslv.empty() );
    return sslv ; 
}


template <typename T>
void GPropertyMap<T>::setEnergyDomain()
{
    m_energy_domain = true ;  
}

template <typename T>
bool GPropertyMap<T>::hasEnergyDomain() const 
{
    return m_energy_domain ;  
}




template <typename T>
GPropertyMap<T>::GPropertyMap(GPropertyMap<T>* other, GDomain<T>* domain) 
    : 
    m_name(other ? other->getName() : NOT_DEFINED ),
    m_shortname(NULL),
    m_type(other ? other->getType() : "" ),
    m_index(other ? other->getIndex() : UINT_MAX ),
    m_valid(other ? other->isValid() : false),
    m_standard_domain(domain),
    m_optical_surface(other ? other->getOpticalSurface() : NULL ),
    m_meta(other && other->getMeta() ? new BMeta(*other->getMeta()) :  new BMeta),
    m_energy_domain(false)  
{
    init();

    if(m_standard_domain)
    {
        LOG(verbose) << "GPropertyMap<T> interpolating copy ctor changing domain "
              << " other step " << other->getDomainStep()
              << " dst step " << this->getDomainStep()
              ;

        addMapStandardized(other);  // interpolation done here 
    }
}

template <typename T>
GPropertyMap<T>* GPropertyMap<T>::spawn_interpolated(T nm)
{
    return new GPropertyMap<T>(this, m_standard_domain->makeInterpolationDomain(nm)); 
}


template <typename T>
GPropertyMap<T>::GPropertyMap(const char* name)
    : 
    m_shortname(NULL),
    m_standard_domain(NULL),
    m_optical_surface(NULL),
    m_meta(new BMeta)
{
   m_name = name ;   // m_name is std::string, no need for strdup 
   m_index = UINT_MAX ;
   m_valid = false ;
   m_type = "" ;

   init();
}


// this ctor is used in eg GSurfaceLib::importForTex2d
template <typename T>
GPropertyMap<T>::GPropertyMap(const char* name, unsigned int index, const char* type, GOpticalSurface* optical_surface, BMeta* meta) 
   : 
   m_index(index), 
   m_valid(false),
   m_standard_domain(NULL),
   m_optical_surface(optical_surface),
   m_meta(meta ? new BMeta(*meta) : new BMeta)
{
   // set the std::string
   m_name = name ; 
   m_type = type ; 

   init();
}

template <typename T>
GPropertyMap<T>::~GPropertyMap()
{
}



template <class T>
void GPropertyMap<T>::init()
{
   findShortName();
   collectMeta();
}


template <typename T>
std::string GPropertyMap<T>::brief() const 
{
    std::stringstream ss ; 
    ss << "GPropertyMap " 
       << " type " << m_type 
       << " name " << m_name
       ; 

    return ss.str();
}




template <class T>
void GPropertyMap<T>::collectMeta()
{
    m_meta->set<int>("index", m_index );
    m_meta->set<std::string>("shortname", m_shortname );
    m_meta->set<std::string>("name", m_name );
    m_meta->set<std::string>("type", m_type );

}

template <class T>
void GPropertyMap<T>::dumpMeta(const char* msg) const 
{
    LOG(info) << msg 
              << " m_type : " << m_type
              ;
    m_meta->dump();
}







template <class T>
void GPropertyMap<T>::setOpticalSurface(GOpticalSurface* optical_surface)
{
    m_optical_surface = optical_surface ;
}
template <class T>
GOpticalSurface* GPropertyMap<T>::getOpticalSurface()
{
    return m_optical_surface ; 
}
template <class T>
bool GPropertyMap<T>::hasNameEnding(const char* end)
{
    std::string suffix(end) ;
    return m_name.size() >= suffix.size() &&
           m_name.compare(m_name.size() - suffix.size(), suffix.size(), suffix) == 0;  
}



template <class T>
const char*  GPropertyMap<T>::EFFICIENCY = "EFFICIENCY" ; 

template <class T>
const char*  GPropertyMap<T>::detect = "detect" ; 


template <class T>
bool GPropertyMap<T>::isSensor()
{
    return hasNonZeroProperty(EFFICIENCY) || hasNonZeroProperty(detect) ; 
}
template <class T>
void GPropertyMap<T>::setSensor(bool sensor)
{
    assert(0 && "sensors are now detected by the prescense of an EFFICIENCY property" ); 
}

template <class T>
bool GPropertyMap<T>::isValid()
{
    return m_valid ; 
}
template <class T>
void GPropertyMap<T>::setValid(bool valid)
{
    m_valid = valid ; 
}


template <typename T>
void GPropertyMap<T>::setStandardDomain(GDomain<T>* standard_domain)
{
    if(standard_domain == NULL)
    {
        standard_domain = GDomain<T>::GetDefaultDomain();
        LOG(LEVEL) << " setStandardDomain(NULL) -> default_domain " << standard_domain->desc() ;
    } 
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

template <typename T>
T GPropertyMap<T>::getDomainLow()
{
    return m_standard_domain->getLow();
}

template <typename T>
T GPropertyMap<T>::getDomainHigh()
{
    return m_standard_domain->getHigh();
}

template <typename T>
T GPropertyMap<T>::getDomainStep()
{
    return m_standard_domain->getStep();
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




/*
template <typename T>
char* GPropertyMap<T>::ndigest()
{
    // simple name based identity
    MD5Digest dig ;
    const char* pdig = getShortName();
    dig.update(pdig, strlen(pdig));
    return dig.finalize();
}
*/

template <typename T>
char* GPropertyMap<T>::pdigest(int ifr, int ito) const
{
    SDigest dig ;

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
        const char* sdig = m_optical_surface->digest();
        dig.update_str(sdig);
    }

    return dig.finalize();
}

template <typename T>
std::string GPropertyMap<T>::getPDigestString(int ifr, int ito) const 
{
    return pdigest(ifr, ito);
}


template <typename T>
const char* GPropertyMap<T>::getName() const 
{
    return m_name.c_str();
}

template <typename T>
std::string GPropertyMap<T>::getShortNameString() const 
{    
    return getShortName(); 
}
template <typename T>
const char* GPropertyMap<T>::getShortName() const 
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
const char* GPropertyMap<T>::FindShortName(const char* name, const char* prefix)
{
    //printf("GPropertyMap<T>::getShortName %s \n", prefix);

    const char* shortname = NULL ; 

    if(strcmp(name, NOT_DEFINED) == 0)
    { 
        shortname = NOT_DEFINED ;
    }  
    else if( prefix && strncmp( name, prefix, strlen(prefix)) == 0)
    { 
        shortname = BStr::trimPointerSuffixPrefix(name, prefix); 
    }
    else
    {
        // when doesnt match the prefix, eg for surface names
        //     __dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib1Surface
        // just provide chars after the last _

        if( name[0] == '_') //  detect dyb names by first char 
        { 
            const char* p = strrchr(name, '_') ;
            shortname = p ? p + 1 :  NOT_DEFINED ; 
        }
        else
        {
            //  JUNO names have no prefix and are short, so just trim the 0x tail
            shortname = BStr::trimPointerSuffixPrefix(name, NULL) ; 
        }
    }

    return shortname ? strdup(shortname) : NULL ; 
}



template <typename T>
void GPropertyMap<T>::findShortName(const char* prefix)
{
    const char* name = m_name.empty() ? NOT_DEFINED : m_name.c_str() ; 
    m_shortname = FindShortName(name, prefix );  

}




template <typename T>
std::string GPropertyMap<T>::description() const 
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
unsigned GPropertyMap<T>::getIndex() const 
{
    return m_index ; 
}

template <typename T>
void GPropertyMap<T>::setIndex(unsigned index)  
{
    m_index = index ; 
}



template <typename T>
const char* GPropertyMap<T>::getType() const 
{
    return m_type.c_str();
}


template <typename T> 
bool GPropertyMap<T>::isSkinSurface() const 
{ 
    return m_type.compare(GSurfaceLib::SKINSURFACE) == 0 ;
}

template <typename T>
bool GPropertyMap<T>::isBorderSurface() const 
{
    return m_type.compare(GSurfaceLib::BORDERSURFACE) == 0 ;
}

template <typename T>
void GPropertyMap<T>::setBorderSurface()  
{
    m_type = GSurfaceLib::BORDERSURFACE ;
}



template <typename T>
void GPropertyMap<T>::setImplicit(bool implicit)  
{
    setMetaKV<int>("implicit", 1); 
}
template <typename T>
bool GPropertyMap<T>::isImplicit() const  
{
    return getMetaKV<int>("implicit","0") ; 
}





template <typename T>
void GPropertyMap<T>::setSkinSurface()  
{
    m_type = GSurfaceLib::SKINSURFACE ;
}

template <typename T>
bool GPropertyMap<T>::isTestSurface() const
{
    return m_type.compare(GSurfaceLib::TESTSURFACE) == 0 ;
}


template <typename T>
bool GPropertyMap<T>::isSurface() const 
{
    return isTestSurface() || isSkinSurface() || isBorderSurface() ;
}

template <typename T>
bool GPropertyMap<T>::isMaterial() const 
{
    return m_type.compare("material") == 0 ;
}


template <typename T>
std::string GPropertyMap<T>::desc() const 
{
    bool isSS = isSkinSurface()  ;
    bool isBS = isBorderSurface()  ;
    bool isTS = isTestSurface() ;
    bool isSU = isSurface() ;
    bool isMT = isMaterial() ;

    std::stringstream ss ; 
    ss 
       << " GPropertyMap " 
       << " type " << std::setw(15) << m_type
       << " name " << std::setw(30) << m_name
       << " isSS " << isSS
       << " isBS " << isBS
       << " isTS " << isTS
       << " isSU " << isSU
       << " isMT " << isMT
       ;

    if(isSS) ss << " sslv " << getSSLV() ; 
    if(isBS) ss << " bpv1 " << getBPV1()  
                << " bpv2 " << getBPV2() 
                ; 

    return ss.str();
}








template <typename T>
void GPropertyMap<T>::addConstantProperty(const char* pname, T value, const char* prefix)
{
   assert(m_standard_domain);
   GProperty<T>* prop = GProperty<T>::from_constant( value, m_standard_domain->getValues(), m_standard_domain->getLength() );
   addPropertyAsis(pname, prop, prefix);
}

template <typename T>
void GPropertyMap<T>::addPropertyStandardized(const char* pname, T* values, T* domain, unsigned int length, const char* prefix)
{
   assert(0) ; // checking where this gets used
   assert(length < 10000);

   GAry<T>* vals = new GAry<T>( length, values );
   GAry<T>* doms  = new GAry<T>( length, domain );
   GProperty<T>* orig = new GProperty<T>(vals, doms)  ;  

   addPropertyStandardized(pname, orig, prefix);

   LOG(debug) 
        << " orig " << orig   
        << " m_name " << m_name   
        << " pname " << pname
        ;  
}


template <typename T>
void GPropertyMap<T>::addPropertyStandardized(const char* pname,  GProperty<T>* orig, const char* prefix)
{
   if(m_standard_domain)
   {
       GProperty<T>* ipol = orig->createInterpolatedProperty(m_standard_domain); 
       addPropertyAsis(pname, ipol, prefix) ;  
   } 
   else
   {
       addPropertyAsis(pname, orig, prefix);
   }
}
 

/**
GPropertyMap<T>::addPropertyAsis
---------------------------------

NB this method makes a copy of the property being added 
prior to inclusion into the collection.  This is necessary 
for independance of the propertyMap, such that it owns 
the properties it holds and can potentially change them 
without effecting other propertyMap instances.

For more on why this is needed see GPropertyLib::getPropertyOrDefault

**/

template <typename T>
void GPropertyMap<T>::addPropertyAsis(const char* pname,  GProperty<T>* prop, const char* _prefix)
{
    if(!prop)
    {
        printf("GPropertyMap<T>::addProperty pname %s NULL PROPERTY \n", pname);
    }
    assert(prop); 


    std::string key(pname) ;
    if(_prefix) key = _prefix + key ;

    m_keys.push_back(key);
    m_prop[key] = prop->copy() ;  
}


template <typename T>
std::vector<std::string>& GPropertyMap<T>::getKeys()
{
    return m_keys ; 
}

template <typename T>
GProperty<T>* GPropertyMap<T>::getProperty(const char* pname) const 
{
   return (m_prop.find(pname) != m_prop.end()) ? m_prop.at(pname) : NULL ;
}



template <typename T>
const GProperty<T>* GPropertyMap<T>::getPropertyConst(const char* pname) const 
{
   return (m_prop.find(pname) != m_prop.end()) ? m_prop.at(pname) : NULL ;
}


template <typename T>
unsigned GPropertyMap<T>::size() const
{
    return m_prop.size(); 
}

template <typename T>
std::string GPropertyMap<T>::dump_ptr() const
{
    typedef std::map<std::string,GProperty<T>*> GPropertyMap_t ;
    std::stringstream ss ; 

    for( typename GPropertyMap_t::const_iterator it=m_prop.begin() ; it != m_prop.end() ; it++)
    {
        ss 
            << " (" << it->first 
            << " " << it->second 
            << " ) " 
            ; 
    }
    return ss.str();  
}

template <typename T>
GProperty<T>* GPropertyMap<T>::getProperty(const char* pname, const char* prefix)
{
    std::string key(pname) ;
    if(prefix) key = prefix + key ;

    return (m_prop.find(key.c_str()) != m_prop.end()) ? m_prop[key.c_str()] : NULL ;
}

template <typename T>
bool GPropertyMap<T>::hasNonZeroProperty(const char* pname) 
{
    if(!hasProperty(pname)) return false ; 
    GProperty<T>* prop = getProperty(pname);
    return !prop->isZero();
}


template <typename T>
bool GPropertyMap<T>::setPropertyValues(const char* pname, T val) 
{
    if(!hasProperty(pname)) return false ; 
    GProperty<T>* prop = getProperty(pname);
    prop->setValues(val);
    return true ; 
}


template <typename T>
bool GPropertyMap<T>::hasProperty(const char* pname) const  
{
    return m_prop.find(pname) != m_prop.end() ;
}

template <typename T>
GProperty<T>* GPropertyMap<T>::getPropertyByIndex(int index) const 
{
   int nprop  = m_keys.size(); 
   if(index < 0) index += nprop ;

   GProperty<T>* prop = NULL ; 
   if( index < nprop )
   { 
       std::string key = m_keys[index];
       prop = getProperty(key.c_str()); 
   } 
   return prop ; 
}

template <typename T>
const char* GPropertyMap<T>::getPropertyNameByIndex(int index) const 
{
   int nprop  = m_keys.size(); 
   if(index < 0) index += nprop ;

   const char* name = NULL ; 
   if( index < nprop )
   {
       name = m_keys[index].c_str(); 
   }
   return name ; 
}



template <typename T>
void GPropertyMap<T>::dump(const char* msg, unsigned int /*nline*/)
{
    //Summary(msg, nline);
    LOG(info) << msg ;
    LOG(info) << make_table();
}


template <typename T>
void GPropertyMap<T>::Summary(const char* msg, unsigned int nline) const 
{
   if(nline == 0) return ;

   unsigned int n = getNumProperties();
   std::string pdig = getPDigestString(0,n);

   printf("%s %s %d %s %s\n", msg, getType(), getIndex(), pdig.c_str(), getName()); 

   typedef std::vector<std::string> VS ;  

   for(VS::const_iterator it=m_keys.begin() ; it != m_keys.end() ; it++ )
   {
       std::string key = *it ;
       GProperty<T>* prop = m_prop.at(key) ; 
       prop->Summary(key.c_str(), nline);
   } 

   if(m_optical_surface) m_optical_surface->Summary(msg, nline);
}


template <typename T>
std::string GPropertyMap<T>::prop_desc() const 
{
   unsigned int n = getNumProperties();
   std::string pdig = getPDigestString(0,n);

   std::stringstream ss ; 

   ss << " typ " << std::setw(10) << getType()
      << " idx " << std::setw(4) << getIndex()
      << " dig " << std::setw(32) << pdig.c_str()
      << " npr " << std::setw(2)  << m_keys.size() 
      << " nam " << getName() 
      << std::endl 
      ; 
    
   typedef std::vector<std::string> VS ; 
 
   for(VS::const_iterator it=m_keys.begin() ; it != m_keys.end() ; it++ )
   {
       std::string key = *it ;
       //GProperty<T>* prop = m_prop[key] ; 
       const GProperty<T>* prop = getPropertyConst(key.c_str()); 

       ss << std::setw(15) << key 
          << " : " << prop->brief()
          << std::endl 
          ;
   } 
   return ss.str();
}






template <typename T>
std::string GPropertyMap<T>::make_table(unsigned int fw, T dscale, bool dreciprocal)
{

   std::vector< GProperty<T>* > vprops ; 
   std::vector< std::string > vtitles ; 

   std::vector< GProperty<T>* > cprops ; 
   std::vector< std::string > ctitles ; 

   std::vector< GProperty<T>* > dprops ; 
   std::vector< std::string > dtitles ; 

   std::vector< GProperty<T>* > eprops ; 
   std::vector< std::string > etitles ; 

   std::vector< GProperty<T>* > fprops ; 
   std::vector< std::string > ftitles ; 

   std::vector< GProperty<T>* > gprops ; 
   std::vector< std::string > gtitles ; 



   unsigned int clim = 5 ; 

   unsigned int nprop = getNumProperties() ;
   for(unsigned int i=0 ; i < nprop ; i++)
   {
       GProperty<T>* prop = getPropertyByIndex(i);
       std::string name = getPropertyNameByIndex(i) ;
       assert(prop);
       if(strlen(name.c_str()) == 0)
           LOG(warning) << "GPropertyMap<T>::make_table " << getName() << " property " << i << " has blank name " ;  
      
       if(prop->isConstant()) 
       {
           if(cprops.size() < clim)
           {
               cprops.push_back(prop);
               ctitles.push_back(name);
           }
           else if(dprops.size() < clim)
           {
               dprops.push_back(prop);
               dtitles.push_back(name);
           }
           else if(eprops.size() < clim)
           {
               eprops.push_back(prop);
               etitles.push_back(name);
           }
           else if(fprops.size() < clim)
           {
               fprops.push_back(prop);
               ftitles.push_back(name);
           }
           else if(gprops.size() < clim)
           {
               gprops.push_back(prop);
               gtitles.push_back(name);
           }

       }
       else
       {
           vprops.push_back(prop);
           vtitles.push_back(name);
       }
   }

   std::stringstream ss ; 
   ss << "GPropertyMap<T>::make_table"
      << " vprops " << vprops.size()
      << " cprops " << cprops.size()
      << " dprops " << dprops.size()
      << " eprops " << eprops.size()
      << " fprops " << fprops.size()
      << " gprops " << gprops.size()
      << std::endl;

   unsigned int cfw = 10 + fw ; 

   if(vprops.size() > 0) ss << GProperty<T>::make_table( fw, dscale, dreciprocal, false,vprops, vtitles ) ;
   if(cprops.size() > 0) ss << GProperty<T>::make_table( cfw, dscale, dreciprocal, true ,cprops, ctitles )  ;
   if(dprops.size() > 0) ss << GProperty<T>::make_table( cfw, dscale, dreciprocal, true ,dprops, dtitles )  ;
   if(eprops.size() > 0) ss << GProperty<T>::make_table( cfw, dscale, dreciprocal, true ,eprops, etitles )  ;
   if(fprops.size() > 0) ss << GProperty<T>::make_table( cfw, dscale, dreciprocal, true ,fprops, ftitles )  ;
   if(gprops.size() > 0) ss << GProperty<T>::make_table( cfw, dscale, dreciprocal, true ,gprops, gtitles )  ;
   return ss.str();

/*
   return GProperty<T>::make_table( 
                           fw, dscale, dreciprocal, 
                           getPropertyByIndex(0), getPropertyNameByIndex(0),
                           getPropertyByIndex(1), getPropertyNameByIndex(1),
                           getPropertyByIndex(2), getPropertyNameByIndex(2),
                           getPropertyByIndex(3), getPropertyNameByIndex(3),
                           getPropertyByIndex(4), getPropertyNameByIndex(4),
                           getPropertyByIndex(5), getPropertyNameByIndex(5),
                           getPropertyByIndex(6), getPropertyNameByIndex(6),
                           getPropertyByIndex(7), getPropertyNameByIndex(7)
                           );

*/

}


template <typename T>
unsigned int GPropertyMap<T>::getNumProperties() const 
{

   if(m_prop.size() != m_keys.size())
      LOG(fatal) << "GPropertyMap<T>::getNumProperties"
                 << " prop/keys mismatch "
                 << " prop " << m_prop.size()
                 << " keys " << m_keys.size()
                 ; 

   assert(m_prop.size() == m_keys.size()); // maybe a duplicated key can trigger this
   return m_prop.size();
}

template <typename T>
std::string GPropertyMap<T>::getKeysString() const 
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
void GPropertyMap<T>::addMapAsis(GPropertyMap<T>* other, const char* prefix)
{
    unsigned int n = other->getNumProperties();
    for(unsigned int i=0 ; i<n ; i++)
    {
         const char* name  = other->getPropertyNameByIndex(i); 
         GProperty<T>* prop = other->getPropertyByIndex(i); 

         addPropertyAsis( name, prop, prefix );
    }
}

template <typename T>
void GPropertyMap<T>::addMapStandardized(GPropertyMap<T>* other, const char* prefix)
{
    unsigned int n = other->getNumProperties();
    for(unsigned int i=0 ; i<n ; i++)
    {
         const char* name  = other->getPropertyNameByIndex(i); 
         GProperty<T>* prop = other->getPropertyByIndex(i); 

         addPropertyStandardized( name, prop, prefix );
    }
}

template <typename T>
void GPropertyMap<T>::save(const char* dir)
{
    std::string shortname = m_shortname ; 
    if(m_energy_domain) shortname += ".en " ;  

    for(std::vector<std::string>::iterator it=m_keys.begin() ; it != m_keys.end() ; it++ )
    {
        std::string key = *it ;
        std::string propname(key) ; 
        propname += ".npy" ;

        GProperty<T>* prop = m_prop[key] ; 
        prop->save(dir, shortname.c_str(), propname.c_str());  // dir, reldir, name
    } 
}

template <typename T>
GPropertyMap<T>* GPropertyMap<T>::load(const char* path, const char* name, const char* type)
{
    // path eg $IDPATH/GScintillatorLib
    // name eg GdDopedLS
    // type eg material

    unsigned int index = 0 ; 
    GPropertyMap<T>* pmap = new GPropertyMap<T>(name,  index, type, NULL );
    fs::path dir(path);
    const char* ext = ".npy"  ;
    if(fs::exists(dir) && fs::is_directory(dir))
    {
        dir /= name ; 
        if(fs::exists(dir) && fs::is_directory(dir))
        {    
            std::vector<std::string> basenames ; 
            BDir::dirlist( basenames, dir.string().c_str(), ext);
            for(std::vector<std::string>::const_iterator it=basenames.begin() ; it != basenames.end()   ; it++)
            {
                 std::string propname = *it ;  
                 fs::path pp(dir) ;
                 pp /= propname + ext ;
                 LOG(debug) << "GPropertyMap<T>::load prop " << propname << " pp " << pp.string() ; 

                 GProperty<T>* prop = GProperty<T>::load( pp.string().c_str() );
                 pmap->addPropertyAsis( propname.c_str(), prop );    
            } 
        }
    }
    return pmap ; 
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

template GGEO_API void GPropertyMap<float>::setMetaKV(const char* name, int value);
template GGEO_API void GPropertyMap<float>::setMetaKV(const char* name, std::string value);

template GGEO_API void GPropertyMap<double>::setMetaKV(const char* name, int value);
template GGEO_API void GPropertyMap<double>::setMetaKV(const char* name, std::string value);

template GGEO_API int         GPropertyMap<float>::getMetaKV(const char* name, const char* fallback) const ;
template GGEO_API std::string GPropertyMap<float>::getMetaKV(const char* name, const char* fallback) const ;

template GGEO_API int         GPropertyMap<double>::getMetaKV(const char* name, const char* fallback) const ;
template GGEO_API std::string GPropertyMap<double>::getMetaKV(const char* name, const char* fallback) const ;


