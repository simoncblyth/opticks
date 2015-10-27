#include "GPropertyLib.hh"
#include "GCache.hh"
#include "GItemList.hh"

#include "NPY.hpp"
#include <cassert>

#include <iostream>
#include <sstream>
#include <iomanip>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



float        GPropertyLib::DOMAIN_LOW  = 60.f ; 
float        GPropertyLib::DOMAIN_HIGH = 810.f ; 
float        GPropertyLib::DOMAIN_STEP = 20.f ; 
unsigned int GPropertyLib::DOMAIN_LENGTH = 39  ; 

std::string GPropertyLib::getCacheDir()
{
    return m_cache->getPropertyLibDir(m_type);
}

unsigned int GPropertyLib::getIndex(const char* shortname)
{
    if(!isClosed())
    {
        LOG(warning) << "GPropertyLib::getIndex type " << m_type 
                     << " TRIGGERED A CLOSE " << ( shortname ? shortname : "" ) ;
        close();
    }
    assert(m_names);
    return m_names->getIndex(shortname);
}

void GPropertyLib::init()
{
    GDomain<float>* domain = new GDomain<float>(DOMAIN_LOW, DOMAIN_HIGH, DOMAIN_STEP ); 
    setStandardDomain( domain );

    assert(getStandardDomainLength() == DOMAIN_LENGTH );
    assert(m_standard_domain->getLow()  == DOMAIN_LOW );
    assert(m_standard_domain->getHigh() == DOMAIN_HIGH );
    assert(m_standard_domain->getStep() == DOMAIN_STEP );

    GPropertyMap<float>* defaults = new GPropertyMap<float>("defaults", UINT_MAX, "defaults");
    defaults->setStandardDomain(getStandardDomain());
    setDefaults(defaults);
}

std::string GPropertyLib::getBufferName()
{
    std::string name = m_type ;  
    return name + ".npy" ; 
}

void GPropertyLib::close()
{
    GItemList* names = createNames();
    NPY<float>* buf = createBuffer() ;

    setNames(names);
    setBuffer(buf);
    setClosed();
}

void GPropertyLib::saveToCache()
{
    if(!isClosed()) close();

    assert(m_buffer);
    assert(m_names);

    std::string dir = getCacheDir(); 
    std::string name = getBufferName();

    m_buffer->save(dir.c_str(), name.c_str());   
    m_names->save(m_cache->getIdPath());
}

void GPropertyLib::loadFromCache()
{
    std::string dir = getCacheDir(); 
    std::string name = getBufferName();

    setBuffer(NPY<float>::load(dir.c_str(), name.c_str())); 
    setNames(GItemList::load(m_cache->getIdPath(), m_type)); 

    import();
}



GDomain<float>* GPropertyLib::getDefaultDomain()
{
   return new GDomain<float>(DOMAIN_LOW, DOMAIN_HIGH, DOMAIN_STEP ); 
}

unsigned int GPropertyLib::getStandardDomainLength()
{
    return m_standard_domain ? m_standard_domain->getLength() : 0 ;
}

GProperty<float>* GPropertyLib::getDefaultProperty(const char* name)
{
    return m_defaults ? m_defaults->getProperty(name) : NULL ;
}


GProperty<float>* GPropertyLib::makeConstantProperty(float value)
{
    GProperty<float>* prop = GProperty<float>::from_constant( value, m_standard_domain->getValues(), m_standard_domain->getLength() );
    return prop ; 
}

GProperty<float>* GPropertyLib::makeRampProperty()
{
   return GProperty<float>::ramp( m_standard_domain->getLow(), m_standard_domain->getStep(), m_standard_domain->getValues(), m_standard_domain->getLength() );
}


GProperty<float>* GPropertyLib::getProperty(GPropertyMap<float>* pmap, const char* dkey)
{
    assert(pmap);

    const char* lkey = getLocalKey(dkey); assert(lkey);  // missing local key mapping 

    GProperty<float>* prop = pmap->getProperty(lkey) ;

    //assert(prop);
    //if(!prop) LOG(warning) << "GPropertyLib::getProperty failed to find property " << dkey << "/" << lkey ;

    return prop ;  
}



GProperty<float>* GPropertyLib::getPropertyOrDefault(GPropertyMap<float>* pmap, const char* dkey)
{
    // convert destination key such as "detect" into local key "EFFICIENCY" 

    const char* lkey = getLocalKey(dkey); assert(lkey);  // missing local key mapping 

    GProperty<float>* fallback = getDefaultProperty(dkey);  assert(fallback);

    GProperty<float>* prop = pmap ? pmap->getProperty(lkey) : NULL ;

    return prop ? prop : fallback ;
}



const char* GPropertyLib::getLocalKey(const char* dkey) // mapping between standard keynames and local key names, eg refractive_index -> RINDEX
{
    return m_keymap[dkey].c_str();
}

void GPropertyLib::setKeyMap(const char* spec)
{
    m_keymap.clear();

    char delim = ',' ;
    std::istringstream f(spec);
    std::string s;
    while (getline(f, s, delim)) 
    {
        std::size_t colon = s.find(":");
        if(colon == std::string::npos)
        {
            printf("GPropertyLib::setKeyMap SKIPPING ENTRY WITHOUT COLON %s\n", s.c_str());
            continue ;
        }
        
        std::string dk = s.substr(0, colon);
        std::string lk = s.substr(colon+1);
        //printf("GPropertyLib::setKeyMap dk [%s] lk [%s] \n", dk.c_str(), lk.c_str());
        m_keymap[dk] = lk ; 
    }
}



