#include "GPropertyLib.hh"
#include "GCache.hh"
#include "GColors.hh"
#include "GItemList.hh"

// npy-
#include "NPY.hpp"
#include "Map.hpp"

#include <cassert>
#include <climits>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <boost/algorithm/string.hpp>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



float        GPropertyLib::DOMAIN_LOW  = 60.f ; 
float        GPropertyLib::DOMAIN_HIGH = 810.f ; 
float        GPropertyLib::DOMAIN_STEP = 20.f ; 
unsigned int GPropertyLib::DOMAIN_LENGTH = 39  ; 

unsigned int GPropertyLib::UNSET = UINT_MAX ; 
unsigned int GPropertyLib::NUM_QUAD = 4  ; 
unsigned int GPropertyLib::NUM_PROP = 4  ; 

std::string GPropertyLib::getCacheDir()
{
    return m_cache->getPropertyLibDir(m_type);
}
std::string GPropertyLib::getPreferenceDir()
{
    return m_cache->getPreferenceDir(m_type);
}



//  GItemIndex approach allows index customization ?
//  eg to put common materials at low indices for truncated compression
//  to retain more info 
//
//
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

const char* GPropertyLib::getName(unsigned int index)
{
    assert(m_names);
    std::string& item = m_names->getItem(index);
    return item.empty() ? NULL : item.c_str(); 
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

    initOrder();
    initColor();
}


void GPropertyLib::initOrder()
{
    std::string prefdir = getPreferenceDir();
    const char* order_ = "order.json" ; 
    typedef Map<std::string, unsigned int> MSU ;  
    MSU* order = MSU::load(prefdir.c_str(), order_ ) ; 
    if(order)
    {
        order->dump("GPropertyLib::initOrder");
        setOrder(order->getMap());
    }
}

void GPropertyLib::initColor()
{
    std::string prefdir = getPreferenceDir();
    const char* color_ = "color.json" ; 
    typedef Map<std::string, std::string> MSS ;  
    MSS* color = MSS::load(prefdir.c_str(), color_ ) ; 
    if(color)
    {
        color->dump("GPropertyLib::initColor");
        setColor(color->getMap());
    }
}


const char* GPropertyLib::getColorName(const char* key)
{
    return m_color.count(key) == 1 ? m_color[key].c_str() : NULL ; 
}

unsigned int GPropertyLib::getColorCode(const char* key )
{
    const char*  colorname =  getColorName(key) ;
    GColors* palette = m_cache->getColors();
    unsigned int colorcode  = palette->getCode(colorname, 0xFFFFFF) ; 
    return colorcode ; 
}


std::vector<unsigned int>& GPropertyLib::getColorCodes()
{
    if(m_color_codes.size() == 0)
    {
        unsigned int ni = m_names->getNumItems();
        for(unsigned int i=0 ; i < ni ; i++)
        {
            std::string& item = m_names->getItem(i);
            assert(!item.empty()); 
            unsigned int code = getColorCode(item.c_str());
            m_color_codes.push_back(code);
        }         
    }
    return m_color_codes ; 
}


void GPropertyLib::dumpItems(const char* items, const char* msg)
{
    typedef std::vector<std::string> VS ; 
    VS elem ; 
    boost::split(elem, items, boost::is_any_of(","));

    LOG(info) << msg << " " << items ; 
    for(VS::const_iterator it=elem.begin() ; it != elem.end() ; it++)
    {
        const char* key = it->c_str();
        unsigned int idx = getIndex(key);
        if(idx == GPropertyLib::UNSET)
        {
             LOG(warning) << "GPropertyLib::dump no item named: " << *it ; 
        }
        else
        {
             const char* colorname = getColorName(key);  
             unsigned int colorcode = getColorCode(key);              

             std::cout << std::setw(5) << idx 
                       << std::setw(30) << *it 
                       << std::setw(10) << std::hex << colorcode << std::dec
                       << std::setw(15) << colorname 
                       << std::endl ; 
        }
    }
}


std::string GPropertyLib::getBufferName(const char* suffix)
{
    std::string name = m_type ;  
    if(suffix) name += suffix ; 
    return name + ".npy" ; 
}

void GPropertyLib::close()
{
    sort();

    GItemList* names = createNames();
    NPY<float>* buf = createBuffer() ;

    setNames(names);
    setBuffer(buf);
    setClosed();
}

void GPropertyLib::saveToCache(NPYBase* buffer, const char* suffix)
{
    assert(suffix);
    std::string dir = getCacheDir(); 
    std::string name = getBufferName(suffix);
    buffer->save(dir.c_str(), name.c_str());   
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




NPY<unsigned int>* GPropertyLib::createUint4Buffer(std::vector<guint4>& vec)
{
    unsigned int ni = vec.size() ;
    unsigned int nj = 4 ;  
    NPY<unsigned int>* ibuf = NPY<unsigned int>::make( ni, nj) ;
    ibuf->zero();
    unsigned int* idat = ibuf->getValues();
    for(unsigned int i=0 ; i < ni ; i++)     
    {
        const guint4& entry = vec[i] ;
        for(unsigned int j=0 ; j < nj ; j++) idat[i*nj+j] = entry[j] ;  
    }
    return ibuf ; 
}


void GPropertyLib::importUint4Buffer(std::vector<guint4>& vec, NPY<unsigned int>* ibuf )
{
    LOG(info) << "GPropertyLib::importUint4Buffer" ; 

    unsigned int* idat = ibuf->getValues();
    unsigned int ni = ibuf->getShape(0);
    unsigned int nj = ibuf->getShape(1);
    assert(nj == 4); 

    for(unsigned int i=0 ; i < ni ; i++)
    {
        guint4 entry(UNSET,UNSET,UNSET,UNSET);

        entry.x = idat[i*nj+0] ;
        entry.y = idat[i*nj+1] ;
        entry.z = idat[i*nj+2] ;
        entry.w = idat[i*nj+3] ;

        vec.push_back(entry);
    }
}


