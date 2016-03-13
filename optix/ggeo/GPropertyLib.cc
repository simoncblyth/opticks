#include "GPropertyLib.hh"
#include "GCache.hh"
#include "GItemList.hh"
#include "GAttrSeq.hh"

// opticks-
#include "Opticks.hh"
#include "OpticksResource.hh"

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


/*



In [120]: np.arange(60,820.1,20)
Out[120]: 
array([  60.,   80.,  100.,  120.,  140.,  160.,  180.,  200.,  220.,
        240.,  260.,  280.,  300.,  320.,  340.,  360.,  380.,  400.,
        420.,  440.,  460.,  480.,  500.,  520.,  540.,  560.,  580.,
        600.,  620.,  640.,  660.,  680.,  700.,  720.,  740.,  760.,
        780.,  800.,  820.])

In [121]: np.linspace(60,820,39)
Out[121]: 
array([  60.,   80.,  100.,  120.,  140.,  160.,  180.,  200.,  220.,
        240.,  260.,  280.,  300.,  320.,  340.,  360.,  380.,  400.,
        420.,  440.,  460.,  480.,  500.,  520.,  540.,  560.,  580.,
        600.,  620.,  640.,  660.,  680.,  700.,  720.,  740.,  760.,
        780.,  800.,  820.])



TODO: try moving 60nm (extreme UV) up to smth more reasonable like 200nm (far UV)

In [126]: np.linspace(200,800,31)
Out[126]: 
array([ 200.,  220.,  240.,  260.,  280.,  300.,  320.,  340.,  360.,
        380.,  400.,  420.,  440.,  460.,  480.,  500.,  520.,  540.,
        560.,  580.,  600.,  620.,  640.,  660.,  680.,  700.,  720.,
        740.,  760.,  780.,  800.])

In [128]: np.arange(200,800.1,20)
Out[128]: 
array([ 200.,  220.,  240.,  260.,  280.,  300.,  320.,  340.,  360.,
        380.,  400.,  420.,  440.,  460.,  480.,  500.,  520.,  540.,
        560.,  580.,  600.,  620.,  640.,  660.,  680.,  700.,  720.,
        740.,  760.,  780.,  800.])



*/


unsigned int GPropertyLib::UNSET = UINT_MAX ; 
unsigned int GPropertyLib::NUM_QUAD = 4  ;    // not a good name refers to the four species om-os-is-im for which props are stored   
unsigned int GPropertyLib::NUM_PROP = BOUNDARY_NUM_PROP  ; 


void GPropertyLib::checkBufferCompatibility(unsigned int nk, const char* msg)
{

    if(nk != NUM_PROP)
    {
        LOG(fatal) << " GPropertyLib::checkBufferCompatibility "
                   << msg     
                   << " nk " << nk
                   << " NUM_PROP " << NUM_PROP
                   << " loading GPropertyLib with last dimension inconsistent with GPropLib::NUM_PROP " 
                   << " resolve by recreating the geocache, run with -G "
                   ;

    }
    assert(nk == NUM_PROP);
}



GDomain<float>* GPropertyLib::getDefaultDomain()
{
   return new GDomain<float>(Opticks::DOMAIN_LOW, Opticks::DOMAIN_HIGH, Opticks::DOMAIN_STEP ); 
}


void GPropertyLib::init()
{
    m_resource = m_cache->getResource();

    m_standard_domain = getDefaultDomain(); 

    unsigned int len = getStandardDomainLength() ;

    if(len != Opticks::DOMAIN_LENGTH)
    { 
        m_standard_domain->Summary("GPropertyLib::m_standard_domain");
        LOG(fatal) << "GPropertyLib::init"
                   << " mismatch "
                   << " DOMAIN_LENGTH " << Opticks::DOMAIN_LENGTH
                   << " len " << len 
                   ;
    }

    assert(len == Opticks::DOMAIN_LENGTH );

    m_defaults = new GPropertyMap<float>("defaults", UINT_MAX, "defaults");
    m_defaults->setStandardDomain(m_standard_domain);

    m_attrnames = new GAttrSeq(m_cache, m_type);
    m_attrnames->loadPrefs(); // color, abbrev and order 

}

std::map<std::string, unsigned int>& GPropertyLib::getOrder()
{
    return m_attrnames->getOrder() ; 
}

std::map<unsigned int, std::string> GPropertyLib::getNamesMap()
{
    return m_attrnames->getNamesMap() ; 
}



std::string GPropertyLib::getCacheDir()
{
    return m_resource->getPropertyLibDir(m_type);
}
std::string GPropertyLib::getPreferenceDir()
{
    return m_resource->getPreferenceDir(m_type);
}

unsigned int GPropertyLib::getIndex(const char* shortname)
{
    if(!isClosed())
    {
        LOG(debug) << "GPropertyLib::getIndex type " << m_type 
                     << " TRIGGERED A CLOSE " << ( shortname ? shortname : "" ) ;
        close();
    }
    assert(m_names);
    return m_names->getIndex(shortname);
}

const char* GPropertyLib::getName(unsigned int index)
{
    assert(m_names);
    const char* key = m_names->getKey(index);
    return key ; 
}





std::string GPropertyLib::getBufferName(const char* suffix)
{
    std::string name = m_type ;  
    if(suffix) name += suffix ; 
    return name + ".npy" ; 
}

void GPropertyLib::close()
{
    //assert(0);

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

    if(m_buffer)
    {
        std::string dir = getCacheDir(); 
        std::string name = getBufferName();
        m_buffer->save(dir.c_str(), name.c_str());   
    }

    if(m_names)
    {
        m_names->save(m_cache->getIdPath());
    }
}

void GPropertyLib::loadFromCache()
{
    std::string dir = getCacheDir(); 
    std::string name = getBufferName();

    setBuffer(NPY<float>::load(dir.c_str(), name.c_str())); 

    GItemList* names = GItemList::load(m_cache->getIdPath(), m_type);
    setNames(names); 

    import();
}


void GPropertyLib::setNames(GItemList* names)
{
    m_names = names ;
    m_attrnames->setSequence(names);
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




NPY<unsigned int>* GPropertyLib::createUint4Buffer(std::vector<guint4>& vec )
{
    unsigned int ni = vec.size() ;
    unsigned int nj = 4  ; 

 
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
    if(ibuf == NULL)
    {
        LOG(warning) << "GPropertyLib::importUint4Buffer NULL buffer "  ; 
        setValid(false);
        return ; 
    } 

    LOG(debug) << "GPropertyLib::importUint4Buffer" ; 

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


