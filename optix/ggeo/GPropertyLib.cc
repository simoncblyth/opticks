#include <cassert>
#include <cstring>
#include <climits>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <boost/algorithm/string.hpp>

// brap-
#include "BDir.hh"

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"
#include "Map.hpp"

// optickscore-
#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksAttrSeq.hh"

// ggeo-
#include "GVector.hh"
#include "GDomain.hh"
#include "GItemList.hh"
#include "GProperty.hh"
#include "GPropertyMap.hh"
#include "GPropertyLib.hh"

#include "PLOG.hh"


unsigned int GPropertyLib::UNSET = UINT_MAX ; 
unsigned int GPropertyLib::NUM_MATSUR = BOUNDARY_NUM_MATSUR  ;    // 4 material/surfaces that comprise a boundary om-os-is-im 
unsigned int GPropertyLib::NUM_PROP = BOUNDARY_NUM_PROP  ; 
unsigned int GPropertyLib::NUM_FLOAT4 = BOUNDARY_NUM_FLOAT4  ; 

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

const char* GPropertyLib::material = "material" ; 
const char* GPropertyLib::surface  = "surface" ; 
const char* GPropertyLib::source   = "source" ; 
const char* GPropertyLib::bnd_     = "bnd" ; 







GPropertyLib::GPropertyLib(Opticks* cache, const char* type) 
     :
     m_cache(cache),
     m_resource(NULL),
     m_buffer(NULL),
     m_attrnames(NULL),
     m_names(NULL),
     m_type(strdup(type)),
     m_comptype(NULL),
     m_standard_domain(NULL),
     m_defaults(NULL),
     m_closed(false),
     m_valid(true)
{
     init();
}

const char* GPropertyLib::getType()
{
    return m_type ; 
}

const char* GPropertyLib::getComponentType()
{
    return m_comptype ; 
}

GPropertyLib::~GPropertyLib()
{
}

GDomain<float>* GPropertyLib::getStandardDomain()
{
    return m_standard_domain ;
}

/*
inline void GPropertyLib::setOrder(std::map<std::string, unsigned int>& order)
{
    m_order = order ; 
}
*/

GPropertyMap<float>* GPropertyLib::getDefaults()
{
    return m_defaults ;
}

void GPropertyLib::setBuffer(NPY<float>* buf)
{
    m_buffer = buf ;
}
NPY<float>* GPropertyLib::getBuffer()
{
    return m_buffer ;
}

GItemList* GPropertyLib::getNames()
{
    return m_names ;
}
OpticksAttrSeq* GPropertyLib::getAttrNames()
{
    return m_attrnames ;
}


void GPropertyLib::setClosed(bool closed)
{
    m_closed = closed ; 
}
bool GPropertyLib::isClosed()
{
    return m_closed ; 
}

void GPropertyLib::setValid(bool valid)
{
    m_valid = valid ; 
}
bool GPropertyLib::isValid()
{
    return m_valid ; 
}

unsigned int GPropertyLib::getNumRaw()
{
    return m_raw.size();
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

    m_attrnames = new OpticksAttrSeq(m_cache, m_type);
    m_attrnames->loadPrefs(); // color, abbrev and order 


    // hmm GPropertyMap expects bordersurface or skinsurface

    if(     strcmp(m_type, "GMaterialLib")==0)      m_comptype=material ;
    else if(strcmp(m_type, "GScintillatorLib")==0)  m_comptype=material ;
    else if(strcmp(m_type, "GSurfaceLib")==0)       m_comptype=surface ;
    else if(strcmp(m_type, "GSourceLib")==0)        m_comptype=source ;
    else if(strcmp(m_type, "GBndLib")==0)           m_comptype=bnd_ ;
    else                                            m_comptype=NULL  ;

    if(!m_comptype)
    {
        LOG(fatal) << "GPropertyLib::init " << m_type ;  
        assert( 0 &&  "unexpected GPropertyLib type");
    }    


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
        LOG(info) << "GPropertyLib::getIndex type " << m_type 
                     << " TRIGGERED A CLOSE " 
                     << " shortname [" << ( shortname ? shortname : "" ) << "]"  
                     ;

        //assert(0);

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
    LOG(trace) << "GPropertyLib::close" ;

    sort();
    LOG(trace) << "GPropertyLib::close after sort " ;

    GItemList* names = createNames();
    LOG(trace) << "GPropertyLib::close " 
               << " names " << names 
               ;


    NPY<float>* buf = createBuffer() ;

    LOG(info) << "GPropertyLib::close"
              << " type " << m_type 
              << " buf " <<  ( buf ? buf->getShapeString() : "NULL" )
              ; 

    //names->dump("GPropertyLib::close") ;

    setNames(names);
    setBuffer(buf);
    setClosed();

    LOG(trace) << "GPropertyLib::close DONE" ;
}

void GPropertyLib::saveToCache(NPYBase* buffer, const char* suffix)
{
    assert(suffix);
    std::string dir = getCacheDir(); 
    std::string name = getBufferName(suffix);

    if(buffer)
    {
        buffer->save(dir.c_str(), name.c_str());   
    }
    else
    {
        LOG(error) << "GPropertyLib::saveToCache"
                   << " NULL BUFFER "
                   << " dir " << dir
                   << " name " << name
                   ; 
    }
}

void GPropertyLib::saveToCache()
{

    LOG(trace) << "GPropertyLib::saveToCache" ; 
 

    if(!isClosed()) close();

    if(m_buffer)
    {
        std::string dir = getCacheDir(); 
        std::string name = getBufferName();
        m_buffer->save(dir.c_str(), name.c_str());   
    }

    if(m_names)
    {
        m_names->save(m_resource->getIdPath());
    }

    LOG(trace) << "GPropertyLib::saveToCache DONE" ; 

}

void GPropertyLib::loadFromCache()
{
    LOG(trace) << "GPropertyLib::loadFromCache" ;

    std::string dir = getCacheDir(); 
    std::string name = getBufferName();

    LOG(trace) << "GPropertyLib::loadFromCache" 
              << " dir " << dir
              << " name " << name 
               ;

  
    NPY<float>* buf = NPY<float>::load(dir.c_str(), name.c_str()); 

    setBuffer(buf); 

    GItemList* names = GItemList::load(m_resource->getIdPath(), m_type);
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



void GPropertyLib::addRaw(GPropertyMap<float>* pmap)
{
    m_raw.push_back(pmap);
}

GPropertyMap<float>* GPropertyLib::getRaw(unsigned int index)
{
    return index < m_raw.size() ? m_raw[index] : NULL ;
}

GPropertyMap<float>* GPropertyLib::getRaw(const char* shortname)
{
    unsigned int nraw = m_raw.size();
    for(unsigned int i=0 ; i < nraw ; i++)
    {  
        GPropertyMap<float>* pmap = m_raw[i];
        const char* name = pmap->getShortName();
        if(strcmp(shortname, name) == 0) return pmap ;         
    }
    return NULL ; 
}



void GPropertyLib::loadRaw()
{
    std::string dir = getCacheDir();   // eg $IDPATH/GScintillatorLib

    std::vector<std::string> names ; 
    BDir::dirdirlist(names, dir.c_str() );   // find sub-directory names for all raw items in lib eg GdDopedLS,LiquidScintillator
   
    for(std::vector<std::string>::iterator it=names.begin() ; it != names.end() ; it++ )
    {
        std::string name = *it ; 
        LOG(debug) << "GPropertyLib::loadRaw " << name << " " << m_comptype ; 

        GPropertyMap<float>* pmap = GPropertyMap<float>::load( dir.c_str(), name.c_str(), m_comptype );
        if(pmap)
        {
            LOG(debug) << "GPropertyLib::loadRaw " << name << " " << m_comptype << " num properties:" << pmap->getNumProperties() ; 
            addRaw(pmap);
        }

    }
}

void GPropertyLib::dumpRaw(const char* msg)
{
    LOG(info) << msg ; 
    unsigned int nraw = m_raw.size();
    for(unsigned int i=0 ; i < nraw ; i++)
    {
        GPropertyMap<float>* pmap = m_raw[i] ;
        LOG(info) << " component " << pmap->getName() ;
        LOG(info) << " table " << pmap->make_table() ;
    }
}


 
void GPropertyLib::saveRaw()
{
    std::string dir = getCacheDir(); 
    unsigned int nraw = m_raw.size();
    for(unsigned int i=0 ; i < nraw ; i++)
    {
        GPropertyMap<float>* pmap = m_raw[i] ;
        pmap->save(dir.c_str());
    }
}


