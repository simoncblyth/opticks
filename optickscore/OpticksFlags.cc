#include "OpticksFlags.hh"

#include <map>
#include <vector>
#include <string>

#include "Index.hpp"
#include "OpticksAttrSeq.hh"
#include "Opticks.hh"

#include "regexsearch.hh"
#include "NLog.hpp"


//const char* OpticksFlags::ENUM_HEADER_PATH = "$ENV_HOME/graphics/optixrap/cu/photon.h" ;
//const char* OpticksFlags::ENUM_HEADER_PATH = "$ENV_HOME/opticks/OpticksPhoton.h" ;
const char* OpticksFlags::ENUM_HEADER_PATH = "$ENV_HOME/optickscore/OpticksPhoton.h" ;

void OpticksFlags::init(const char* path)
{
    m_index = parseFlags(path);
    unsigned int num_flags = m_index ? m_index->getNumItems() : 0 ;

    LOG(info) << "OpticksFlags::init"
              << " path " << path 
              << " num_flags " << num_flags 
              << " " << ( m_index ? m_index->description() : "NULL index" )
              ;
    
    assert(num_flags > 0 && "missing envvar ENV_HOME or you need to update ENUM_HEADER_PATH ");

    m_aindex = new OpticksAttrSeq(m_cache, "GFlags");
    m_aindex->loadPrefs(); // color, abbrev and order 
    m_aindex->setSequence(m_index);
}

void OpticksFlags::save(const char* idpath)
{
    m_index->setExt(".ini"); 
    m_index->save(idpath);    
}

Index* OpticksFlags::parseFlags(const char* path)
{
    typedef std::pair<unsigned int, std::string>  upair_t ;
    typedef std::vector<upair_t>                  upairs_t ;
    upairs_t ups ;
    enum_regexsearch( ups, path ); 

    Index* index = new Index("GFlags");
    for(unsigned int i=0 ; i < ups.size() ; i++)
    {
        upair_t p = ups[i];
        unsigned int mask = p.first ;
        unsigned int bitpos = ffs(mask);  // first set bit, 1-based bit position
        unsigned int xmask = 1 << (bitpos-1) ; 
        assert( mask == xmask);
        index->add( p.second.c_str(), bitpos );
    }
    return index ; 
}

std::map<unsigned int, std::string> OpticksFlags::getNamesMap()
{
    return m_aindex->getNamesMap() ; 
}

