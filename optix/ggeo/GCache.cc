#include "GCache.hh"
#include "OpticksFlags.hh"

#include <sstream>
#include <cassert>
#include <cstdio>

//opticks-
#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksColors.hh"

// npy-
#include "NLog.hpp"
#include "Types.hpp"
#include "Typ.hpp"


GCache* GCache::g_instance = NULL ; 


void GCache::init()
{
    m_resource = m_opticks->getResource();
    assert(g_instance == NULL && "GCache::init only one instance is allowed");
    g_instance = this ; 

}

OpticksQuery* GCache::getQuery()
{
    return m_resource ? m_resource->getQuery() : NULL  ; 
}


// lazy constituent construction : as want to avoid any output until after logging is configured

OpticksColors* GCache::getColors()
{
    return m_resource->getColors() ;
}

OpticksFlags* GCache::getFlags()
{
    return m_resource->getFlags() ;
}





Typ* GCache::getTyp()
{
    if(m_typ == NULL)
    {
       m_typ = new Typ ; 
    }
    return m_typ ; 
}


Types* GCache::getTypes()
{
    if(!m_types)
    {
        // deferred because idpath not known at init ?
        m_types = new Types ;  
        m_types->saveFlags(getIdPath(), ".ini");
    }
    return m_types ;
}









const char* GCache::getGDMLPath()
{
    return m_resource->getGDMLPath();
}
const char* GCache::getDAEPath()
{
    return m_resource->getDAEPath();
}
const char* GCache::getIdPath()
{
    return m_resource->getIdPath();
}

const char* GCache::getIdFold()
{
    return m_resource->getIdFold();
}
std::string GCache::getRelativePath(const char* path)
{
    return m_resource->getRelativePath(path);
}
int GCache::getLastArgInt()
{
    return m_opticks->getLastArgInt();
}
const char* GCache::getLastArg()
{
    return m_opticks->getLastArg();
}


void GCache::Summary(const char* msg)
{
    printf("%s \n", msg );
}

