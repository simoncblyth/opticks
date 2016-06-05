#include "GCache.hh"

#include <sstream>
#include <cassert>
#include <cstdio>

//opticks-
#include "Opticks.hh"
#include "OpticksResource.hh"

// npy-
#include "NLog.hpp"

GCache* GCache::g_instance = NULL ; 

void GCache::init()
{
    m_resource = m_opticks->getResource();
    assert(g_instance == NULL && "GCache::init only one instance is allowed");
    g_instance = this ; 

}

OpticksQuery* GCache::getQuery()
{
    return m_resource->getQuery() ; 
}
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
    return m_resource->getTyp() ;
}
Types* GCache::getTypes()
{
    return m_resource->getTypes() ;
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

