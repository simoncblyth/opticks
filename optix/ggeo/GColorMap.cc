#include <cassert>
#include <iostream>
#include <iomanip>

#include "BLog.hh"
#include "BFile.hh"
#include "BMap.hh"

#include "GColorMap.hh"


GColorMap* GColorMap::load(const char* dir, const char* name)
{
    assert(0);

    if(!BFile::existsPath(dir, name))
    {
        LOG(warning) << "GColorMap::load FAILED no file at  " << dir << "/" << name ; 
        return NULL ;
    }
    GColorMap* cm = new GColorMap ; 
    cm->loadMaps(dir, name);
    return cm ; 
}

void GColorMap::loadMaps(const char* idpath, const char* name)
{
    BMap<std::string, std::string>::load( &m_iname2color, idpath, name );  
}

void GColorMap::dump(const char* msg)
{
    LOG(info) << msg ;
    typedef std::map<std::string, std::string> MSS ; 
    for(MSS::iterator it=m_iname2color.begin() ; it != m_iname2color.end() ; it++ ) 
    {
         std::cout 
             << std::setw(25) << it->first 
             << std::setw(25) << it->second
             << std::endl ;  
    }
}




