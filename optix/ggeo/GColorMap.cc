#include "GColorMap.hh"
#include "jsonutil.hpp"

#include <cassert>
#include <iostream>
#include <iomanip>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


GColorMap* GColorMap::load(const char* dir, const char* name)
{
    assert(0);

    if(!existsPath(dir, name))
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
    loadMap<std::string, std::string>( m_iname2color, idpath, name );  
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




