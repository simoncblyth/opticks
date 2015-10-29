#include "Map.hpp"

#include <iostream>
#include <iomanip>

#include "jsonutil.hpp"
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



Map* Map::load(const char* dir, const char* name)
{
    Map* m = new Map ; 
    m->loadFromCache(dir, name);
    return m ; 
}

void Map::loadFromCache(const char* dir, const char* name )
{
    loadMap<std::string, unsigned int>( m_map, dir, name );  
}

void Map::add(const char* name, unsigned int value)
{
    m_map[name] = value ; 
}

void Map::save(const char* dir, const char* name)
{
    saveMap<std::string, unsigned int>( m_map, dir, name);
}

void Map::dump(const char* msg)
{
    LOG(info) << msg ; 
    typedef std::map<std::string, unsigned int> MSU ; 
    for(MSU::iterator it=m_map.begin() ; it != m_map.end() ; it++ ) 
    {
        std::cout << std::setw(5) << it->second 
                  << std::setw(30) << it->first 
                  << std::endl ; 
    }    
}




 
