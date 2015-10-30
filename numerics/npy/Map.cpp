#include "Map.hpp"

#include <string>
#include <iostream>
#include <iomanip>

#include "jsonutil.hpp"
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

template <typename K, typename V>
Map<K,V>* Map<K,V>::load(const char* dir, const char* name)
{
    if(!existsPath(dir, name)) return NULL ;  
    Map* m = new Map<K,V>() ; 
    m->loadFromCache(dir, name);
    return m ; 
}

template <typename K, typename V>
void Map<K,V>::loadFromCache(const char* dir, const char* name )
{
    loadMap<K, V>( m_map, dir, name );  
}

template <typename K, typename V>
void Map<K,V>::add(K key, V value)
{
    m_map[key] = value ; 
}

template <typename K, typename V>
void Map<K,V>::save(const char* dir, const char* name)
{
    saveMap<K, V>( m_map, dir, name);
}

template <typename K, typename V>
void Map<K,V>::dump(const char* msg)
{
    LOG(info) << msg ; 
    typedef std::map<K, V> MKV ; 
    for(typename MKV::iterator it=m_map.begin() ; it != m_map.end() ; it++ ) 
    {
        std::cout << std::setw(5) << it->second 
                  << std::setw(30) << it->first 
                  << std::endl ; 
    }    
}


template class Map<std::string, unsigned int>;
template class Map<std::string, std::string>;
 
