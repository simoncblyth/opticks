#include <string>
#include <iostream>
#include <iomanip>

#include "BFile.hh"
#include "BMap.hh"
#include "BLog.hh"

#include "Map.hpp"


template <typename K, typename V>
Map<K,V>::Map()
{
}

template <typename K, typename V>
std::map<K, V>& Map<K,V>::getMap()
{
    return m_map ; 
}


template <typename K, typename V>
Map<K,V>* Map<K,V>::load(const char* dir, const char* name)
{
    if(!BFile::existsPath(dir, name)) return NULL ;  
    Map* m = new Map<K,V>() ; 
    m->loadFromCache(dir, name);
    return m ; 
}

template <typename K, typename V>
Map<K,V>* Map<K,V>::load(const char* path)
{
    if(!BFile::existsPath(path)) return NULL ;  
    Map* m = new Map<K,V>() ; 
    m->loadFromCache(path);
    return m ; 
}


template <typename K, typename V>
void Map<K,V>::loadFromCache(const char* dir, const char* name )
{
    BMap<K, V>::load( &m_map, dir, name );  
}

template <typename K, typename V>
void Map<K,V>::loadFromCache(const char* path )
{
    BMap<K, V>::load( &m_map, path );  
}


template <typename K, typename V>
void Map<K,V>::add(K key, V value)
{
    m_map[key] = value ; 
}

template <typename K, typename V>
void Map<K,V>::save(const char* dir, const char* name)
{
    BMap<K, V>::save( &m_map, dir, name);
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
 
