#include <string>
#include <iostream>
#include <iomanip>

#include "BFile.hh"
#include "BStr.hh"
#include "BMap.hh"

#include "Map.hh"

#include "PLOG.hh"


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
    if(!BFile::ExistsFile(dir, name))   
    {
       LOG(debug) << "Map<K,V>::load no such path : dir " << dir << " name " << name  ;
       return NULL ;  
    }
    Map* m = new Map<K,V>() ; 
    m->loadFromCache(dir, name);
    return m ; 
}

template <typename K, typename V>
Map<K,V>* Map<K,V>::load(const char* path)
{
    LOG(trace) << " path " << path ; 
    if(!BFile::ExistsFile(path))
    {
       LOG(trace) << " no path " << path ;
       return NULL ;  
    }
    else
    {
       LOG(trace) << " Map<K,V>::load path " << path ;
    }
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
void Map<K,V>::save(const char* path)
{
    BMap<K, V>::save( &m_map, path);
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


template <typename K, typename V>
Map<K,V>* Map<K,V>::makeSelection(const char* prefix, char delim)
{
    std::vector<std::string> elem ;  
    BStr::split(elem, prefix, delim );
    size_t npfx = elem.size();
    // multiple selection via delimited prefix 

    Map* m = new Map<K,V>() ; 
    typedef std::map<K, V> MKV ; 
    for(typename MKV::iterator it=m_map.begin() ; it != m_map.end() ; it++ ) 
    {
        K k = it->first ; 
        V v = it->second ; 

        bool select = false ; 
        for(size_t p=0 ; p < npfx ; p++)
        {
            std::string pfx = elem[p];
            if(k.find(pfx.c_str()) == 0) select = true ;
            // restricts keys to std::string 
        }
        if(select) m->add(k,v);
    }
    return m ; 
}


template class Map<std::string, unsigned int>;
template class Map<std::string, std::string>;
 
