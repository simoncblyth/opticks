#pragma once

#include <string>
#include <map>


#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

// TODO: merge Map with BMap that is uses ?

template <typename K, typename V> class Map ; 

template <typename K, typename V>
class BRAP_API Map {
    public:
        Map();

        static Map<K,V>* load(const char* dir, const char* name);     
        static Map<K,V>* load(const char* path);     

        void loadFromCache(const char* dir, const char* name);
        void loadFromCache(const char* path);

        void add(K key, V value);
    
         Map<K,V>* makeSelection(const char* prefix, char delim=',');

        void save(const char* dir, const char* name);
        void save(const char* path);

        void dump(const char* msg="Map::dump");
        std::map<K, V>& getMap(); 
    private:
        std::map<K, V> m_map ; 

};

#include "BRAP_TAIL.hh"


