#pragma once

#include <string>
#include <map>


#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"


template <typename K, typename V>
class NPY_API Map {
    public:
        Map();

        static Map<K,V>* load(const char* dir, const char* name);     
        static Map<K,V>* load(const char* path);     

        void loadFromCache(const char* dir, const char* name);
        void loadFromCache(const char* path);

        void add(K key, V value);
        void save(const char* dir, const char* name);
        void dump(const char* msg="Map::dump");
        std::map<K, V>& getMap(); 
    private:
        std::map<K, V> m_map ; 

};

#include "NPY_TAIL.hh"


