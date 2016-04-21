#pragma once

#include <string>
#include <map>


template <typename K, typename V>
class Map {
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


template <typename K, typename V>
inline Map<K,V>::Map()
{
}

template <typename K, typename V>
inline std::map<K, V>& Map<K,V>::getMap()
{
    return m_map ; 
}

