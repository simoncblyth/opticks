#pragma once

#include <string>
#include <map>

class Map {
    public:
        Map();
        static Map* load(const char* dir, const char* name);     
        void loadFromCache(const char* dir, const char* name);
        void add(const char* name, unsigned int value);
        void save(const char* dir, const char* name);
        void dump(const char* msg="Map::dump");
        std::map<std::string, unsigned int>& getMap(); 
    private:
        std::map<std::string, unsigned int> m_map ; 

};


inline Map::Map()
{
}

inline std::map<std::string, unsigned int>& Map::getMap()
{
    return m_map ; 
}

