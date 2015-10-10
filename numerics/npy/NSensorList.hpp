#pragma once

#include <cassert>
#include <vector>
#include <set>
#include <string>
#include <map>

class NSensor ; 

// TODO: migrate to npy-

class NSensorList {
    public:
        NSensorList();
        void load(const char* idpath, const char* ext="idmap");
        unsigned int getNumSensors();
        void dump(const char* msg="NSensorList::dump");
        std::string description();

    public:
        NSensor* getSensor(unsigned int index);
        NSensor* findSensorForNode(unsigned int nodeIndex);
    private:
        void read(const char* path);
        void add(NSensor* sensor);
        NSensor* createSensor(std::vector<std::string>& elem );
        unsigned int parseHexString(std::string& str);

    private:
        std::vector<NSensor*>    m_sensors ; 
        std::set<unsigned int>   m_ids ; 
        std::map<unsigned int, NSensor*>   m_nid2sen ; 

};


inline NSensorList::NSensorList()
{
}
inline NSensor* NSensorList::getSensor(unsigned int index)
{
    //assert(0);
    return index < m_sensors.size() ? m_sensors[index] : NULL ; 
}
inline unsigned int NSensorList::getNumSensors()
{
    return m_sensors.size();
}
