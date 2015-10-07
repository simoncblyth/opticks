#pragma once

#include <vector>
#include <set>
#include <string>
#include <map>

class GSensor ; 

class GSensorList {
    public:
        GSensorList();
        void load(const char* idpath, const char* ext="idmap");
        unsigned int getNumSensors();
        void dump(const char* msg="GSensorList::dump");
        std::string description();

    public:
        GSensor* getSensor(unsigned int index);
        GSensor* findSensorForNode(unsigned int nodeIndex);
    private:
        void read(const char* path);
        void add(GSensor* sensor);
        GSensor* createSensor(std::vector<std::string>& elem );
        unsigned int parseHexString(std::string& str);

    private:
        std::vector<GSensor*>    m_sensors ; 
        std::set<unsigned int>   m_ids ; 
        std::map<unsigned int, GSensor*>   m_nid2sen ; 

};


inline GSensorList::GSensorList()
{
}
inline GSensor* GSensorList::getSensor(unsigned int index)
{
    return index < m_sensors.size() ? m_sensors[index] : NULL ; 
}
inline unsigned int GSensorList::getNumSensors()
{
    return m_sensors.size();
}
